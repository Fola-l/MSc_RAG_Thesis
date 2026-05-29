"""
Evaluation script — runs after all four configs have produced their result JSONs.

Retrieval metrics (vs BEIR NQ qrels):  Recall@5, MRR@5, NDCG@5
Generation metrics (via RAGAS):        Faithfulness, Answer Relevancy, Context Precision

Output: results/evaluation_summary.csv
"""

import os
import json
import csv
import time
import math
from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

CONFIG_FILES = {
    "config1_dense"        : os.path.join(RESULTS_DIR, "config1_dense.json"),
    "config2_dense_rerank" : os.path.join(RESULTS_DIR, "config2_dense_rerank.json"),
    "config3_hybrid"       : os.path.join(RESULTS_DIR, "config3_hybrid.json"),
    "config4_hybrid_rerank": os.path.join(RESULTS_DIR, "config4_hybrid_rerank.json"),
}

# ── LOAD RESULT JSONS ─────────────────────────────────────────
all_results = {}
for config_name, path in CONFIG_FILES.items():
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — skipping {config_name}")
        continue
    with open(path) as f:
        all_results[config_name] = json.load(f)
    print(f"Loaded {len(all_results[config_name])} results for {config_name}")

if not all_results:
    raise SystemExit("No result files found. Run all config scripts first.")

# ── LOAD BEIR NQ QRELS ────────────────────────────────────────
print("\nLoading BEIR NQ qrels...")
from datasets import load_dataset

qrels_raw = load_dataset('BeIR/nq', 'qrels', split='test')
qrels = {}
for row in qrels_raw:
    qid = str(row['query-id'])
    did = str(row['corpus-id'])
    if int(row['score']) > 0:
        qrels.setdefault(qid, set()).add(did)

print(f"Loaded qrels for {len(qrels)} queries")

# ── RETRIEVAL METRICS ─────────────────────────────────────────
def recall_at_k(retrieved_ids, relevant_ids, k=5):
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)

def mrr_at_k(retrieved_ids, relevant_ids, k=5):
    for rank, did in enumerate(retrieved_ids[:k], 1):
        if did in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    dcg = 0.0
    for rank, did in enumerate(retrieved_ids[:k], 1):
        if did in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    # ideal DCG: all relevant docs in top-k positions
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0

def compute_retrieval_metrics(results):
    recalls, mrrs, ndcgs = [], [], []
    skipped = 0
    for r in results:
        qid      = str(r['query_id'])
        relevant = qrels.get(qid)
        if not relevant:
            skipped += 1
            continue
        retrieved = [str(d) for d in r['retrieved_ids']]
        recalls.append(recall_at_k(retrieved, relevant))
        mrrs.append(mrr_at_k(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant))
    n = len(recalls)
    if skipped:
        print(f"  ({skipped} queries had no qrels entry — excluded from retrieval metrics)")
    return {
        "recall@5" : round(sum(recalls) / n, 4) if n else 0.0,
        "mrr@5"    : round(sum(mrrs)   / n, 4) if n else 0.0,
        "ndcg@5"   : round(sum(ndcgs)  / n, 4) if n else 0.0,
        "n_eval"   : n,
    }

# ── RAGAS EVALUATION ──────────────────────────────────────────
def compute_ragas_metrics(results, config_name):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"  RAGAS import error: {e}")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

    print(f"  Configuring RAGAS with Groq LLM...")
    groq_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    ))
    hf_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ))

    faithfulness.llm          = groq_llm
    answer_relevancy.llm      = groq_llm
    answer_relevancy.embeddings = hf_embeddings
    context_precision.llm     = groq_llm

    data = {
        "question": [r["query"]    for r in results],
        "answer"  : [r["answer"]   for r in results],
        "contexts": [r["contexts"] for r in results],
    }
    dataset = Dataset.from_dict(data)

    BATCH_SIZE = 50
    n_batches  = math.ceil(len(dataset) / BATCH_SIZE)
    print(f"  Running RAGAS on {len(dataset)} samples in {n_batches} batches of {BATCH_SIZE} (Groq rate-limit safe)...")

    all_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
    try:
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(start, end))

            batch_scores = evaluate(
                batch,
                metrics=[faithfulness, answer_relevancy, context_precision],
            )
            for key in all_scores:
                all_scores[key].append(float(batch_scores[key]))

            print(f"  Batch {batch_idx + 1}/{n_batches} done")
            if batch_idx < n_batches - 1:
                time.sleep(60)  # ~50 calls/batch; pause to respect Groq ~30 req/min

        return {
            "faithfulness"     : round(sum(all_scores["faithfulness"])      / n_batches, 4),
            "answer_relevancy" : round(sum(all_scores["answer_relevancy"])  / n_batches, 4),
            "context_precision": round(sum(all_scores["context_precision"]) / n_batches, 4),
        }
    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

# ── RUN EVERYTHING ────────────────────────────────────────────
summary = []

for config_name, results in all_results.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}  ({len(results)} queries)")
    print(f"{'='*50}")

    print("  Computing retrieval metrics...")
    ret_metrics = compute_retrieval_metrics(results)
    print(f"  Recall@5 : {ret_metrics['recall@5']}")
    print(f"  MRR@5    : {ret_metrics['mrr@5']}")
    print(f"  NDCG@5   : {ret_metrics['ndcg@5']}")

    print("  Computing RAGAS metrics...")
    gen_metrics = compute_ragas_metrics(results, config_name)
    print(f"  Faithfulness      : {gen_metrics['faithfulness']}")
    print(f"  Answer Relevancy  : {gen_metrics['answer_relevancy']}")
    print(f"  Context Precision : {gen_metrics['context_precision']}")

    summary.append({
        "config"            : config_name,
        "n_queries"         : ret_metrics["n_eval"],
        "recall@5"          : ret_metrics["recall@5"],
        "mrr@5"             : ret_metrics["mrr@5"],
        "ndcg@5"            : ret_metrics["ndcg@5"],
        "faithfulness"      : gen_metrics["faithfulness"],
        "answer_relevancy"  : gen_metrics["answer_relevancy"],
        "context_precision" : gen_metrics["context_precision"],
    })

# ── SAVE CSV ──────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
fieldnames = [
    "config", "n_queries",
    "recall@5", "mrr@5", "ndcg@5",
    "faithfulness", "answer_relevancy", "context_precision",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)

print(f"\n{'='*50}")
print(f"✓ Evaluation complete. Summary saved to {csv_path}")
print(f"{'='*50}\n")

# ── PRINT TABLE ───────────────────────────────────────────────
col_w = 22
header = f"{'Config':<{col_w}} {'Recall@5':>9} {'MRR@5':>7} {'NDCG@5':>8} {'Faith.':>7} {'AnsRel.':>8} {'CtxPrec.':>9}"
print(header)
print("-" * len(header))
for row in summary:
    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "
    print(
        f"{row['config']:<{col_w}} "
        f"{fmt(row['recall@5']):>9} "
        f"{fmt(row['mrr@5']):>7} "
        f"{fmt(row['ndcg@5']):>8} "
        f"{fmt(row['faithfulness']):>7} "
        f"{fmt(row['answer_relevancy']):>8} "
        f"{fmt(row['context_precision']):>9}"
    )
