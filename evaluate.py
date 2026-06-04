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
import random
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

# ── FIXED 100-QUERY RAGAS SAMPLE (same across all configs) ────
random.seed(42)
reference_ids  = [r['query_id'] for r in list(all_results.values())[0]]
sampled_ids    = set(random.sample(reference_ids, 100))

ragas_sample_path = os.path.join(BASE_DIR, "ragas_sample_ids.json")
with open(ragas_sample_path, "w") as f:
    json.dump(sorted(sampled_ids), f)
print(f"\nRAGAS sample: 100 query IDs fixed (seed=42) — saved to ragas_sample_ids.json")

# ── LOAD BEIR NQ QRELS ────────────────────────────────────────
print("\nLoading BEIR NQ qrels...")
import ir_datasets

qrels = {}
for qrel in ir_datasets.load('beir/nq').qrels_iter():
    if qrel.relevance > 0:
        qrels.setdefault(str(qrel.query_id), set()).add(str(qrel.doc_id))

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
def _extract_score(val):
    """Handle RAGAS returning either a float or a list of per-sample scores."""
    if isinstance(val, (list, tuple)):
        valid = [float(v) for v in val
                 if v is not None and isinstance(v, (int, float)) and not math.isnan(float(v))]
        return sum(valid) / len(valid) if valid else float('nan')
    if val is None:
        return float('nan')
    return float(val)

def compute_ragas_metrics(results, config_name, sampled_ids):
    results = [r for r in results if r['query_id'] in sampled_ids]
    print(f"  RAGAS subset: {len(results)} queries")
    try:
        import sys, types
        # ragas/llms/base.py imports ChatVertexAI at module level; stub it out
        # since langchain-community >= 0.3 removed chat_models.vertexai
        if 'langchain_community.chat_models.vertexai' not in sys.modules:
            _stub = types.ModuleType('langchain_community.chat_models.vertexai')
            _stub.ChatVertexAI = type('ChatVertexAI', (), {})
            sys.modules['langchain_community.chat_models.vertexai'] = _stub

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
    except ImportError as e:
        print(f"  RAGAS import error: {e}")
        return {"faithfulness": None}

    print(f"  Configuring RAGAS with Groq LLM...")
    groq_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.0,
    ))
    faithfulness.llm = groq_llm

    data = {
        "question": [r["query"]    for r in results],
        "answer"  : [r["answer"]   for r in results],
        "contexts": [r["contexts"] for r in results],
    }
    dataset = Dataset.from_dict(data)

    BATCH_SIZE = 50
    n_batches  = math.ceil(len(dataset) / BATCH_SIZE)
    print(f"  Running RAGAS faithfulness on {len(dataset)} samples in {n_batches} batches...")

    batch_faith = []
    try:
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, len(dataset))
            batch = dataset.select(range(start, end))

            batch_scores = evaluate(batch, metrics=[faithfulness])
            score = _extract_score(batch_scores["faithfulness"])
            batch_faith.append(score)

            valid_so_far = [s for s in batch_faith if not math.isnan(s)]
            mean_so_far  = sum(valid_so_far) / len(valid_so_far) if valid_so_far else float('nan')
            print(f"  Batch {batch_idx + 1}/{n_batches} done — faithfulness={score:.4f}  running mean={mean_so_far:.4f}")

            if batch_idx < n_batches - 1:
                time.sleep(60)  # pause between batches for Groq rate limit

        valid = [s for s in batch_faith if not math.isnan(s)]
        mean_faith = round(sum(valid) / len(valid), 4) if valid else None
        return {"faithfulness": mean_faith}
    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return {"faithfulness": None}

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
    gen_metrics = compute_ragas_metrics(results, config_name, sampled_ids)
    print(f"  Faithfulness : {gen_metrics['faithfulness']}")

    summary.append({
        "config"      : config_name,
        "n_queries"   : ret_metrics["n_eval"],
        "ragas_n"     : 100,
        "recall@5"    : ret_metrics["recall@5"],
        "mrr@5"       : ret_metrics["mrr@5"],
        "ndcg@5"      : ret_metrics["ndcg@5"],
        "faithfulness": gen_metrics["faithfulness"],
    })

# ── SAVE CSV ──────────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
fieldnames = [
    "config", "n_queries", "ragas_n",
    "recall@5", "mrr@5", "ndcg@5",
    "faithfulness",
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
header = f"{'Config':<{col_w}} {'Recall@5':>9} {'MRR@5':>7} {'NDCG@5':>8} {'Faith.':>8}"
print(header)
print("-" * len(header))
for row in summary:
    def fmt(v):
        return f"{v:.4f}" if v is not None else "   N/A"
    print(
        f"{row['config']:<{col_w}} "
        f"{fmt(row['recall@5']):>9} "
        f"{fmt(row['mrr@5']):>7} "
        f"{fmt(row['ndcg@5']):>8} "
        f"{fmt(row['faithfulness']):>8}"
    )
