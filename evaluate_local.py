"""
evaluate_local.py

Retrieval metrics (Recall@5, MRR@5, NDCG@5) vs BEIR NQ qrels, plus refusal
rate, for the four local Ollama-generated result files (config*_local.json).

Local counterpart to evaluate.py. RAGAS faithfulness needs an LLM judge and
is scored separately by score_faithfulness_perquery_local.py.

Output: results/evaluation_summary_local.csv
"""

import os
import json
import csv
import math
import re
import string

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIG_FILES = {
    "config1_dense"        : os.path.join(RESULTS_DIR, "config1_dense_local.json"),
    "config2_dense_rerank" : os.path.join(RESULTS_DIR, "config2_dense_rerank_local.json"),
    "config3_hybrid"       : os.path.join(RESULTS_DIR, "config3_hybrid_local.json"),
    "config4_hybrid_rerank": os.path.join(RESULTS_DIR, "config4_hybrid_rerank_local.json"),
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
    raise SystemExit("No local result files found. Run the local config scripts first.")

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

# ── REFUSAL DETECTION (same logic as score_correctness.py) ────
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

REFUSAL_PATTERNS = [
    "i dont know",
    "i do not know",
    "i am not sure",
    "not enough information",
    "insufficient information",
    "cannot be determined",
    "unable to determine",
    "the context does not provide",
    "the context does not contain",
    "the provided context does not",
    "does not provide enough",
    "no information provided",
]

def is_refusal(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    norm = normalize_answer(answer)
    return any(pattern in norm for pattern in REFUSAL_PATTERNS)

def compute_refusal_rate(results):
    n_total = len(results)
    n_refusal = sum(1 for r in results if is_refusal(r.get('answer') or ''))
    return n_refusal / n_total if n_total else 0.0

# ── RUN EVERYTHING ────────────────────────────────────────────
csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_local.csv")
fieldnames = ["config", "n_queries", "recall@5", "mrr@5", "ndcg@5", "refusal_rate"]
summary = []

for config_name, results in all_results.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}  ({len(results)} queries)")
    print(f"{'='*50}")

    ret_metrics  = compute_retrieval_metrics(results)
    refusal_rate = compute_refusal_rate(results)

    print(f"  Recall@5     : {ret_metrics['recall@5']}")
    print(f"  MRR@5        : {ret_metrics['mrr@5']}")
    print(f"  NDCG@5       : {ret_metrics['ndcg@5']}")
    print(f"  Refusal rate : {refusal_rate:.4f}")

    summary.append({
        "config"      : config_name,
        "n_queries"   : ret_metrics["n_eval"],
        "recall@5"    : ret_metrics["recall@5"],
        "mrr@5"       : ret_metrics["mrr@5"],
        "ndcg@5"      : ret_metrics["ndcg@5"],
        "refusal_rate": round(refusal_rate, 4),
    })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

print(f"\n{'='*50}")
print(f"✓ Evaluation complete. Summary saved to {csv_path}")
print(f"{'='*50}\n")

col_w = 22
header = f"{'Config':<{col_w}} {'Recall@5':>9} {'MRR@5':>7} {'NDCG@5':>8} {'Refusal':>8}"
print(header)
print("-" * len(header))
for row in summary:
    print(
        f"{row['config']:<{col_w}} "
        f"{row['recall@5']:>9.4f} "
        f"{row['mrr@5']:>7.4f} "
        f"{row['ndcg@5']:>8.4f} "
        f"{row['refusal_rate']*100:>7.1f}%"
    )
