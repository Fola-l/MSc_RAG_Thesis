"""
materialize_retrieval_per_query_local.py

Recomputes Recall@5, MRR@5, NDCG@5 per query per config from the raw
retrieved_ids field in results/config{1..4}_*_local.json against BEIR NQ
qrels, and persists them (same logic as significance_tests_local.py, which
previously only computed these in-memory and discarded them).

Output: results/retrieval_per_query_local.csv
  columns: query_id, config, recall@5, mrr@5, ndcg@5   (long format,
  matching correctness_per_query_local.csv's structure)

After writing, verifies per-config means against results/evaluation_summary_local.csv
as a sanity check and reports any mismatch.
"""

import os, json, csv, math

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIGS = [
    "config1_dense",
    "config2_dense_rerank",
    "config3_hybrid",
    "config4_hybrid_rerank",
]

print("Loading BEIR qrels...")
import ir_datasets
qrels = {}
for qrel in ir_datasets.load('beir/nq').qrels_iter():
    if qrel.relevance > 0:
        qrels.setdefault(str(qrel.query_id), set()).add(str(qrel.doc_id))
print(f"  Qrels loaded for {len(qrels)} queries")

def recall_at_k(ret, rel, k=5):
    if not rel: return None
    return len(set(ret[:k]) & rel) / len(rel)

def mrr_at_k(ret, rel, k=5):
    if not rel: return None
    for rank, d in enumerate(ret[:k], 1):
        if d in rel: return 1.0 / rank
    return 0.0

def ndcg_at_k(ret, rel, k=5):
    if not rel: return None
    dcg  = sum(1.0/math.log2(r+1) for r,d in enumerate(ret[:k],1) if d in rel)
    idcg = sum(1.0/math.log2(r+1) for r in range(1, min(len(rel),k)+1))
    return dcg / idcg if idcg else 0.0

rows = []
per_config_means = {}

for config in CONFIGS:
    path = os.path.join(RESULTS_DIR, f"{config}_local.json")
    with open(path) as f:
        results = json.load(f)

    recalls, mrrs, ndcgs = [], [], []
    for r in results:
        qid = str(r['query_id'])
        rel = qrels.get(qid)
        ret = [str(d) for d in r['retrieved_ids']]

        rec  = recall_at_k(ret, rel)
        mrr  = mrr_at_k(ret, rel)
        ndcg = ndcg_at_k(ret, rel)

        if rec is not None:
            recalls.append(rec); mrrs.append(mrr); ndcgs.append(ndcg)

        rows.append({
            "query_id" : qid,
            "config"   : config,
            "recall@5" : "" if rec  is None else round(rec, 6),
            "mrr@5"    : "" if mrr  is None else round(mrr, 6),
            "ndcg@5"   : "" if ndcg is None else round(ndcg, 6),
        })

    n = len(recalls)
    per_config_means[config] = {
        "n_eval"   : n,
        "recall@5" : round(sum(recalls)/n, 4) if n else 0.0,
        "mrr@5"    : round(sum(mrrs)/n, 4)    if n else 0.0,
        "ndcg@5"   : round(sum(ndcgs)/n, 4)   if n else 0.0,
    }
    print(f"  {config}: {n}/500 with qrels")

out_path = os.path.join(RESULTS_DIR, "retrieval_per_query_local.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["query_id", "config", "recall@5", "mrr@5", "ndcg@5"])
    w.writeheader()
    w.writerows(rows)
print(f"\n✓ Saved {out_path}  ({len(rows)} rows)")

# ── SANITY CHECK vs evaluation_summary_local.csv ──────────────────────────
print(f"\n{'='*70}")
print(" VERIFICATION vs results/evaluation_summary_local.csv")
print(f"{'='*70}")

eval_summary_path = os.path.join(RESULTS_DIR, "evaluation_summary_local.csv")
existing = {}
with open(eval_summary_path) as f:
    for row in csv.DictReader(f):
        existing[row["config"]] = row

any_mismatch = False
hdr = f"  {'Config':<25} {'Metric':<10} {'Recomputed':>12} {'Existing':>12} {'Match':>6}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for config in CONFIGS:
    recomputed = per_config_means[config]
    existing_row = existing.get(config)
    if existing_row is None:
        print(f"  ⚠ {config}: not found in evaluation_summary_local.csv")
        any_mismatch = True
        continue
    for metric in ["recall@5", "mrr@5", "ndcg@5"]:
        r_val = recomputed[metric]
        e_val = round(float(existing_row[metric]), 4)
        match = abs(r_val - e_val) < 1e-9
        if not match:
            any_mismatch = True
        print(f"  {config:<25} {metric:<10} {r_val:>12.4f} {e_val:>12.4f} {'OK' if match else 'MISMATCH':>6}")
    n_r = recomputed["n_eval"]
    n_e = int(existing_row["n_queries"])
    n_match = n_r == n_e
    if not n_match:
        any_mismatch = True
    print(f"  {config:<25} {'n_eval':<10} {n_r:>12d} {n_e:>12d} {'OK' if n_match else 'MISMATCH':>6}")

print(f"\n{'='*70}")
if any_mismatch:
    print(" ✗ MISMATCH DETECTED — see rows flagged above. Do not proceed to CI computation")
    print("   until this is resolved.")
else:
    print(" ✓ All per-config means match evaluation_summary_local.csv exactly.")
    print("   Safe to proceed to confidence interval computation.")
print(f"{'='*70}")
