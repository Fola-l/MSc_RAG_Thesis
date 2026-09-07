"""
significance_tests_local.py
Pairwise significance tests across the four LOCAL (Ollama-generated) RAG
configurations. Local counterpart to significance_tests.py.

Statistical design (same as the original):
  Continuous metrics (Recall@5, MRR@5, NDCG@5, F1_all, Faithfulness):
    Wilcoxon signed-rank test, paired, two-sided.
    Effect size: rank-biserial correlation r = |Z| / sqrt(n_nonzero)
      where Z = (W+ - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24), W+ from scipy.

  Binary metrics (EM, Refusal):
    McNemar's test with Yates continuity correction.

  Multiple comparison correction:
    Benjamini-Hochberg FDR within each metric family (6 pairs × BH).

  Faithfulness: unlike the original run, per-query scores WERE saved
    (faithfulness_per_query_local.csv), so Wilcoxon runs on complete-case
    pairs here instead of falling back to means-only.

  Threshold: alpha = 0.05 after BH correction.

Outputs:
  results/significance_master_local.csv
  Final printed comparison across every metric, plus a count of how many
  pairwise comparisons were statistically significant per metric family.
"""

import os, sys, json, csv, math
import numpy as np
from scipy.stats import wilcoxon, chi2 as chi2_dist
from statsmodels.stats.multitest import multipletests
import ir_datasets

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIGS = [
    "config1_dense",
    "config2_dense_rerank",
    "config3_hybrid",
    "config4_hybrid_rerank",
]

SHORT = {
    "config1_dense"             : "C1",
    "config2_dense_rerank"      : "C2",
    "config3_hybrid"            : "C3",
    "config4_hybrid_rerank"     : "C4",
}

PAIRS = [
    ("config1_dense", "config2_dense_rerank"),          # reranking on dense
    ("config3_hybrid", "config4_hybrid_rerank"),        # reranking on hybrid
    ("config1_dense", "config3_hybrid"),                # fusion, no rerank
    ("config2_dense_rerank", "config4_hybrid_rerank"),  # fusion, with rerank
    ("config1_dense", "config4_hybrid_rerank"),         # full additive effect
    ("config2_dense_rerank", "config3_hybrid"),         # cross-comparison
]

print("Loading BEIR qrels...")
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

print("Recomputing per-query retrieval metrics (local results)...")
retrieval = {}   # config → {qid: {recall, mrr, ndcg}}

for config in CONFIGS:
    path = os.path.join(RESULTS_DIR, f"{config}_local.json")
    with open(path) as f:
        results = json.load(f)
    per_q = {}
    skipped = 0
    for r in results:
        qid = str(r['query_id'])
        rel = qrels.get(qid)
        ret = [str(d) for d in r['retrieved_ids']]
        rec = recall_at_k(ret, rel)
        if rec is None:
            skipped += 1
        per_q[qid] = {
            'recall': rec,
            'mrr'   : mrr_at_k(ret, rel),
            'ndcg'  : ndcg_at_k(ret, rel),
        }
    retrieval[config] = per_q
    n_valid = sum(1 for v in per_q.values() if v['recall'] is not None)
    print(f"  {config}: {n_valid}/500 with qrels, {skipped} skipped (no qrels)")

print("\nLoading correctness_per_query_local.csv...")
correctness = {c: {} for c in CONFIGS}
with open(os.path.join(RESULTS_DIR, "correctness_per_query_local.csv")) as f:
    for row in csv.DictReader(f):
        c   = row['config']
        qid = row['query_id']
        correctness[c][qid] = {
            'is_refusal': int(row['is_refusal']),
            'em'        : float(row['em']) if row['em'] != '' else None,
            'f1'        : float(row['f1']) if row['f1'] != '' else None,
        }
n_per_config = {c: len(correctness[c]) for c in CONFIGS}
print(f"  Loaded: {n_per_config}")

print("\nLoading faithfulness_per_query_local.csv...")
faithfulness = {c: {} for c in CONFIGS}
faith_pq_path = os.path.join(RESULTS_DIR, "faithfulness_per_query_local.csv")
if os.path.exists(faith_pq_path):
    with open(faith_pq_path) as f:
        for row in csv.DictReader(f):
            c   = row['config']
            qid = row['query_id']
            val = row['faithfulness']
            faithfulness[c][qid] = {'faithfulness': float(val) if val != '' else None}
    n_faith = {c: len(faithfulness[c]) for c in CONFIGS}
    print(f"  Loaded: {n_faith}")
else:
    print("  ⚠ faithfulness_per_query_local.csv not found — run score_faithfulness_perquery_local.py first")

with open(os.path.join(RESULTS_DIR, "faithfulness_cache_local.json")) as f:
    faith_cache = json.load(f)
faith_means = {c: faith_cache.get(c, {}).get('mean') for c in CONFIGS}

def get_pairs(ca, cb, data, key, allow_none=False):
    """
    Returns aligned (xa, xb) lists for queries present in both configs.
    Skips pairs where either value is None unless allow_none=True.
    """
    qids = sorted(set(data[ca]) & set(data[cb]))
    xa, xb = [], []
    for qid in qids:
        a_v = data[ca][qid][key]
        b_v = data[cb][qid][key]
        if not allow_none and (a_v is None or b_v is None):
            continue
        xa.append(a_v)
        xb.append(b_v)
    return np.array(xa, dtype=float), np.array(xb, dtype=float)

def run_wilcoxon(xa, xb):
    """
    Paired Wilcoxon signed-rank, two-sided.
    Effect size: rank-biserial r via Z-score of W+.
    Returns dict with keys: mean_a, mean_b, diff, n, raw_p, r_rb.
    raw_p=None if n_nonzero < 10.
    """
    n        = len(xa)
    diffs    = xa - xb
    nz_mask  = diffs != 0
    n_nonzero = int(np.sum(nz_mask))

    d = {
        'mean_a': float(np.mean(xa)) if n else float('nan'),
        'mean_b': float(np.mean(xb)) if n else float('nan'),
        'diff'  : float(np.mean(diffs)) if n else float('nan'),
        'n'     : n,
        'raw_p' : None,
        'r_rb'  : None,
    }

    if n_nonzero < 10:
        return d   # insufficient non-tied pairs

    stat, p = wilcoxon(diffs, zero_method='wilcox', alternative='two-sided')

    W_mu  = n_nonzero * (n_nonzero + 1) / 4
    W_sd  = math.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24)
    Z     = (stat - W_mu) / W_sd if W_sd > 0 else 0.0
    r_rb  = abs(Z) / math.sqrt(n_nonzero)

    d['raw_p'] = float(p)
    d['r_rb']  = round(r_rb, 4)
    return d

def run_mcnemar(xa, xb):
    """
    McNemar's test with Yates continuity correction for paired binary data.
    Returns dict with mean_a, mean_b, diff, n, raw_p.
    """
    xa, xb = xa.astype(int), xb.astype(int)
    n   = len(xa)
    bc  = int(np.sum((xa == 1) & (xb == 0)))   # A correct, B not
    cc  = int(np.sum((xa == 0) & (xb == 1)))   # B correct, A not

    d = {
        'mean_a': float(np.mean(xa)) if n else float('nan'),
        'mean_b': float(np.mean(xb)) if n else float('nan'),
        'diff'  : float(np.mean(xa) - np.mean(xb)) if n else float('nan'),
        'n'     : n,
        'raw_p' : None,
        'r_rb'  : None,
    }

    if bc + cc == 0:
        d['raw_p'] = 1.0
        return d

    chi2_stat = (abs(bc - cc) - 1) ** 2 / (bc + cc)
    d['raw_p'] = float(chi2_dist.sf(chi2_stat, df=1))
    return d

def collect_family(metric, data, key, test_fn, allow_none=False):
    rows = []
    for ca, cb in PAIRS:
        xa, xb = get_pairs(ca, cb, data, key, allow_none=allow_none)
        n_flag = ' ⚠<30' if len(xa) < 30 else ''
        res = test_fn(xa, xb)
        higher = SHORT[ca] if res['diff'] > 1e-9 else (SHORT[cb] if res['diff'] < -1e-9 else 'tie')
        rows.append({
            'metric'   : metric,
            'config_a' : SHORT[ca],
            'config_b' : SHORT[cb],
            'mean_a'   : round(res['mean_a'], 4),
            'mean_b'   : round(res['mean_b'], 4),
            'diff'     : round(res['diff'], 4),
            'higher'   : higher,
            'n'        : res['n'],
            'n_flag'   : n_flag,
            'raw_p'    : res['raw_p'],
            'r_rb'     : res.get('r_rb'),
        })

    p_vals   = [r['raw_p'] for r in rows]
    valid    = [(i, p) for i, p in enumerate(p_vals) if p is not None]
    adj_map  = {}
    if valid:
        idxs, ps = zip(*valid)
        _, adj_ps, _, _ = multipletests(list(ps), method='fdr_bh', alpha=0.05)
        adj_map = dict(zip(idxs, adj_ps))

    for i, r in enumerate(rows):
        adj = adj_map.get(i)
        r['adj_p']       = round(float(adj), 4) if adj is not None else None
        r['significant'] = ('yes' if adj < 0.05 else 'no') if adj is not None else None

    return rows

all_rows = []
all_rows += collect_family('Recall@5',     retrieval,    'recall',       run_wilcoxon)
all_rows += collect_family('MRR@5',        retrieval,    'mrr',          run_wilcoxon)
all_rows += collect_family('NDCG@5',       retrieval,    'ndcg',         run_wilcoxon)
all_rows += collect_family('F1_all',       correctness,  'f1',           run_wilcoxon)
all_rows += collect_family('EM',           correctness,  'em',           run_mcnemar)
all_rows += collect_family('Refusal',      correctness,  'is_refusal',   run_mcnemar, allow_none=True)
all_rows += collect_family('Faithfulness', faithfulness, 'faithfulness', run_wilcoxon)

HDR = f"  {'Pair':<8} {'N':>4} {'MeanA':>7} {'MeanB':>7} {'Diff':>8} {'Hi':>3} {'raw_p':>8} {'adj_p':>8} {'Sig':>5} {'r_rb':>6}"
SEP = "  " + "-" * 74

test_name_for = lambda m: 'Wilcoxon' if m not in ('EM', 'Refusal') else 'McNemar'

current = None
sig_counts = {}   # metric → (n_significant, n_tested)
for r in all_rows:
    if r['metric'] != current:
        current = r['metric']
        sig_counts[current] = [0, 0]
        print(f"\n{'='*76}")
        print(f" {current}   [{test_name_for(current)}]")
        print(f"{'='*76}")
        print(HDR)
        print(SEP)

    pair = f"{r['config_a']}v{r['config_b']}"
    raw_p = f"{r['raw_p']:.4f}" if r['raw_p'] is not None else "  N/A"
    adj_p = f"{r['adj_p']:.4f}" if r['adj_p'] is not None else "  N/A"
    sig   = r.get('significant') or 'N/A'
    r_rb  = f"{r['r_rb']:.3f}"  if r.get('r_rb') is not None else "  N/A"
    diff  = f"{r['diff']:+.4f}" if r['diff'] is not None else "  N/A"
    nflag = r.get('n_flag', '')

    print(f"  {pair:<8} {r['n']:>4} {r['mean_a']:>7.4f} {r['mean_b']:>7.4f} {diff:>8} {r['higher']:>3} "
          f"{raw_p:>8} {adj_p:>8} {sig:>5} {r_rb:>6}{nflag}")

    if r.get('adj_p') is not None:
        sig_counts[current][1] += 1
        if sig == 'yes':
            sig_counts[current][0] += 1

out_path = os.path.join(RESULTS_DIR, "significance_master_local.csv")
fields   = ['metric','config_a','config_b','mean_a','mean_b','diff','higher',
            'n','test','raw_p','adj_p','significant','effect_size']
with open(out_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in all_rows:
        w.writerow({
            'metric'      : r['metric'],
            'config_a'    : r['config_a'],
            'config_b'    : r['config_b'],
            'mean_a'      : r['mean_a'],
            'mean_b'      : r['mean_b'],
            'diff'        : r['diff'],
            'higher'      : r['higher'],
            'n'           : r['n'],
            'test'        : test_name_for(r['metric']),
            'raw_p'       : r['raw_p'],
            'adj_p'       : r['adj_p'],
            'significant' : r.get('significant'),
            'effect_size' : r.get('r_rb'),
        })

print(f"\n✓ Saved: {out_path}  ({len(all_rows)} rows)")

# ── FINAL SUMMARY: per-config metrics + significant-pair counts ───────────
print(f"\n{'='*90}")
print(" FINAL SUMMARY — Local Ollama run (llama3.2:3b generator, llama3.1:8b judge)")
print(f"{'='*90}")

col = 25
hdr = (f"{'Config':<{col}} {'Recall@5':>9} {'MRR@5':>7} {'NDCG@5':>8} "
       f"{'Refusal%':>9} {'EM_all':>8} {'F1_all':>8} {'Faithful.':>10}")
print(hdr)
print("-" * len(hdr))
for c in CONFIGS:
    ret_vals = [v for v in retrieval[c].values() if v['recall'] is not None]
    recall_m = sum(v['recall'] for v in ret_vals) / len(ret_vals) if ret_vals else float('nan')
    mrr_m    = sum(v['mrr']    for v in ret_vals) / len(ret_vals) if ret_vals else float('nan')
    ndcg_m   = sum(v['ndcg']   for v in ret_vals) / len(ret_vals) if ret_vals else float('nan')

    corr_vals   = correctness[c].values()
    refusal_m   = sum(v['is_refusal'] for v in corr_vals) / len(corr_vals) if corr_vals else float('nan')
    em_vals     = [v['em'] for v in corr_vals if v['em'] is not None]
    f1_vals     = [v['f1'] for v in corr_vals if v['f1'] is not None]
    em_m        = sum(em_vals) / len(em_vals) if em_vals else float('nan')
    f1_m        = sum(f1_vals) / len(f1_vals) if f1_vals else float('nan')

    faith_m = faith_means.get(c)
    faith_s = f"{faith_m:.4f}" if faith_m is not None else "N/A"

    print(f"{c:<{col}} {recall_m:>9.4f} {mrr_m:>7.4f} {ndcg_m:>8.4f} "
          f"{refusal_m*100:>8.1f}% {em_m:>8.4f} {f1_m:>8.4f} {faith_s:>10}")

print(f"\n{'-'*50}")
print(" SIGNIFICANT PAIRWISE COMPARISONS (of 6, per metric family, adj_p<0.05)")
print(f"{'-'*50}")
for metric, (n_sig, n_tested) in sig_counts.items():
    print(f"  {metric:<14} {n_sig}/{n_tested} significant")

print(f"\nDone.")
