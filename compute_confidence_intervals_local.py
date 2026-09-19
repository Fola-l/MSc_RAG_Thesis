"""
compute_confidence_intervals_local.py

95% confidence intervals for all metric families on the local Ollama run,
complementing the existing Wilcoxon/McNemar p-values in
significance_master_local.csv.

Continuous metrics (Recall@5, MRR@5, NDCG@5, F1, Faithfulness):
  - Pairwise: 95% CI on the paired mean difference via paired bootstrap
    (resample query IDs with replacement, 2000 resamples, percentile method).
  - Individual: 95% CI on each config's own mean via bootstrap (2000 resamples,
    percentile method) over that config's own available per-query values.
  Bootstrap (not normal approximation) because several of these distributions
  are bounded/skewed (e.g. F1 in [0,1] with a mass at 0, Faithfulness in [0,1]).

Binary metrics (EM, Refusal):
  - Individual: standard Wilson score interval on the single proportion.
  - Pairwise: 95% CI on the difference in paired proportions via Newcombe's
    Method 10 (Newcombe 1998, Statistics in Medicine) — the standard
    Wilson-score-based extension for correlated/paired binary data, which
    matches the paired McNemar design already used in significance testing.

Inputs:
  results/retrieval_per_query_local.csv
  results/correctness_per_query_local.csv
  results/faithfulness_per_query_local.csv

Output:
  results/confidence_intervals_local.csv
  columns: metric, level, config_a, config_b, estimate, ci_lower, ci_upper, method

Does not modify any existing file.
"""

import os, csv, math
import numpy as np

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIGS = [
    "config1_dense",
    "config2_dense_rerank",
    "config3_hybrid",
    "config4_hybrid_rerank",
]
SHORT = {
    "config1_dense"        : "C1",
    "config2_dense_rerank" : "C2",
    "config3_hybrid"       : "C3",
    "config4_hybrid_rerank": "C4",
}
PAIRS = [
    ("config1_dense", "config2_dense_rerank"),
    ("config3_hybrid", "config4_hybrid_rerank"),
    ("config1_dense", "config3_hybrid"),
    ("config2_dense_rerank", "config4_hybrid_rerank"),
    ("config1_dense", "config4_hybrid_rerank"),
    ("config2_dense_rerank", "config3_hybrid"),
]

N_BOOT = 2000
SEED   = 42
Z95    = 1.959963985

# ── LOAD DATA ───────────────────────────────────────────────────────────
def load_long_csv(path, value_cols):
    """query_id, config, <value_cols...> -> {col: {config: {qid: float or None}}}"""
    data = {col: {c: {} for c in CONFIGS} for col in value_cols}
    with open(path) as f:
        for row in csv.DictReader(f):
            c   = row['config']
            qid = row['query_id']
            if c not in CONFIGS:
                continue
            for col in value_cols:
                v = row[col]
                data[col][c][qid] = float(v) if v != '' else None
    return data

retrieval    = load_long_csv(os.path.join(RESULTS_DIR, "retrieval_per_query_local.csv"),
                              ["recall@5", "mrr@5", "ndcg@5"])
correctness  = load_long_csv(os.path.join(RESULTS_DIR, "correctness_per_query_local.csv"),
                              ["is_refusal", "em", "f1"])
faithfulness = load_long_csv(os.path.join(RESULTS_DIR, "faithfulness_per_query_local.csv"),
                              ["faithfulness"])

CONTINUOUS_METRICS = {
    "Recall@5"    : (retrieval,    "recall@5"),
    "MRR@5"       : (retrieval,    "mrr@5"),
    "NDCG@5"      : (retrieval,    "ndcg@5"),
    "F1"          : (correctness,  "f1"),
    "Faithfulness": (faithfulness, "faithfulness"),
}
BINARY_METRICS = {
    "EM"      : (correctness, "em"),
    "Refusal" : (correctness, "is_refusal"),
}

# ── CONTINUOUS: PAIRED BOOTSTRAP ───────────────────────────────────────
def get_complete_pairs(data, key, ca, cb):
    d = data[key]
    common = sorted(set(d[ca]) & set(d[cb]))
    xa, xb = [], []
    for qid in common:
        a_v, b_v = d[ca][qid], d[cb][qid]
        if a_v is None or b_v is None:
            continue
        xa.append(a_v); xb.append(b_v)
    return np.array(xa, dtype=float), np.array(xb, dtype=float)

def get_own_values(data, key, c):
    d = data[key][c]
    return np.array([v for v in d.values() if v is not None], dtype=float)

def bootstrap_paired_diff_ci(xa, xb, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(xa)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(xa[idx]) - np.mean(xb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)

def bootstrap_mean_ci(x, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(x[idx])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

# ── BINARY: WILSON SCORE / NEWCOMBE PAIRED DIFFERENCE ──────────────────
def wilson_ci(k, n, z=Z95):
    if n == 0:
        return (float('nan'), float('nan'))
    phat  = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adj    = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return (lo, hi)

def newcombe_paired_diff_ci(xa, xb, z=Z95):
    """
    Newcombe (1998) Method 10: 95% CI for the difference between two
    correlated (paired) proportions, built from each margin's Wilson interval.
    xa, xb: paired 0/1 arrays of equal length.
    """
    xa = xa.astype(int); xb = xb.astype(int)
    n = len(xa)
    n11 = int(np.sum((xa == 1) & (xb == 1)))
    n10 = int(np.sum((xa == 1) & (xb == 0)))
    n01 = int(np.sum((xa == 0) & (xb == 1)))
    n00 = int(np.sum((xa == 0) & (xb == 0)))

    p_a = (n11 + n10) / n
    p_b = (n11 + n01) / n
    d   = p_a - p_b

    l_a, u_a = wilson_ci(n11 + n10, n, z)
    l_b, u_b = wilson_ci(n11 + n01, n, z)

    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    r = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0

    L = d - math.sqrt((p_a - l_a)**2 - 2 * r * (p_a - l_a) * (u_b - p_b) + (u_b - p_b)**2)
    U = d + math.sqrt((u_a - p_a)**2 - 2 * r * (u_a - p_a) * (p_b - l_b) + (p_b - l_b)**2)
    return d, L, U

# ── RUN ─────────────────────────────────────────────────────────────────
out_rows = []

print(f"{'='*76}\n CONTINUOUS METRICS — paired bootstrap (n_boot={N_BOOT}, seed={SEED})\n{'='*76}")
for metric, (data, key) in CONTINUOUS_METRICS.items():
    print(f"\n{metric}")
    # individual
    for c in CONFIGS:
        x = get_own_values(data, key, c)
        mean = float(np.mean(x))
        lo, hi = bootstrap_mean_ci(x)
        out_rows.append({
            "metric": metric, "level": "individual",
            "config_a": SHORT[c], "config_b": "",
            "estimate": round(mean, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "method": f"Bootstrap (n={N_BOOT})",
        })
        print(f"  {SHORT[c]:<4} mean={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(x)})")
    # pairwise
    for ca, cb in PAIRS:
        xa, xb = get_complete_pairs(data, key, ca, cb)
        diff = float(np.mean(xa) - np.mean(xb))
        lo, hi = bootstrap_paired_diff_ci(xa, xb)
        out_rows.append({
            "metric": metric, "level": "pairwise",
            "config_a": SHORT[ca], "config_b": SHORT[cb],
            "estimate": round(diff, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "method": f"Paired bootstrap (n={N_BOOT})",
        })
        print(f"  {SHORT[ca]}v{SHORT[cb]}  diff={diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(xa)})")

print(f"\n{'='*76}\n BINARY METRICS — Wilson score / Newcombe paired difference\n{'='*76}")
for metric, (data, key) in BINARY_METRICS.items():
    print(f"\n{metric}")
    # individual
    for c in CONFIGS:
        x = get_own_values(data, key, c)
        k, n = int(np.sum(x)), len(x)
        phat = k / n
        lo, hi = wilson_ci(k, n)
        out_rows.append({
            "metric": metric, "level": "individual",
            "config_a": SHORT[c], "config_b": "",
            "estimate": round(phat, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "method": "Wilson score",
        })
        print(f"  {SHORT[c]:<4} p={phat:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={n})")
    # pairwise
    for ca, cb in PAIRS:
        xa, xb = get_complete_pairs(data, key, ca, cb)
        d, lo, hi = newcombe_paired_diff_ci(xa, xb)
        out_rows.append({
            "metric": metric, "level": "pairwise",
            "config_a": SHORT[ca], "config_b": SHORT[cb],
            "estimate": round(d, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "method": "Wilson score (Newcombe paired diff)",
        })
        print(f"  {SHORT[ca]}v{SHORT[cb]}  diff={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(xa)})")

# ── SAVE ───────────────────────────────────────────────────────────────
out_path = os.path.join(RESULTS_DIR, "confidence_intervals_local.csv")
fields = ["metric", "level", "config_a", "config_b", "estimate", "ci_lower", "ci_upper", "method"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"\n✓ Saved: {out_path}  ({len(out_rows)} rows)")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────
print(f"\n{'='*90}\n SUMMARY — 95% CIs by metric\n{'='*90}")
current = None
for r in out_rows:
    if r["metric"] != current:
        current = r["metric"]
        print(f"\n{current}")
        print(f"  {'Level':<11} {'A':>3} {'B':>3} {'Estimate':>9} {'CI Lower':>9} {'CI Upper':>9}  Method")
        print("  " + "-" * 78)
    print(f"  {r['level']:<11} {r['config_a']:>3} {r['config_b']:>3} "
          f"{r['estimate']:>9.4f} {r['ci_lower']:>9.4f} {r['ci_upper']:>9.4f}  {r['method']}")

print(f"\nDone.")
