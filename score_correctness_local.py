"""
score_correctness_local.py
Scores EM, Token F1, and refusal rate for the four LOCAL (Ollama-generated)
configs against NQ gold answers. Zero API calls — pure local computation.

Local counterpart to score_correctness.py — same logic, same gold_answers.json,
pointed at the config*_local.json result files.

Outputs:
  results/correctness_per_query_local.csv   — per-query scores across all configs
  results/correctness_summary_local.csv     — per-config summary statistics
"""

import os
import re
import json
import csv
import string
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIG_FILES = {
    "config1_dense"        : os.path.join(RESULTS_DIR, "config1_dense_local.json"),
    "config2_dense_rerank" : os.path.join(RESULTS_DIR, "config2_dense_rerank_local.json"),
    "config3_hybrid"       : os.path.join(RESULTS_DIR, "config3_hybrid_local.json"),
    "config4_hybrid_rerank": os.path.join(RESULTS_DIR, "config4_hybrid_rerank_local.json"),
}

GOLD_PATH = os.path.join(BASE_DIR, "gold_answers.json")

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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

# punctuation is stripped before matching, so "don't" → "dont"
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

def compute_em(prediction: str, gold_answers: list) -> int:
    norm_pred = normalize_answer(prediction)
    return int(any(norm_pred == normalize_answer(g) for g in gold_answers))

def _token_f1_single(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common   = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_f1(prediction: str, gold_answers: list) -> float:
    return max(_token_f1_single(prediction, g) for g in gold_answers)

with open(GOLD_PATH) as f:
    gold_answers = json.load(f)

scored_ids = {qid for qid, answers in gold_answers.items() if answers}
print(f"Gold answers loaded : {len(gold_answers)} query IDs total")
print(f"With short answers  : {len(scored_ids)} (these are scored for EM/F1)")
print(f"Without answers     : {len(gold_answers) - len(scored_ids)} (counted for refusal rate only)")

print("\nRefusal patterns (applied after SQuAD normalization):")
for p in REFUSAL_PATTERNS:
    print(f"  '{p}'")

per_query_rows = []
summary_rows   = []

for config_name, path in CONFIG_FILES.items():
    with open(path) as f:
        results = json.load(f)

    print(f"\n{'='*60}")
    print(f"Config : {config_name}  ({len(results)} queries total)")
    print(f"{'='*60}")

    n_total           = len(results)
    n_refusal_all     = 0
    n_refusal_scored  = 0
    n_scored          = 0
    em_all, f1_all    = [], []
    em_nr,  f1_nr     = [], []   # non-refusal only

    for r in results:
        qid     = r["query_id"]
        answer  = r.get("answer") or ""
        refusal = is_refusal(answer)

        if refusal:
            n_refusal_all += 1

        has_gold = qid in scored_ids

        if not has_gold:
            per_query_rows.append({
                "query_id"  : qid,
                "config"    : config_name,
                "is_refusal": int(refusal),
                "em"        : "",
                "f1"        : "",
            })
            continue

        golds = gold_answers[qid]
        em    = compute_em(answer, golds)
        f1    = compute_f1(answer, golds)
        n_scored += 1

        if refusal:
            n_refusal_scored += 1

        em_all.append(em)
        f1_all.append(f1)
        if not refusal:
            em_nr.append(em)
            f1_nr.append(f1)

        per_query_rows.append({
            "query_id"  : qid,
            "config"    : config_name,
            "is_refusal": int(refusal),
            "em"        : round(em,   4),
            "f1"        : round(f1,   4),
        })

    if n_scored != 395:
        print(f"  ⚠ WARNING: expected 395 scored queries, got {n_scored}")
    else:
        print(f"  N scored (EM/F1) : {n_scored} ✓")

    mean_em    = sum(em_all) / len(em_all) if em_all else 0.0
    mean_f1    = sum(f1_all) / len(f1_all) if f1_all else 0.0
    mean_em_nr = sum(em_nr)  / len(em_nr)  if em_nr  else 0.0
    mean_f1_nr = sum(f1_nr)  / len(f1_nr)  if f1_nr  else 0.0

    rr_500 = n_refusal_all    / n_total
    rr_395 = n_refusal_scored / n_scored if n_scored else 0.0

    print(f"  Refusal rate (all 500)  : {n_refusal_all}/{n_total} = {rr_500:.4f}")
    print(f"  Refusal rate (395 sub)  : {n_refusal_scored}/{n_scored} = {rr_395:.4f}")
    print(f"  EM   (all 395)          : {mean_em:.4f}")
    print(f"  F1   (all 395)          : {mean_f1:.4f}")
    print(f"  EM   (non-refusals, n={len(em_nr)}) : {mean_em_nr:.4f}")
    print(f"  F1   (non-refusals, n={len(f1_nr)}) : {mean_f1_nr:.4f}")

    summary_rows.append({
        "config"          : config_name,
        "n_scored"        : n_scored,
        "refusal_rate_500": round(rr_500, 4),
        "refusal_rate_395": round(rr_395, 4),
        "EM_all"          : round(mean_em,    4),
        "F1_all"          : round(mean_f1,    4),
        "EM_nonrefusal"   : round(mean_em_nr, 4),
        "F1_nonrefusal"   : round(mean_f1_nr, 4),
    })

pq_path = os.path.join(RESULTS_DIR, "correctness_per_query_local.csv")
pq_fields = ["query_id", "config", "is_refusal", "em", "f1"]
with open(pq_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=pq_fields)
    writer.writeheader()
    writer.writerows(per_query_rows)
print(f"\n✓ Per-query CSV saved : {pq_path}")

sum_path = os.path.join(RESULTS_DIR, "correctness_summary_local.csv")
sum_fields = [
    "config", "n_scored", "refusal_rate_500", "refusal_rate_395",
    "EM_all", "F1_all", "EM_nonrefusal", "F1_nonrefusal",
]
with open(sum_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sum_fields)
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"✓ Summary CSV saved   : {sum_path}")

col = 25
print(f"\n{'='*80}")
print(f"{'Config':<{col}} {'Refusal%':>9} {'EM_all':>8} {'F1_all':>8} {'EM_nonref':>10} {'F1_nonref':>10}")
print("-" * 80)
for row in summary_rows:
    print(
        f"{row['config']:<{col}} "
        f"{row['refusal_rate_500']*100:>8.1f}% "
        f"{row['EM_all']:>8.4f} "
        f"{row['F1_all']:>8.4f} "
        f"{row['EM_nonrefusal']:>10.4f} "
        f"{row['F1_nonrefusal']:>10.4f}"
    )
