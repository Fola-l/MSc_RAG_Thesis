"""
Step 2: Build gold_answers.json mapping BEIR query IDs → NQ short answer strings.

Run this AFTER verify_nq_split.py confirms which split to use.
Update NQ_SPLIT below if verify script says 'train'.

Output: gold_answers.json
  {
    "test102": ["Alexander Graham Bell", "Bell"],  # multiple valid answers
    "test456": [],                                  # no short answer in NQ
    ...
  }
"""

import json
from datasets import load_dataset

# ── CONFIG ────────────────────────────────────────────────────
NQ_SPLIT = 'validation'   # change to 'train' if verify script says so
# ──────────────────────────────────────────────────────────────

print(f"Loading BEIR NQ queries...")
beir_queries = load_dataset('BeIR/nq', 'queries', split='queries')
id_to_text   = {q['_id']: q['text'] for q in beir_queries}

with open("sampled_queries.json") as f:
    sampled_ids = json.load(f)

sampled_texts = {id_to_text[qid]: qid for qid in sampled_ids}
print(f"Loaded {len(sampled_ids)} sampled query IDs")
print(f"Target: match all {len(sampled_texts)} unique query texts\n")

print(f"Scanning NQ {NQ_SPLIT} split...")
nq = load_dataset('google-research-datasets/natural_questions', split=NQ_SPLIT, streaming=True)

gold_answers  = {qid: [] for qid in sampled_ids}
matched_texts = set()
checked       = 0

for row in nq:
    q_text_raw = row['question']['text'].lower().strip().rstrip('?')

    # Try exact match first, then with trailing ?
    matched_qid = None
    for candidate_text, qid in sampled_texts.items():
        if candidate_text.lower().strip().rstrip('?') == q_text_raw:
            matched_qid = qid
            matched_texts.add(candidate_text)
            break

    if matched_qid:
        # Collect all non-empty short answer texts across all annotators
        answers = []
        for annotation in row['annotations']['short_answers']:
            for text in annotation['text']:
                if text and text not in answers:
                    answers.append(text)
        gold_answers[matched_qid] = answers

    checked += 1
    if checked % 20000 == 0:
        print(f"  ...{checked} NQ entries scanned | {len(matched_texts)}/{len(sampled_texts)} matched")

    if len(matched_texts) == len(sampled_texts):
        print(f"  All {len(sampled_texts)} queries matched — stopping early at {checked} entries")
        break

# ── COVERAGE REPORT ───────────────────────────────────────────
with_answers    = sum(1 for v in gold_answers.values() if v)
without_answers = sum(1 for v in gold_answers.values() if not v)
unmatched       = sum(1 for qid in sampled_ids if qid not in matched_texts
                      and id_to_text.get(qid, '') not in matched_texts)

print(f"\n{'='*50}")
print(f"Coverage Report")
print(f"{'='*50}")
print(f"Total sampled queries  : {len(sampled_ids)}")
print(f"Matched in NQ          : {len(matched_texts)}")
print(f"  - With short answers : {with_answers}")
print(f"  - No short answer    : {without_answers - (len(sampled_ids) - len(matched_texts))}")
print(f"Not found in NQ        : {len(sampled_ids) - len(matched_texts)}")

# ── SAVE ──────────────────────────────────────────────────────
with open("gold_answers.json", "w") as f:
    json.dump(gold_answers, f, indent=2)

print(f"\n✓ Saved gold_answers.json")
print(f"  Queries with usable gold answers: {with_answers}/500")

# Show a few examples
print(f"\nSample entries:")
shown = 0
for qid, answers in gold_answers.items():
    if answers:
        print(f"  {qid}: {id_to_text[qid][:60]}")
        print(f"    → {answers}")
        shown += 1
    if shown >= 3:
        break
