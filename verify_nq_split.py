"""
Step 1: Verify which NQ split contains the BEIR test queries.
Checks 5 sample BEIR query texts against NQ validation and train splits.
Run this first — takes ~1-2 min for validation, longer for train.
"""

import json
from datasets import load_dataset

with open("sampled_queries.json") as f:
    sampled_ids = json.load(f)

# Load BEIR queries to get the actual question texts for our 5 samples
from datasets import load_dataset as lds
beir_queries = lds('BeIR/nq', 'queries', split='queries')
id_to_text = {q['_id']: q['text'] for q in beir_queries}

sample_ids   = sampled_ids[:5]
sample_texts = [id_to_text[qid] for qid in sample_ids]

print("5 sample BEIR query texts:")
for i, (qid, text) in enumerate(zip(sample_ids, sample_texts)):
    print(f"  [{qid}] {text}")

print()

for split in ['validation', 'train']:
    print(f"Checking NQ {split} split...")
    nq = load_dataset('google-research-datasets/natural_questions', split=split, streaming=True)

    found = set()
    checked = 0
    for row in nq:
        q_text = row['question']['text'].lower().strip().rstrip('?')
        for sample_text in sample_texts:
            if sample_text.lower().strip().rstrip('?') == q_text:
                found.add(sample_text)
        checked += 1
        if checked % 10000 == 0:
            print(f"  ...checked {checked} NQ entries, found {len(found)}/5 so far")
        if len(found) == 5:
            break

    print(f"  Result: {len(found)}/5 sample queries found in NQ {split} split after {checked} entries")
    print(f"  Found: {found}\n")

    if len(found) >= 3:
        print(f">>> USE NQ '{split}' SPLIT for the full build <<<")
        break
