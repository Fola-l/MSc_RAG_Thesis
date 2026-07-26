"""
Configuration 2: Dense + Reranker
- Retriever : DPR (Dense Passage Retrieval)
- Reranker  : cross-encoder/ms-marco-MiniLM-L-6-v2
- Fusion    : None
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
from datasets import load_dataset
from shared import (
    load_faiss,
    load_question_encoder,
    load_reranker,
    dense_retrieve,
    rerank,
    encode_queries,
    generate_answer,
)

random.seed(42)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
QUERIES_PATH = os.path.join(BASE_DIR, "sampled_queries.json")

load_faiss()
load_question_encoder()
load_reranker()

print("Loading queries...")
queries_full = load_dataset('BeIR/nq', 'queries', split='queries')

if os.path.exists(QUERIES_PATH):
    with open(QUERIES_PATH) as f:
        query_ids = set(json.load(f))
    queries = [q for q in queries_full if q['_id'] in query_ids]
else:
    queries = random.sample(list(queries_full), 500)
    with open(QUERIES_PATH, "w") as f:
        json.dump([q['_id'] for q in queries], f)

print(f"Loaded {len(queries)} queries")

query_texts      = [q['text'] for q in queries]
query_embeddings = encode_queries(query_texts)

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
output_path = os.path.join(BASE_DIR, "results", "config2_dense_rerank.json")

if os.path.exists(output_path):
    with open(output_path) as f:
        results = json.load(f)
    print(f"Resuming from {len(results)} completed queries")
else:
    results = []

completed_ids = {r['query_id'] for r in results}

print("\nRunning Configuration 2: Dense + Reranker...")
for i, query_item in enumerate(queries):
    query_id   = query_item['_id']
    query_text = query_item['text']

    if query_id in completed_ids:
        continue

    query_emb = query_embeddings[i:i+1]

    retrieved         = dense_retrieve(query_emb, top_k=50)
    reranked          = rerank(query_text, retrieved, top_k=5)
    contexts          = [text for _, text, _ in reranked]
    doc_ids_retrieved = [doc_id for doc_id, _, _ in reranked]

    answer = generate_answer(query_text, contexts)

    results.append({
        "query_id"     : query_id,
        "query"        : query_text,
        "retrieved_ids": doc_ids_retrieved,
        "contexts"     : contexts,
        "answer"       : answer,
        "config"       : "config2_dense_rerank"
    })

    if len(results) % 50 == 0:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Progress: {len(results)}/500 — checkpoint saved")

    time.sleep(0.5)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Config 2 complete. Results saved to {output_path}")
print(f"  Total queries processed: {len(results)}")
