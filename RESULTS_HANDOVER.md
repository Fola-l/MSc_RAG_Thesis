# RAG Thesis — Experiment Results Handover

## What Was Built

A pipeline comparing four RAG retrieval configurations on the BEIR Natural Questions benchmark.

### Four Configurations

| Config | Retriever | Reranker | Fusion |
|--------|-----------|----------|--------|
| config1_dense | DPR (FAISS) | None | None |
| config2_dense_rerank | DPR (FAISS) | Cross-encoder (ms-marco-MiniLM-L-6-v2) | None |
| config3_hybrid | DPR + BM25 | None | RRF |
| config4_hybrid_rerank | DPR + BM25 | Cross-encoder (ms-marco-MiniLM-L-6-v2) | RRF |

### Pipeline Per Config

- **Config1**: Dense retrieve top-5 → generate
- **Config2**: Dense retrieve top-10 → cross-encoder rerank → take top-5 → generate
- **Config3**: Dense retrieve top-10 + BM25 top-10 → RRF fusion → take top-5 → generate
- **Config4**: Dense retrieve top-10 + BM25 top-10 → RRF fusion → cross-encoder rerank fused top-10 → take top-5 → generate

### Generator (identical across all configs)
- Model: Llama 3.1 8B Instruct via Groq API
- Temperature: 0.0
- Max tokens: 150
- Top-k contexts passed to generator: 5

---

## Dataset

- **Corpus**: BEIR Natural Questions, 100K document subset (randomly sampled from full ~2.68M corpus)
- **Queries**: 500 randomly sampled from BEIR NQ query set (seed=42, saved in `sampled_queries.json`)
- **Query set identical across all four configs**

### Corpus Coverage Check (run post-evaluation)

```
Queries with gold passage IN 100K subset: 425 / 500
Coverage: 85.0%
Queries with NO gold passage in subset:   75 / 500
```

---

## Evaluation Setup

### Retrieval Metrics
- Computed against BEIR NQ qrels (loaded via `ir_datasets`)
- Run on all 500 queries
- Metrics: Recall@5, MRR@5, NDCG@5

### Generation Metrics (RAGAS)
- Metric used: **Faithfulness only**
  - RAGAS `answer_relevancy` was attempted but Groq rejects requests with `n > 1` completions, which that metric requires internally. Faithfulness uses sequential single calls and succeeded.
- Subset: 100 queries (fixed, seed=42, saved in `ragas_sample_ids.json`)
  - Same 100 query IDs used across all four configs
  - Subset chosen due to Groq free tier token limits (500K tokens/day)
- RAGAS version: 0.4.3
- LLM backend for RAGAS: Llama 3.1 8B Instruct via Groq (same model as generator)
- Embeddings for RAGAS: `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost)

---

## Results

### Retrieval Metrics (n=500 queries)

| Config | Recall@5 | MRR@5 | NDCG@5 |
|--------|----------|-------|--------|
| config1_dense | 0.1933 | 0.1392 | 0.1451 |
| config2_dense_rerank | 0.2515 | 0.2297 | 0.2256 |
| config3_hybrid | 0.3330 | 0.2270 | 0.2434 |
| config4_hybrid_rerank | 0.4092 | 0.3731 | 0.3691 |

### Generation Metrics — RAGAS Faithfulness (n=100 queries)

| Config | Faithfulness |
|--------|--------------|
| config1_dense | 0.2038 |
| config2_dense_rerank | 0.2674 |
| config3_hybrid | 0.4689 |
| config4_hybrid_rerank | 0.4618 |

Full results saved to:
- `results/evaluation_summary.csv`
- `results/config1_dense.json` — `results/config4_hybrid_rerank.json` (500 entries each: query, retrieved doc IDs, contexts, generated answer)

---

## Key Methodological Decisions (code-level)

1. **Query consistency**: `sampled_queries.json` created on config1's first run; all subsequent configs load from it — guarantees identical 500 queries
2. **RAGAS sample consistency**: `ragas_sample_ids.json` stores the 100 fixed query IDs drawn with seed=42 from the shared 500
3. **Checkpoint saves**: configs 3 and 4 save results every 50 queries to guard against mid-run token limit crashes
4. **Resume logic**: configs 3 and 4 check for existing partial output and skip completed query IDs on re-run
5. **Faithfulness batch size**: RAGAS run in batches of 50 with 60s sleep between batches to stay within Groq rate limits

---

## Files

```
~/rag_thesis/                         GitHub: https://github.com/Fola-l/MSc_RAG_Thesis
├── shared.py                         all retrievers, reranker, generator
├── config1_dense.py
├── config2_dense_rerank.py
├── config3_hybrid.py
├── config4_hybrid_rerank.py
├── evaluate.py
├── sampled_queries.json              500 query IDs
├── ragas_sample_ids.json             100 RAGAS query IDs
└── results/
    ├── config1_dense.json
    ├── config2_dense_rerank.json
    ├── config3_hybrid.json
    ├── config4_hybrid_rerank.json
    └── evaluation_summary.csv
```
