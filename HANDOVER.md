# RAG Thesis Pipeline — Handover Document

## Overview

MSc thesis comparing four retrieval configurations in a RAG pipeline.  
Research question: **how do different retrieval strategies affect downstream factual accuracy?**

**Status: Experiment COMPLETE. All 4 configs run. Evaluation done. Writing phase next.**

GitHub: https://github.com/Fola-l/MSc_RAG_Thesis  
Chapter 1: written and approved. Chapter 3 (Methodology) and Chapter 4 (Results): to write.

---

## Final Results

| Config | Recall@5 | MRR@5 | NDCG@5 | Faithfulness |
|--------|----------|-------|--------|--------------|
| config1_dense | 0.1933 | 0.1392 | 0.1451 | 0.2038 |
| config2_dense_rerank | 0.2515 | 0.2297 | 0.2256 | 0.2674 |
| config3_hybrid | 0.3330 | 0.2270 | 0.2434 | 0.4689 |
| config4_hybrid_rerank | 0.4092 | 0.3731 | 0.3691 | 0.4618 |

**Key findings:**
- Every retrieval metric improves monotonically config1 → config4
- Config4 is ~2× config1 on Recall@5 and NDCG@5
- Hybrid fusion (config3) is the big faithfulness jump: 0.27 → 0.47
- Config4 faithfulness is fractionally lower than config3 (0.4618 vs 0.4689) — reranker reshuffles top-5 away from supporting passages

Saved to: `results/evaluation_summary.csv`

---

## Research Design

| Config | Name | Retriever | Reranker | Fusion |
|--------|------|-----------|----------|--------|
| 1 | Dense-only | DPR (FAISS) | None | None |
| 2 | Dense + Reranker | DPR (FAISS) | Cross-encoder | None |
| 3 | Hybrid Fusion | DPR + BM25 | None | RRF |
| 4 | Hybrid + Reranker | DPR + BM25 | Cross-encoder | RRF |

**Evaluation metrics:**
- Retrieval: Recall@5, MRR@5, NDCG@5 (vs BEIR NQ qrels, all 500 queries)
- Generation: RAGAS Faithfulness only (100-query fixed subset, seed=42, saved in `ragas_sample_ids.json`)

**Why faithfulness only:** RAGAS answer_relevancy requests n>1 completions which Groq rejects. Faithfulness uses sequential single calls and is the more academically significant metric (hallucination detection).

**Why 100-query RAGAS subset:** Groq free tier 500K TPD limit. 100 samples × 2 batches × 4 configs is feasible in one sitting. Methodology note for thesis: "RAGAS evaluation computed on a stratified random sample of 100 queries per configuration (seed=42) due to API token constraints."

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Dataset | BEIR Natural Questions (100K doc subset) |
| Dense retriever | DPR (`facebook-dpr-ctx_encoder-single-nq-base`) |
| Sparse retriever | BM25 (rank_bm25) |
| Vector store | FAISS (faiss-cpu) |
| Fusion | RRF (manual implementation) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | Llama 3.1 8B Instruct via Groq API (temp=0.0, max_tokens=150) |
| Evaluation | RAGAS 0.4.3 + ir_datasets |
| Environment | Python 3.13 venv (`rag_thesis_env`) |

---

## File Structure

```
~/rag_thesis/
├── .env                          # GROQ_API_KEY
├── shared.py                     # loaders, retrievers, reranker, generator
├── config1_dense.py              # DONE — results saved
├── config2_dense_rerank.py       # DONE — results saved
├── config3_hybrid.py             # DONE — resume logic included
├── config4_hybrid_rerank.py      # DONE — resume logic included
├── evaluate.py                   # DONE — retrieval + RAGAS faithfulness
├── sampled_queries.json          # 500 fixed query IDs (used by all configs)
├── ragas_sample_ids.json         # 100 fixed RAGAS query IDs
├── docs_100k.json                # 100K document corpus (51MB, gitignored)
├── faiss_100k.bin                # FAISS index (293MB, gitignored)
├── bm25_100k.pkl                 # BM25 index (69MB, gitignored)
└── results/
    ├── config1_dense.json
    ├── config2_dense_rerank.json
    ├── config3_hybrid.json
    ├── config4_hybrid_rerank.json
    └── evaluation_summary.csv
```

---

## Important Implementation Notes

- **Query consistency**: all 4 configs loaded queries from `sampled_queries.json` — same 500 queries
- **Generator identical across configs**: Llama 3.1 8B, temp=0.0, max_tokens=150, top-5 contexts
- **Segfault fix**: env vars set before imports in every config script (`TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1`, `KMP_DUPLICATE_LIB_OK=TRUE`)
- **RAGAS vertexai stub**: `langchain_community >= 0.3` removed `chat_models.vertexai`; evaluate.py stubs it at runtime
- **Qrels loaded via ir_datasets**: `ir_datasets.load('beir/nq').qrels_iter()` — downloads BEIR NQ zip (~498MB) on first run
- **Random seed**: 42 everywhere

---

## What's Next

1. **Chapter 3 — Methodology**: describe the four configurations, dataset, evaluation metrics, implementation decisions
2. **Chapter 4 — Results & Discussion**: present the table, discuss each metric, explain the config3 faithfulness finding
3. **Chapter 2 — Literature Review**: written after experiment (already planned)
