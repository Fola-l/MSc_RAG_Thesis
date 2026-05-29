# RAG Thesis Pipeline — Claude Code Handover Document

## Overview

This is an MSc thesis project comparing four retrieval configurations in a
Retrieval-Augmented Generation (RAG) pipeline. The research question is:
**how do different retrieval strategies affect downstream factual accuracy
of generated answers?**

The thesis has Chapter 1 written and approved. Chapter 3 (Methodology) is
being built now by constructing the actual experiment pipeline. Chapter 2
(Literature Review) will be written after the experiment is complete.

---

## Research Design

### Four Configurations Being Compared

| Config | Name | Retriever | Reranker | Fusion |
|--------|------|-----------|----------|--------|
| 1 | Dense-only | DPR (FAISS) | None | None |
| 2 | Dense + Reranker | DPR (FAISS) | Cross-encoder | None |
| 3 | Hybrid Fusion | DPR + BM25 | None | RRF |
| 4 | Hybrid + Reranker | DPR + BM25 | Cross-encoder | RRF |

### Evaluation Metrics
- **Retrieval level**: Recall@K, MRR, NDCG
- **Generation level**: Faithfulness, Answer Relevance, Context Precision (via RAGAS)

---

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Dataset | BEIR Natural Questions (100K doc subset) | Loaded from HuggingFace |
| Dense retriever | DPR (`facebook-dpr-ctx_encoder-single-nq-base`) | via sentence-transformers |
| Sparse retriever | BM25 | via rank_bm25 |
| Vector store | FAISS | faiss-cpu |
| Fusion | RRF (Reciprocal Rank Fusion) | implemented manually |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | NOT YET IMPLEMENTED |
| Generator | Llama 3.1 8B Instruct | via Groq API (free tier) |
| Framework | LangChain | installed, not yet used in scripts |
| Evaluation | RAGAS | installed, not yet implemented |
| Environment | Python venv (`rag_thesis_env`) | inside ~/rag_thesis/ |

---

## Hardware

- **Machine**: MacBook Pro M4 Pro, 16GB unified memory
- **Python version**: 3.13 (potential issue — see Current Blocker below)
- **Groq API key**: stored in `.env` file as `GROQ_API_KEY`

---

## Project File Structure

```
~/rag_thesis/
├── .env                        # GROQ_API_KEY stored here
├── shared.py                   # shared loaders, retrievers, generator
├── config1_dense.py            # Configuration 1 script
├── config2_dense_rerank.py     # Configuration 2 script (empty)
├── config3_hybrid.py           # Configuration 3 script (empty)
├── config4_hybrid_rerank.py    # Configuration 4 script (empty)
├── evaluate.py                 # evaluation script (empty)
├── docs_100k.json              # 100K document texts + IDs (saved corpus)
├── faiss_100k.bin              # FAISS index for 100K docs (pre-built)
├── bm25_100k.pkl               # BM25 index for 100K docs (pre-built)
├── results/                    # output JSONs go here
└── rag_thesis_env/             # Python virtual environment
```

---

## Current State of Each File

### `shared.py` — WRITTEN, has segfault issue (see below)

Contains:
- Doc loading from `docs_100k.json`
- Conditional loaders: `load_faiss()`, `load_encoder()`, `load_bm25()`
- `encode_queries(query_texts)` — batch encodes all queries upfront
- `dense_retrieve(query_embedding, top_k)` — FAISS search
- `bm25_retrieve(query_text, top_k)` — BM25 search
- `rrf_fusion(dense_results, bm25_results, k=60)` — RRF implementation
- `generate_answer(query, contexts)` — Groq/Llama 3.1 8B generator
- Groq client initialisation

### `config1_dense.py` — WRITTEN, crashes with segfault

Contains:
- Loads FAISS + encoder only (conditional loading working)
- Loads 500 sampled queries from BEIR NQ
- Encodes all queries upfront as batch
- Loops through queries: retrieve → generate → save
- Saves results to `results/config1_dense.json`

### `config2_dense_rerank.py` — EMPTY
### `config3_hybrid.py` — EMPTY
### `config4_hybrid_rerank.py` — EMPTY
### `evaluate.py` — EMPTY

---

## Current Blocker — Segfault on FAISS Search

### Symptom
```
Running Configuration 1: Dense-Only...
zsh: segmentation fault  python3 config1_dense.py
```

Crash happens on the **first call to `dense_retrieve()`** inside the query loop,
immediately after encoding completes successfully.

### What Works Fine
- Loading docs ✓
- Loading FAISS index ✓
- Loading encoder ✓
- Loading BM25 ✓
- Loading queries ✓
- Encoding all 500 queries as batch ✓
- Groq API connection ✓ (tested separately)

### What Crashes
- First `dense_retrieve()` call using the pre-encoded query embedding

### Root Cause Analysis
Three-way conflict on Apple Silicon macOS:
1. **Python 3.13** — very new, faiss-cpu has known instability
2. **joblib semaphore leak** — sentence-transformers leaves joblib workers
   running; FAISS threading conflicts with them
3. **Apple Silicon memory management** — non-contiguous numpy arrays
   cause FAISS to segfault

Evidence: leaked semaphore warning appears every crash:
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean
up at shutdown: {'/loky-XXXX-XXXXXXXX'}
```

### Fixes Already Attempted
1. Forced `device='cpu'` on encoder — did not fix
2. Used `np.ascontiguousarray(..., dtype=np.float32)` before FAISS search — did not fix
3. Added environment variables at top of script:
   ```python
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
   ```
   — not yet confirmed if this fixes it (try this first)

### Suggested Fix Options (in order of preference)

**Option A — Environment variables (try first)**
Add to top of `config1_dense.py` before all imports:
```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

**Option B — Replace FAISS with numpy search**
For 100K docs, pure numpy cosine similarity is fast enough and completely
stable. Extract embeddings matrix from FAISS index using
`faiss_index.reconstruct_n()` then use `embeddings_matrix @ q_emb.T`
for similarity search. No threading, no segfault risk.

**Option C — Downgrade to Python 3.11**
Nuclear option. Recreate venv with Python 3.11 which is rock solid
with faiss-cpu. Reinstall all dependencies.
```bash
deactivate
cd ~/rag_thesis
rm -rf rag_thesis_env
python3.11 -m venv rag_thesis_env
source rag_thesis_env/bin/activate
pip install langchain langchain-community groq langchain-groq ragas \
    sentence-transformers rank-bm25 faiss-cpu datasets python-dotenv
```

---

## Full Script Contents

### `shared.py` (current version)

```python
import os
import json
import pickle
import random
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

load_dotenv()
random.seed(42)
np.random.seed(42)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH  = os.path.join(BASE_DIR, "docs_100k.json")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_100k.bin")
BM25_PATH  = os.path.join(BASE_DIR, "bm25_100k.pkl")

print("Loading docs...")
with open(DOCS_PATH, "r") as f:
    data = json.load(f)

doc_ids   = data["ids"]
doc_texts = data["texts"]
print(f"Loaded {len(doc_texts)} documents")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

faiss_index = None
encoder     = None
bm25        = None

def load_faiss():
    global faiss_index
    if faiss_index is None:
        print("Loading FAISS index...")
        faiss_index = faiss.read_index(FAISS_PATH)
        print(f"FAISS loaded: {faiss_index.ntotal} vectors")

def load_encoder():
    global encoder
    if encoder is None:
        print("Loading encoder...")
        encoder = SentenceTransformer(
            'facebook-dpr-ctx_encoder-single-nq-base',
            device='cpu'
        )
        print("Encoder loaded")

def load_bm25():
    global bm25
    if bm25 is None:
        if os.path.exists(BM25_PATH):
            print("Loading BM25 index...")
            with open(BM25_PATH, "rb") as f:
                bm25 = pickle.load(f)
            print("BM25 loaded")
        else:
            print("Building BM25 index (first time only)...")
            tokenized_docs = [doc.lower().split() for doc in doc_texts]
            bm25 = BM25Okapi(tokenized_docs)
            with open(BM25_PATH, "wb") as f:
                pickle.dump(bm25, f)
            print("BM25 built and saved")

def encode_queries(query_texts):
    load_encoder()
    load_faiss()
    print("Encoding queries...")
    embeddings = encoder.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)

def dense_retrieve(query_embedding, top_k=10):
    q_emb = np.ascontiguousarray(
        query_embedding.reshape(1, -1), dtype=np.float32
    )
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, top_k)
    return [(doc_ids[i], doc_texts[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])]

def bm25_retrieve(query_text, top_k=10):
    load_bm25()
    tokenized_query = query_text.lower().split()
    scores          = bm25.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[i], doc_texts[i], float(scores[i]))
            for i in top_indices]

def rrf_fusion(dense_results, bm25_results, k=60):
    scores       = {}
    doc_text_map = {}
    for rank, (doc_id, text, _) in enumerate(dense_results):
        scores[doc_id]       = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_text_map[doc_id] = text
    for rank, (doc_id, text, _) in enumerate(bm25_results):
        scores[doc_id]       = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_text_map[doc_id] = text
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, doc_text_map[doc_id], score)
            for doc_id, score in sorted_docs]

def generate_answer(query, contexts):
    context_str = "\n\n".join([f"Context {i+1}: {c}"
                               for i, c in enumerate(contexts)])
    prompt = f"""Answer the question based only on the provided context.
Be concise and factual. If the context does not contain the answer,
say "I don't know".

{context_str}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.0
    )
    return response.choices[0].message.content
```

### `config1_dense.py` (current version)

```python
"""
Configuration 1: Dense-Only Retrieval
- Retriever : DPR (Dense Passage Retrieval)
- Reranker  : None
- Fusion    : None
"""

import json
import time
import random
from datasets import load_dataset
from shared import (
    load_faiss,
    load_encoder,
    dense_retrieve,
    encode_queries,
    generate_answer,
    doc_ids,
    doc_texts
)

random.seed(42)

load_faiss()
load_encoder()

print("Loading queries...")
queries_full = load_dataset('BeIR/nq', 'queries', split='queries')
queries      = random.sample(list(queries_full), 500)
print(f"Loaded {len(queries)} queries")

query_texts      = [q['text'] for q in queries]
query_embeddings = encode_queries(query_texts)

results = []

print("\nRunning Configuration 1: Dense-Only...")
for i, query_item in enumerate(queries):
    query_id   = query_item['_id']
    query_text = query_item['text']
    query_emb  = query_embeddings[i:i+1]

    retrieved         = dense_retrieve(query_emb, top_k=5)
    contexts          = [text for _, text, _ in retrieved]
    doc_ids_retrieved = [doc_id for doc_id, _, _ in retrieved]

    answer = generate_answer(query_text, contexts)

    results.append({
        "query_id"     : query_id,
        "query"        : query_text,
        "retrieved_ids": doc_ids_retrieved,
        "contexts"     : contexts,
        "answer"       : answer,
        "config"       : "config1_dense"
    })

    if (i + 1) % 50 == 0:
        print(f"  Progress: {i+1}/500 queries done")

    time.sleep(0.5)

output_path = "results/config1_dense.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Config 1 complete. Results saved to {output_path}")
print(f"  Total queries processed: {len(results)}")
```

---

## What Still Needs to Be Built

### 1. Fix segfault (immediate)
See options above.

### 2. `config2_dense_rerank.py`
Same as config1 but after dense retrieval, pass top-10 results through
cross-encoder reranker, then take top-5 for generation.
Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

### 3. `config3_hybrid.py`
Run dense_retrieve(top_k=10) + bm25_retrieve(top_k=10),
fuse with rrf_fusion(), take top-5 for generation.

### 4. `config4_hybrid_rerank.py`
Run dense + BM25, fuse with RRF, rerank top-10, take top-5 for generation.

### 5. `evaluate.py`
Load all four result JSONs from results/.
Compute RAGAS metrics: faithfulness, answer_relevance, context_precision.
Compute retrieval metrics: Recall@5, MRR against BEIR qrels.
Output a summary table to results/evaluation_summary.csv.

### 6. Query consistency across configs
All four configs must use the **exact same 500 queries**.
Fix: save sampled query IDs to a file on first run, load from it on all runs.
```python
QUERIES_PATH = "sampled_queries.json"
if os.path.exists(QUERIES_PATH):
    with open(QUERIES_PATH) as f:
        query_ids = json.load(f)
    queries = [q for q in queries_full if q['_id'] in set(query_ids)]
else:
    queries = random.sample(list(queries_full), 500)
    with open(QUERIES_PATH, "w") as f:
        json.dump([q['_id'] for q in queries], f)
```

---

## Important Academic Constraints

- Generator model **must stay identical** across all four configs
  (Llama 3.1 8B Instruct via Groq, temperature=0.0, max_tokens=150)
- Top-k passed to generator **must stay identical** (top-5 contexts)
- Random seed **must be 42** everywhere for reproducibility
- Results must be saved as JSON with full query, retrieved docs,
  and generated answer for later evaluation

---

## Installed Packages

```
langchain
langchain-community
groq
langchain-groq
ragas
sentence-transformers
rank-bm25
faiss-cpu
datasets
python-dotenv
```

---

## Notes for Claude Code

- The `.env` file exists and contains a valid `GROQ_API_KEY`
- `faiss_100k.bin` and `bm25_100k.pkl` and `docs_100k.json` are all
  pre-built and saved — do not rebuild them unless necessary
- The FAISS index was built on Colab T4 GPU and contains 100K document
  embeddings from the BEIR NQ corpus
- The BM25 index was built locally and covers the same 100K documents
- All scripts run from `~/rag_thesis/` directory with venv activated
- Groq free tier rate limit: ~30 requests/minute — the 0.5s sleep handles this
