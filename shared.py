import os

# Must be set before faiss/OpenMP/tokenizer threads are spawned
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
import random
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

load_dotenv()
random.seed(42)
np.random.seed(42)

# ── PATHS ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH  = os.path.join(BASE_DIR, "docs_100k.json")
FAISS_PATH = os.path.join(BASE_DIR, "faiss_100k_v2.bin")
BM25_PATH  = os.path.join(BASE_DIR, "bm25_100k.pkl")

# ── LOAD DOCS (always needed) ─────────────────────────────────
print("Loading docs...")
with open(DOCS_PATH, "r") as f:
    data = json.load(f)

doc_ids   = data["ids"]
doc_texts = data["texts"]
print(f"Loaded {len(doc_texts)} documents")

# ── GROQ CLIENT (always needed) ───────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── CONDITIONAL LOADERS ───────────────────────────────────────
faiss_index    = None
question_encoder = None
bm25           = None
reranker       = None

def load_faiss():
    global faiss_index
    if faiss_index is None:
        print("Loading FAISS index...")
        faiss_index = faiss.read_index(FAISS_PATH)
        print(f"FAISS loaded: {faiss_index.ntotal} vectors")

def load_question_encoder():
    global question_encoder
    if question_encoder is None:
        print("Loading question encoder...")
        question_encoder = SentenceTransformer(
            'facebook-dpr-question_encoder-single-nq-base',
            device='cpu'
        )
        print("Question encoder loaded")

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

def load_reranker():
    global reranker
    if reranker is None:
        print("Loading reranker...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Reranker loaded")

# ── ENCODE QUERIES ────────────────────────────────────────────
def encode_queries(query_texts):
    load_question_encoder()
    load_faiss()
    print("Encoding queries...")
    embeddings = question_encoder.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    # Force contiguous float32 — fixes FAISS segfault on Apple Silicon
    return np.ascontiguousarray(embeddings, dtype=np.float32)

# ── RETRIEVAL FUNCTIONS ───────────────────────────────────────
def dense_retrieve(query_embedding, top_k=10):
    # Explicit contiguous float32 array — critical for FAISS stability
    q_emb = np.ascontiguousarray(
        query_embedding.reshape(1, -1), dtype=np.float32
    )
    scores, indices = faiss_index.search(q_emb, top_k)
    return [(doc_ids[i], doc_texts[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])]

def rerank(query_text, candidate_docs, top_k=5):
    pairs  = [(query_text, text) for _, text, _ in candidate_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    return [(doc_id, text, float(score)) for score, (doc_id, text, _) in ranked[:top_k]]

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

# ── GENERATOR ─────────────────────────────────────────────────
def generate_answer(query, contexts):
    context_str = "\n\n".join([f"Context {i+1}: {c}"
                               for i, c in enumerate(contexts)])
    prompt = f"""Answer the question based only on the provided context.
Be concise and factual. If the context does not contain the answer, say "I don't know".

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