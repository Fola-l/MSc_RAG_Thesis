#!/bin/bash
set -e
cd /Users/lad/Documents/rag_thesis_local
PY=./rag_thesis_env/bin/python3

echo "=== [$(date)] Starting config1_dense_local ==="
$PY config1_dense.py
echo "=== [$(date)] Starting config2_dense_rerank_local ==="
$PY config2_dense_rerank.py
echo "=== [$(date)] Starting config3_hybrid_local ==="
$PY config3_hybrid.py
echo "=== [$(date)] Starting config4_hybrid_rerank_local ==="
$PY config4_hybrid_rerank.py
echo "=== [$(date)] ALL CONFIGS COMPLETE ==="
