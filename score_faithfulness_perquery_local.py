"""
score_faithfulness_perquery_local.py

RAGAS faithfulness on the fixed 100-query sample (ragas_sample_ids.json),
scored against the four LOCAL (Ollama-generated) result files, using a LOCAL
Ollama judge model instead of the original Groq-hosted judge.

Local counterpart to score_faithfulness_perquery.py. Same methodology:
refusals excluded before scoring, per-query scores persisted (both for
resumability and for the Wilcoxon test in significance_tests_local.py).

Judge : llama3.1:8b  (local, via Ollama at localhost:11434)

Outputs
  results/faithfulness_per_query_local.csv
  results/faithfulness_cache_local.json
"""

# must monkey-patch before any ragas import
import sys, types
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    _stub = types.ModuleType('langchain_community.chat_models.vertexai')
    _stub.ChatVertexAI = type('ChatVertexAI', (), {})
    sys.modules['langchain_community.chat_models.vertexai'] = _stub

import os, json, csv, math, time, re, string, warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

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

JUDGE_MODEL     = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
BATCH_SIZE         = 5     # samples per ragas evaluate() call
MAX_BATCH_RETRIES  = 3

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

def score_config(config_name: str, samples: list, judge_llm, pq_path: str = None) -> list:
    """
    Score faithfulness for a list of non-refusal sample dicts.
    Each dict must have: query_id, query, answer, contexts.

    Returns list of dicts: {query_id, faithfulness (float or None)}.
    Processes in batches of BATCH_SIZE. If pq_path is given, appends each
    batch's rows to the CSV immediately so progress survives interruption.
    """
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness as faith_metric
    from datasets import Dataset

    faith_metric.llm = judge_llm
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=2)

    n        = len(samples)
    n_batch  = math.ceil(n / BATCH_SIZE)
    all_rows = []

    print(f"  {n} non-refusal samples → {n_batch} batches of ≤{BATCH_SIZE}")

    for b_idx in range(n_batch):
        start = b_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, n)
        batch = samples[start:end]

        raw_scores = None
        for attempt in range(1, MAX_BATCH_RETRIES + 1):
            data = {
                "user_input"        : [s["query"]    for s in batch],
                "response"          : [s["answer"]   for s in batch],
                "retrieved_contexts": [s["contexts"] for s in batch],
            }
            ds = Dataset.from_dict(data)

            t0 = time.time()
            try:
                result = evaluate(
                    ds,
                    metrics=[faith_metric],
                    run_config=run_config,
                    raise_exceptions=False,
                    show_progress=True,
                )
                raw_scores = list(result["faithfulness"])
            except Exception as e:
                print(f"  Batch {b_idx+1} attempt {attempt} EXCEPTION: {e}")
                raw_scores = [float('nan')] * len(batch)
            dt = time.time() - t0

            n_valid_now = sum(
                1 for s in raw_scores
                if s is not None and not math.isnan(float(s))
            )
            print(f"  Batch {b_idx+1} attempt {attempt}: {n_valid_now}/{len(batch)} valid ({dt:.1f}s)")

            if n_valid_now > 0:
                break
            if attempt < MAX_BATCH_RETRIES:
                print(f"  Retrying batch {b_idx+1}...")

        batch_result = []
        for sample, score in zip(batch, raw_scores):
            try:
                val = float(score)
            except (TypeError, ValueError):
                val = float('nan')
            batch_result.append({
                "query_id"    : sample["query_id"],
                "faithfulness": None if math.isnan(val) else val,
            })
        all_rows.extend(batch_result)

        # Persist this batch immediately — preserves progress on interruption
        if pq_path:
            with open(pq_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["query_id", "config", "faithfulness"])
                for row in batch_result:
                    w.writerow({
                        "query_id"    : row["query_id"],
                        "config"      : config_name,
                        "faithfulness": "" if row["faithfulness"] is None else row["faithfulness"],
                    })

        n_valid = sum(1 for r in batch_result if r["faithfulness"] is not None)
        vals    = [r["faithfulness"] for r in batch_result if r["faithfulness"] is not None]
        batch_mean = sum(vals) / len(vals) if vals else float('nan')
        print(f"  Batch {b_idx+1}/{n_batch}: {n_valid}/{len(batch)} valid  mean={batch_mean:.4f}")

    return all_rows

def main():
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    print(f"Judge model: {JUDGE_MODEL} (local via Ollama @ {OLLAMA_BASE_URL})")
    judge_llm = LangchainLLMWrapper(ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=1024,
    ))

    with open(os.path.join(BASE_DIR, "ragas_sample_ids.json")) as f:
        sampled_ids = set(json.load(f))
    print(f"Loaded {len(sampled_ids)} fixed RAGAS sample IDs (same 100 used for the original run)")

    faith_cache_path = os.path.join(RESULTS_DIR, "faithfulness_cache_local.json")
    faith_cache = {}
    if os.path.exists(faith_cache_path):
        with open(faith_cache_path) as f:
            faith_cache = json.load(f)
        print(f"Loaded faithfulness_cache_local.json: {list(faith_cache.keys())}")

    pq_path = os.path.join(RESULTS_DIR, "faithfulness_per_query_local.csv")
    existing_rows = []
    if os.path.exists(pq_path):
        with open(pq_path) as f:
            existing_rows = list(csv.DictReader(f))
        print(f"Loaded {len(existing_rows)} existing rows from faithfulness_per_query_local.csv")

    new_rows = list(existing_rows)

    for config_name in CONFIGS:
        if config_name in faith_cache:
            print(f"\n{'='*60}\n {config_name}  → CACHED (skipping)\n{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f" {config_name}")
        print(f"{'='*60}")

        path = os.path.join(RESULTS_DIR, f"{config_name}_local.json")
        with open(path) as f:
            results = json.load(f)

        sampled = [r for r in results if r['query_id'] in sampled_ids]
        print(f"  Sampled:    {len(sampled)}/100")

        non_refusals = [r for r in sampled if not is_refusal(r.get('answer', ''))]
        refusals     = [r for r in sampled if     is_refusal(r.get('answer', ''))]
        print(f"  Non-refusals to score: {len(non_refusals)}")
        print(f"  Refusals excluded    : {len(refusals)}")

        if not non_refusals:
            print("  ⚠ No non-refusal answers — skipping config")
            faith_cache[config_name] = {"n_nonrefusal": 0, "n_valid": 0, "mean": None}
            with open(faith_cache_path, "w") as f:
                json.dump(faith_cache, f, indent=2)
            continue

        # Intra-config resume: skip samples already in the CSV for this config
        attempted_ids = {r['query_id'] for r in new_rows if r.get('config') == config_name}
        pending = [s for s in non_refusals if s['query_id'] not in attempted_ids]
        if attempted_ids:
            print(f"  Resuming: {len(attempted_ids)} already attempted, {len(pending)} pending")

        scored = score_config(config_name, pending, judge_llm, pq_path=pq_path) \
                 if pending else []

        for row in scored:
            new_rows.append({
                "query_id"    : row["query_id"],
                "config"      : config_name,
                "faithfulness": row["faithfulness"] if row["faithfulness"] is not None else "",
            })

        n_valid = 0
        sum_val = 0.0
        for r in new_rows:
            if r.get('config') == config_name and r.get('faithfulness') not in ('', None):
                try:
                    v = float(r['faithfulness'])
                    n_valid += 1
                    sum_val += v
                except (ValueError, TypeError):
                    pass

        new_mean = round(sum_val / n_valid, 4) if n_valid else None
        print(f"\n  n non-refusals: {len(non_refusals)}")
        print(f"  n valid scores: {n_valid}")
        print(f"  Mean faithfulness: {new_mean}")

        faith_cache[config_name] = {
            "n_nonrefusal": len(non_refusals),
            "n_valid"     : n_valid,
            "mean"        : new_mean,
        }
        with open(faith_cache_path, "w") as f:
            json.dump(faith_cache, f, indent=2)
        print(f"  ✓ Cached (n_valid={n_valid})")

        with open(pq_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "config", "faithfulness"])
            w.writeheader()
            w.writerows(new_rows)
        print(f"  ✓ Per-query CSV updated ({len(new_rows)} total rows so far)")

    print(f"\n{'='*60}")
    print(f" FAITHFULNESS SUMMARY  (local judge: {JUDGE_MODEL})")
    print(f"{'='*60}")
    for c in CONFIGS:
        v = faith_cache.get(c, {})
        mean_s = f"{v.get('mean'):.4f}" if v.get('mean') is not None else "N/A"
        print(f"  {SHORT[c]:<4} {c:<25} mean={mean_s}  n_valid={v.get('n_valid')}/{v.get('n_nonrefusal')}")

    print(f"\nDone.")

if __name__ == "__main__":
    main()
