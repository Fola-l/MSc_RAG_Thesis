"""
smoketest_judge_models.py

Smoke test: RAGAS Faithfulness scoring on 10 queries (6 answered, 4 refused)
from results/config1_dense_local.json, run once with phi3:medium and once
with mistral-small3.1 as the local Ollama judge — candidates being evaluated
to replace/complement llama3.1:8b as the RAGAS judge.

Same scoring approach as score_faithfulness_perquery_local.py (ragas
faithfulness metric, LangchainLLMWrapper(ChatOllama(...))), but run one
query at a time (not batched) so per-query timing and validity are visible
individually, and no resume/checkpoint machinery since this is a one-off test.

This does NOT touch any existing result file. Output: printed comparison
table + results/judge_smoketest_local.csv (new file, for reference only).
"""

import sys, types
if 'langchain_community.chat_models.vertexai' not in sys.modules:
    _stub = types.ModuleType('langchain_community.chat_models.vertexai')
    _stub.ChatVertexAI = type('ChatVertexAI', (), {})
    sys.modules['langchain_community.chat_models.vertexai'] = _stub

import os, json, csv, math, time, warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

OLLAMA_BASE_URL = "http://localhost:11434"
JUDGES = {
    "phi3"    : "phi3:medium",
    "mistral" : "mistral-small3.1",
}

QUERY_IDS = [
    "test2", "test8", "test13", "test14", "test26", "test32",   # answered
    "test29", "test38", "test102", "test108",                    # refused
]

def score_one(judge_llm, sample):
    """Run RAGAS faithfulness on a single sample. Returns (score_or_None, elapsed_s, error_or_None)."""
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness as faith_metric
    from datasets import Dataset

    faith_metric.llm = judge_llm
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=1)

    ds = Dataset.from_dict({
        "user_input"        : [sample["query"]],
        "response"          : [sample["answer"]],
        "retrieved_contexts": [sample["contexts"]],
    })

    t0 = time.time()
    try:
        result = evaluate(ds, metrics=[faith_metric], run_config=run_config,
                           raise_exceptions=False, show_progress=False)
        raw = list(result["faithfulness"])[0]
        elapsed = time.time() - t0
        try:
            val = float(raw)
            if math.isnan(val):
                return None, elapsed, "NaN returned (parsing/scoring failure inside ragas)"
            return val, elapsed, None
        except (TypeError, ValueError):
            return None, elapsed, f"unparseable score: {raw!r}"
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, f"EXCEPTION: {type(e).__name__}: {e}"

def main():
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    print("Loading config1_dense_local.json...")
    with open(os.path.join(RESULTS_DIR, "config1_dense_local.json")) as f:
        results = {r["query_id"]: r for r in json.load(f)}

    samples = []
    for qid in QUERY_IDS:
        r = results[qid]
        samples.append({
            "query_id": qid,
            "query"   : r["query"],
            "answer"  : r["answer"],
            "contexts": r["contexts"],
        })
    print(f"Selected {len(samples)} queries: {QUERY_IDS}\n")

    judges = {}
    for name, model in JUDGES.items():
        print(f"Initializing judge '{name}' = {model} @ {OLLAMA_BASE_URL}")
        judges[name] = LangchainLLMWrapper(ChatOllama(
            model=model, base_url=OLLAMA_BASE_URL, temperature=0.0, num_predict=1024,
        ))

    rows = []
    for i, sample in enumerate(samples, 1):
        qid = sample["query_id"]
        print(f"\n[{i}/10] {qid}  —  \"{sample['query'][:60]}\"")
        print(f"        answer: \"{sample['answer'][:70]}\"")

        row = {"query_id": qid}
        for name in JUDGES:
            score, elapsed, err = score_one(judges[name], sample)
            valid = score is not None
            row[f"{name}_score"] = round(score, 4) if valid else ""
            row[f"{name}_valid"] = valid
            row[f"time_{name}"]  = round(elapsed, 1)
            row[f"{name}_error"] = err or ""

            status = f"score={score:.4f}" if valid else f"INVALID ({err})"
            print(f"        [{name:<8}] {status}  ({elapsed:.1f}s)")

        rows.append(row)

    # ── COMPARISON TABLE ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(" COMPARISON TABLE")
    print(f"{'='*100}")
    hdr = (f"  {'query_id':<10} {'phi3_score':>10} {'phi3_valid':>10} "
           f"{'mistral_score':>13} {'mistral_valid':>13} {'time_phi3':>9} {'time_mistral':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['query_id']:<10} "
              f"{(r['phi3_score'] if r['phi3_score'] != '' else 'N/A'):>10} "
              f"{str(r['phi3_valid']):>10} "
              f"{(r['mistral_score'] if r['mistral_score'] != '' else 'N/A'):>13} "
              f"{str(r['mistral_valid']):>13} "
              f"{r['time_phi3']:>9} "
              f"{r['time_mistral']:>12}")

    n_phi3_valid    = sum(1 for r in rows if r["phi3_valid"])
    n_mistral_valid = sum(1 for r in rows if r["mistral_valid"])
    avg_time_phi3    = sum(r["time_phi3"] for r in rows) / len(rows)
    avg_time_mistral = sum(r["time_mistral"] for r in rows) / len(rows)

    print(f"\n  phi3:medium      — {n_phi3_valid}/10 valid, avg {avg_time_phi3:.1f}s/query")
    print(f"  mistral-small3.1 — {n_mistral_valid}/10 valid, avg {avg_time_mistral:.1f}s/query")

    errs = [(r['query_id'], name, r[f'{name}_error']) for r in rows for name in JUDGES if r[f'{name}_error']]
    if errs:
        print(f"\n  Errors/malformed output:")
        for qid, name, err in errs:
            print(f"    {qid} [{name}]: {err}")
    else:
        print(f"\n  No errors or malformed output.")

    # ── SAVE (new file only, does not touch any existing result file) ──
    out_path = os.path.join(RESULTS_DIR, "judge_smoketest_local.csv")
    fields = ["query_id", "phi3_score", "phi3_valid", "mistral_score", "mistral_valid",
              "time_phi3", "time_mistral", "phi3_error", "mistral_error"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Saved: {out_path}")

if __name__ == "__main__":
    main()
