"""Query rewriting benchmark: does an LLM rewrite of the question retrieve the gold
sections better than the question itself, and what does it cost?

    python -m eval.bench_rewrite               # uses cached rewrites when available
    python -m eval.bench_rewrite --refresh     # call the LLM again for every question
    python -m eval.bench_rewrite --use-stored  # replay the stored rewrites, no LLM at all (also
                                               # if the rewrite prompt changed since they were made)
    python -m eval.bench_rewrite --pace 5      # seconds between LLM calls (rate limits)

Step 1 (LLM, once per question, cached in eval/results/rewrites.jsonl): the rewriter
gets the company's policy outline + the question, returns 1-3 search queries.
The cache key includes the model and a hash of the prompt, so changing either
re-runs the LLM; every variant below uses the SAME rewrites.

Step 2 (no LLM): retrieval variants, all hybrid search, top 3:
    baseline         the original question only (what the app does today)
    A rewrite-only   only the rewritten queries, fused with RRF
    B original+rw    the original question + the rewritten queries, fused with RRF
    C / D            as A / B, but merged round-robin (each query's best hit in turn:
                     coverage) instead of RRF (agreement)
If a rewrite failed, A and B fall back to the original question (as the app would).

Reports the three golden metrics plus the cost: rewrite latency (the LLM call) and
total = rewrite + retrieval.
"""
import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from app.config import PRIMARY_MODEL, get_settings
from app.llm import build_llm
from app.retriever import TOP_K, rank_query, retrieve, round_robin, rrf
from app.rewriter import SYSTEM_PROMPT, rewrite_query
from app.store import PolicyStore
from app.versioning import ingest_all
from eval.run_retrieval import CATEGORIES, RESULTS_DIR, load_questions, score_case, summarise, validate_gold

CACHE = RESULTS_DIR / "rewrites.jsonl"
VARIANTS = [("baseline (original question)", None, None),
            ("A: rewrites only, RRF", False, "rrf"),
            ("B: original + rewrites, RRF", True, "rrf"),
            ("C: rewrites only, round-robin", False, "round_robin"),
            ("D: original + rw, round-robin", True, "round_robin")]


def prompt_key(model: str) -> str:
    return hashlib.sha256(f"{model}\n{SYSTEM_PROMPT}".encode()).hexdigest()[:12]


def load_cache(key: str | None) -> dict:
    """Stored rewrites made with this exact model + prompt (key=None: all stored rewrites)."""
    if not CACHE.exists():
        return {}
    rows = [json.loads(line) for line in CACHE.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows if key is None or r.get("key") == key}


def merged_top3(store, company: str, queries: list[str], fusion: str) -> list:
    """Rank the company's chunks for each query (hybrid), then merge the rankings by
    round-robin (each query's best first) or by RRF (agreement), and keep 3."""
    rankings = [rank_query(store, company, None, q)[0] for q in queries]
    if fusion == "round_robin":
        return round_robin(rankings)[:TOP_K]
    by_id = {c.id: c for ranking in rankings for c in ranking}
    order = rrf([[c.id for c in ranking] for ranking in rankings])
    return [by_id[i] for i in order[:TOP_K]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--use-stored", action="store_true")
    ap.add_argument("--pace", type=float, default=5.0)
    args = ap.parse_args()

    s = get_settings()
    cases = load_questions()
    validate_gold(cases, s.policies_dir)
    llm, model = build_llm(s), PRIMARY_MODEL
    key = prompt_key(model)
    if args.use_stored:
        cache = load_cache(None)
        if len(cache) < 90:
            raise SystemExit("--use-stored: eval/results/rewrites.jsonl does not hold all 90 rewrites")
    elif args.refresh:
        cache = {}
    else:
        cache = load_cache(key)

    with tempfile.TemporaryDirectory() as tmp:
        store = PolicyStore(Path(tmp))
        ingest_all(store, s.policies_dir)
        retrieve(store, "acme", ["warm up"])

        # ---- step 1: rewrites (LLM) ----
        rewrites = {}
        for i, c in enumerate(cases, 1):
            if c["id"] in cache:
                rewrites[c["id"]] = cache[c["id"]]
                continue
            t0 = time.perf_counter()
            rw = rewrite_query(llm, store, c["company"], c["question"], model, s.llm_timeout_s)
            ms = (time.perf_counter() - t0) * 1000
            rewrites[c["id"]] = {"id": c["id"], "key": key, "model": model, "question": c["question"],
                                 "queries": rw.queries, "ok": rw.ok, "error": rw.error,
                                 "latency_ms": round(ms, 1)}
            print(f"[{i:>2}/{len(cases)}] {c['id']:9} {ms:6.0f} ms  {rw.queries if rw.ok else 'FAILED ' + rw.error}",
                  flush=True)
            if i < len(cases):
                time.sleep(args.pace)
        RESULTS_DIR.mkdir(exist_ok=True)
        CACHE.write_text("\n".join(json.dumps(rewrites[c["id"]]) for c in cases) + "\n")

        # ---- step 2: retrieval variants (no LLM) ----
        results = []
        for label, include_original, fusion in VARIANTS:
            rows = []
            for c in cases:
                rw = rewrites[c["id"]]
                queries = rw["queries"] if rw["ok"] else []
                t0 = time.perf_counter()
                if include_original is None or not queries:
                    top = retrieve(store, c["company"], [c["question"]]).chunks
                else:
                    searched = [c["question"], *queries] if include_original else queries
                    top = merged_top3(store, c["company"], searched, fusion)
                retrieval_ms = (time.perf_counter() - t0) * 1000
                rewrite_ms = 0.0 if include_original is None else rw["latency_ms"]
                ranked = [ch.section for ch in top]
                leaks = [ch.id for ch in top if ch.company_id != c["company"]]
                row = {"id": c["id"], "input": {"company": c["company"], "question": c["question"]},
                       "expected": {"required": c["required"], "category": c["category"]},
                       **score_case(c, ranked, leaks, retrieval_ms + rewrite_ms)}
                row["actual"]["retrieval_ms"] = round(retrieval_ms, 1)
                row["actual"]["rewrite_ms"] = rewrite_ms
                rows.append(row)
            summ = summarise(rows)
            results.append({"label": label, **summ,
                            "avg_retrieval_ms": round(statistics.mean(r["actual"]["retrieval_ms"] for r in rows), 1),
                            "avg_rewrite_ms": round(statistics.mean(r["actual"]["rewrite_ms"] for r in rows), 1),
                            "failed": [r["id"] for r in rows if not r["pass"]]})

    ok = [r for r in rewrites.values() if r["ok"]]
    lat = sorted(r["latency_ms"] for r in ok)
    print(f"\nrewriter: {model}   ok {len(ok)}/{len(cases)}   "
          f"queries per question {statistics.mean(len(r['queries']) for r in ok):.1f}   "
          f"rewrite latency p50 {statistics.median(lat):.0f} ms, p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")
    print(f"\n{'variant':30}{'recall@1':>9}{'recall@3':>9}{'pass':>8}{'retrieval ms':>14}{'rewrite ms':>12}{'total ms':>10}")
    for r in results:
        o = r["overall"]
        print(f"{r['label']:30}{o['recall_at_1']:>9.2f}{o['recall_at_3']:>9.2f}{o['pass']:>5}/{o['answerable']:<2}"
              f"{r['avg_retrieval_ms']:>14.1f}{r['avg_rewrite_ms']:>12.0f}{o['avg_latency_ms']:>10.0f}")
    print(f"\nrecall@3 by category\n{'variant':30}" + "".join(f"{c[:9]:>10}" for c in CATEGORIES[:-1]))
    for r in results:
        print(f"{r['label']:30}" + "".join(f"{r['by_category'][c]['recall_at_3']:>10.2f}" for c in CATEGORIES[:-1]))
    base = set(results[0]["failed"])
    for r in results[1:]:
        f = set(r["failed"])
        print(f"\n{r['label']}: fixed {sorted(base - f)}  broke {sorted(f - base)}")
    print("\nrewrites for questions that changed or still fail:")
    changed = set().union(*[set(r["failed"]) for r in results])
    for c in cases:
        if c["id"] in changed:
            print(f"  {c['id']:9} {c['question'][:60]!r}\n            -> {rewrites[c['id']]['queries']}")

    (RESULTS_DIR / "rewrite_summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rewriter_model": model, "prompt_key": key,
         "rewrite_ok": len(ok), "variants": results}, indent=2))
    print(f"\nwrote {RESULTS_DIR / 'rewrite_summary.json'} and {CACHE}")


if __name__ == "__main__":
    main()
