"""End-to-end evaluation: every gold question through the real HTTP POST /ask.

    python -m eval.run_e2e                 # configured LLM (Groq when GROQ_API_KEY is set)
    python -m eval.run_e2e --offline       # FakeLLM: no key, no quota (plumbing check only)
    python -m eval.run_e2e --pace 20       # seconds between LLM-backed questions (default 20: free-tier limit)
    python -m eval.run_e2e --only acme-01,bolt-19

Path per question (exactly what an employee hits):
    JWT for an employee of that company -> POST /ask -> auth -> validation
    -> hybrid retrieval (3 chunks) -> evidence gate -> LLM (<= 2 calls)
    -> citation checks -> JSON response

Checks per case (deterministic string and section checks, no LLM judge):
    status     actual status == expected ("answered" / "not_covered"); "unavailable" fails
    facts      every must_include group has at least one alternative in the answer
    citation   an answered case cites at least one REQUIRED section
    forbidden  no must_not_include string appears (other companies' values, injected text)
    leak       every citation's doc_id belongs to the asker's company
    pass       all checks that apply

Scorecard (the same every run):
    1. Answer quality   pass, status, facts, citation          (count / applicable cases)
    2. Safety           leaks, forbidden text, confidently wrong (not_covered expected,
                        but answered) - all must be 0
    3. Budgets          latency p50/p95/max (target p95 < 3 s), max LLM calls (limit 2),
                        max citations (limit 3)
    4. Failure causes   each failed case gets ONE cause, in this order:
                        unsafe       leak or forbidden text
                        unavailable  LLM/retrieval failure -> "try again"
                        retrieval    a required section was not among the 3 chunks
                        over_cautious answerable, retrieval complete, model said not covered
                        confident_wrong  not covered expected, but the model answered
                        incomplete   answered, but a required fact is missing
                        citation     answered with the facts, but cited no required section
    Also: answer quality on cases where retrieval was complete, so generation is
    judged separately from retrieval.

Outputs  eval/results/e2e.jsonl          one line per case: id, input, expected, actual, pass, notes
         eval/results/e2e_summary.json   the scorecard numbers
"""
import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.config import FALLBACK_MODEL, PRIMARY_MODEL, get_settings
from app.embeddings import embed_query
from app.llm import FakeLLM, LLMError, build_llm
from app.store import PolicyStore
from app.versioning import ingest_all
from eval.run_retrieval import CATEGORIES, QUESTIONS, RESULTS_DIR, load_questions, validate_gold

LLM_CACHE = RESULTS_DIR / "llm_cache.jsonl"


class CachingLLM:
    """Wraps the real LLM for the eval: a response is stored under a hash of
    (model, system prompt, user prompt). Re-running after a change that doesn't
    alter a prompt (e.g. a code-only guardrail) re-uses the stored response and
    costs no tokens. Errors are never cached. `hits` counts cached calls, so
    latency stats can exclude them (a cached call takes ~0 ms)."""

    def __init__(self, llm, path: Path = LLM_CACHE, cache_only: bool = False):
        self.llm, self.path, self.hits, self.provider = llm, path, 0, getattr(llm, "provider", "")
        self.cache_only = cache_only     # never call the LLM: a cache miss counts as an outage
        self.store = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.store[row["key"]] = row["raw"]

    @staticmethod
    def key(system: str, user: str, model: str) -> str:
        return hashlib.sha256(f"{model}\n{system}\n{user}".encode()).hexdigest()

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        k = self.key(system, user, model)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        if self.cache_only:
            raise LLMError("unavailable", f"{model}: not in the response cache (--cache-only)")
        raw = self.llm.complete_json(system, user, model, timeout_s)
        self.store[k] = raw
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": k, "model": model, "raw": raw}) + "\n")
        return raw


EMPLOYEE_FOR = {"acme": "E-ACME-001", "bolt": "E-BOLT-001", "crest": "E-CRST-001"}
DOC_FOR = {"acme": "acme-transport-policy", "bolt": "bolt-commute-policy",
           "crest": "crest-transport-guidelines"}
CAUSES = ["unsafe", "unavailable", "retrieval", "over_cautious", "confident_wrong",
          "incomplete", "citation"]


# ---------------------------------------------------------------- scoring --

def score(case: dict, http_status: int, body: dict, res) -> dict:
    expected = case["expected_status"]
    status = body.get("status") or ("unavailable" if http_status == 503 else f"http_{http_status}")
    answer = body.get("answer") or body.get("message") or ""
    low = answer.lower()
    cited = [c["section"] for c in body.get("citations", [])]
    cited_docs = {c["doc_id"] for c in body.get("citations", [])}
    retrieved = [x["section"] for x in res.retrieved] if res else []
    retrieval_complete = all(s in retrieved for s in case["required"])

    checks, notes = {"status": status == expected}, []
    if not checks["status"]:
        notes.append(f"status {status} (expected {expected})")
    leaked = sorted(cited_docs - {DOC_FOR[case["company"]]})
    checks["no_leak"] = not leaked
    if leaked:
        notes.append(f"LEAK cited {leaked}")
    forbidden = [s for s in case["must_not_include"] if s.lower() in low]
    checks["no_forbidden"] = not forbidden
    if forbidden:
        notes.append(f"forbidden text {forbidden}")
    if expected == "answered" and status == "answered":
        missing = [g for g in case["must_include"] if not any(a.lower() in low for a in g)]
        checks["facts"] = not missing
        if missing:
            notes.append(f"missing fact {[g[0] for g in missing]}")
        checks["citation"] = any(s in case["required"] for s in cited)
        if not checks["citation"]:
            notes.append(f"cited {cited}, none of required {case['required']}")
    if case["required"] and not retrieval_complete:
        notes.append(f"retrieval missed {[s for s in case['required'] if s not in retrieved]}")
    if res is not None:
        if FALLBACK_MODEL in res.models:
            notes.append(f"fallback model used ({res.reason})")
        elif res.reason and res.reason != "ok":
            notes.append(res.reason)
        if res.reason.startswith("gate"):
            notes.append("stopped by gate (0 LLM calls)")

    passed = all(checks.values())
    cause = None
    if not passed:
        if leaked or forbidden:
            cause = "unsafe"
        elif status == "unavailable":
            cause = "unavailable"
        elif expected == "answered" and not retrieval_complete:
            cause = "retrieval"
        elif expected == "answered" and status == "not_covered":
            cause = "over_cautious"
        elif expected == "not_covered" and status == "answered":
            cause = "confident_wrong"
        elif not checks.get("facts", True):
            cause = "incomplete"
        else:
            cause = "citation"
    return {"pass": passed, "checks": checks, "notes": notes, "cause": cause,
            "status": status, "answer": answer, "cited": cited, "retrieved": retrieved,
            "retrieval_complete": retrieval_complete}


# ---------------------------------------------------------------- summary --

def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def quality(rows: list[dict]) -> dict:
    exp_ans = [r for r in rows if r["expected"]["status"] == "answered"]
    both = [r for r in exp_ans if r["actual"]["status"] == "answered"]
    return {
        "n": len(rows),
        "pass": sum(r["pass"] for r in rows),
        "status": sum(r["_checks"]["status"] for r in rows),
        "answerable": len(exp_ans),
        "facts": sum(r["_checks"].get("facts", False) for r in both),
        "citation": sum(r["_checks"].get("citation", False) for r in both),
    }


def summarise(rows: list[dict]) -> dict:
    groups = {"category": defaultdict(list), "company": defaultdict(list)}
    for r in rows:
        groups["category"][r["expected"]["category"]].append(r)
        groups["company"][r["input"]["company"]].append(r)
    # Cached LLM responses take ~0 ms, so they'd flatter latency: measure only live calls.
    live = [r for r in rows if not r["actual"].get("llm_cached")]
    lat = [r["actual"]["latency_ms"] for r in live]
    lat_llm = [r["actual"]["latency_ms"] for r in live if r["actual"]["llm_calls"] > 0]

    def spread(values):
        if not values:
            return "not measured (every LLM response came from the cache)"
        return {"p50": pct(values, .5), "p95": pct(values, .95), "max": round(max(values), 1)}
    calls = Counter(r["actual"]["llm_calls"] for r in rows)
    conf = Counter(f'{r["expected"]["status"]}->{r["actual"]["status"]}' for r in rows)
    complete = [r for r in rows if r["expected"]["status"] == "answered" and r["_retrieval_complete"]]
    return {
        "quality": quality(rows),
        "quality_when_retrieval_complete": quality(complete),
        "safety": {
            "leaks": sum(not r["_checks"]["no_leak"] for r in rows),
            "forbidden_text": sum(not r["_checks"]["no_forbidden"] for r in rows),
            "confident_wrong": conf.get("not_covered->answered", 0),
        },
        "budgets": {
            "latency_ms_all": spread(lat),
            "latency_ms_with_llm": spread(lat_llm),
            "llm_calls": {str(k): v for k, v in sorted(calls.items())},
            "max_llm_calls": max(calls),
            "max_citations": max(len(r["actual"]["citations"]) for r in rows),
            "latency_measured_on": f"{len(live)} live questions ({len(rows) - len(live)} from LLM cache)",
        },
        "failure_causes": {c: sum(r["_cause"] == c for r in rows) for c in CAUSES},
        "status_confusion": dict(sorted(conf.items())),
        "by_category": {k: quality(groups["category"][k]) for k in CATEGORIES if k in groups["category"]},
        "by_company": {k: quality(v) for k, v in sorted(groups["company"].items())},
        "models_used": dict(Counter(m for r in rows for m in r["actual"]["models"])),
    }


def print_scorecard(s: dict, provider: str, models: list[str]) -> None:
    q, qc = s["quality"], s["quality_when_retrieval_complete"]
    b, sf = s["budgets"], s["safety"]
    print("\n" + "=" * 72 + f"\nEND-TO-END SCORECARD   llm={provider} {models}\n" + "=" * 72)
    print("1. Answer quality")
    print(f"   pass {q['pass']}/{q['n']}   status {q['status']}/{q['n']}   "
          f"facts {q['facts']}/{q['answerable']}   citation {q['citation']}/{q['answerable']}")
    print(f"   when retrieval was complete: pass {qc['pass']}/{qc['n']}   facts {qc['facts']}/{qc['n']}   "
          f"citation {qc['citation']}/{qc['n']}")
    print(f"2. Safety (all must be 0)\n   leaks {sf['leaks']}   forbidden text {sf['forbidden_text']}   "
          f"confidently wrong {sf['confident_wrong']}")
    print("3. Budgets")
    print(f"   latency, all questions : {b['latency_ms_all']}")
    print(f"   latency, LLM-backed    : {b['latency_ms_with_llm']}   (target: p95 < 3000 ms)")
    print(f"   LLM calls per question : {b['llm_calls']}   max {b['max_llm_calls']} (limit 2)")
    print(f"   max citations          : {b['max_citations']} (limit 3)")
    print(f"   latency measured on    : {b['latency_measured_on']}")
    print(f"4. Failure causes\n   {s['failure_causes']}")
    print(f"   status confusion (expected->actual): {s['status_confusion']}")
    print(f"\n{'by category':16}{'pass':>9}{'status':>9}{'facts':>9}{'citation':>10}")
    for name, key in (("by category", "by_category"), ("by company", "by_company")):
        if key == "by_company":
            print(f"-- {name}")
        for k, x in s[key].items():
            ae = x["answerable"]
            print(f"{k:16}{x['pass']:>5}/{x['n']:<3}{x['status']:>5}/{x['n']:<3}"
                  f"{(str(x['facts']) + '/' + str(ae)) if ae else '-':>9}"
                  f"{(str(x['citation']) + '/' + str(ae)) if ae else '-':>10}")


# ------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use FakeLLM (no key, no quota)")
    ap.add_argument("--pace", type=float, default=20.0,
                    help="seconds to wait after each LLM-backed question: Groq's free tier allows 8K "
                         "tokens/min and a question with rewriting uses ~2.5K (2 calls)")
    ap.add_argument("--only", help="comma-separated case ids to run")
    ap.add_argument("--cache-only", action="store_true",
                    help="never call the LLM: replay cached responses only (a miss counts as 'unavailable')")
    ap.add_argument("--no-cache", action="store_true",
                    help="always call the LLM (default: reuse responses stored in eval/results/llm_cache.jsonl)")
    ap.add_argument("--max-unavailable-streak", type=int, default=5,
                    help="stop the run after this many 'unavailable' answers in a row "
                         "(network/provider outage: the rest of the run would be meaningless)")
    args = ap.parse_args()

    s = get_settings()
    cases = load_questions(QUESTIONS)
    validate_gold(cases, s.policies_dir)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    llm = FakeLLM() if args.offline else build_llm(s)
    cache = None
    if not args.offline and not args.no_cache:
        llm = cache = CachingLLM(llm, cache_only=args.cache_only)
    provider = "fake" if args.offline else s.llm_provider
    models = ["fake"] if provider == "fake" else [PRIMARY_MODEL, FALLBACK_MODEL]
    pace = 0 if provider == "fake" else args.pace

    # Record the pipeline's internal result (calls, models, retrieved, reason) for each
    # request while still going through the real HTTP endpoint.
    captured, real_ask = {}, main_module.ask

    def recording_ask(*a, **kw):
        captured["last"] = real_ask(*a, **kw)
        return captured["last"]

    main_module.ask = recording_ask
    rows, started = [], time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = PolicyStore(Path(tmp))
            ingest_all(store, s.policies_dir)
            embed_query("warm up")
            main_module.app.state.store, main_module.app.state.llm = store, llm
            client = TestClient(main_module.app, raise_server_exceptions=False)
            def token(company):   # fresh token per question: a long run can't outlive the TTL
                return client.post("/auth/token", json={"employee_id": EMPLOYEE_FOR[company]}).json()["access_token"]

            streak = 0
            for i, case in enumerate(cases, 1):
                captured.pop("last", None)
                hits_before = cache.hits if cache else 0
                t0 = time.perf_counter()
                r = client.post("/ask", json={"question": case["question"]},
                                headers={"Authorization": f"Bearer {token(case['company'])}"})
                latency = (time.perf_counter() - t0) * 1000
                res = captured.get("last")
                sc = score(case, r.status_code, r.json(), res)
                rows.append({
                    "id": case["id"],
                    "input": {"company": case["company"], "question": case["question"]},
                    "expected": {"status": case["expected_status"], "required": case["required"],
                                 "must_include": case["must_include"],
                                 "must_not_include": case["must_not_include"],
                                 "category": case["category"]},
                    "actual": {"http_status": r.status_code, "status": sc["status"],
                               "answer": sc["answer"], "citations": sc["cited"],
                               "retrieved": sc["retrieved"],
                               "llm_calls": res.llm_calls if res else 0,
                               "models": res.models if res else [],
                               "latency_ms": round(latency, 1),
                               "llm_cached": bool(cache and cache.hits > hits_before)},
                    "pass": sc["pass"],
                    "notes": "; ".join(([f"cause: {sc['cause']}"] if sc["cause"] else []) + sc["notes"]) or "ok",
                    "_checks": sc["checks"], "_cause": sc["cause"],
                    "_retrieval_complete": sc["retrieval_complete"],
                })
                print(f"[{i:>2}/{len(cases)}] {'PASS' if sc['pass'] else 'FAIL'} {case['id']:9} "
                      f"{latency:6.0f} ms  {sc['status']:12} {rows[-1]['notes'][:80]}", flush=True)
                streak = streak + 1 if sc["status"] == "unavailable" else 0
                if streak >= args.max_unavailable_streak:
                    raise SystemExit(f"\nABORTED: {streak} 'unavailable' answers in a row (network or "
                                     "provider outage). Results not written; rerun when it's back.")
                if res and res.llm_calls and i < len(cases) and not rows[-1]["actual"]["llm_cached"]:
                    time.sleep(pace)
    finally:
        main_module.ask = real_ask

    summary = summarise(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_offline" if args.offline else ""
    with open(RESULTS_DIR / f"e2e{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    (RESULTS_DIR / f"e2e{suffix}_summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "llm_provider": provider,
         "models": models,
         "run_seconds": round(time.time() - started), **summary}, indent=2))

    print_scorecard(summary, provider, models)
    failures = [r for r in rows if not r["pass"]]
    print(f"\nfailed cases ({len(failures)}):")
    for r in failures:
        print(f"  {r['id']:9} [{r['expected']['category']}] {r['input']['question'][:70]!r}")
        print(f"            -> {r['actual']['status']}: {(r['actual']['answer'] or '')[:160]!r}")
        print(f"               {r['notes']}")
    print(f"\nwrote {RESULTS_DIR / f'e2e{suffix}.jsonl'}")


if __name__ == "__main__":
    main()
