"""End-to-end evaluation: every gold question through the real HTTP POST /ask.

    python -m eval.run_e2e                 # configured LLM (Groq if GROQ_API_KEY is set)
    python -m eval.run_e2e --offline       # FakeLLM: no key, no quota (plumbing check only)
    python -m eval.run_e2e --pace 8        # seconds between LLM-backed questions (rate limits)

Path per question:  JWT for an employee of that company -> POST /ask -> auth -> validation
-> retrieval -> gate -> LLM (<= 2 calls) -> output checks -> JSON response.

Scoring per case (deterministic string/section checks, no LLM judge):
    status      actual status == expected ("answered" / "not_covered"); "unavailable" fails
    facts       every must_include group has at least one alternative in the answer
    forbidden   no must_not_include string appears in the answer (catches leaks / injection)
    citation    answered cases cite at least one REQUIRED section
    leak        every citation's doc_id belongs to the asker's company
    pass        all of the above that apply

Outputs  eval/results/e2e.jsonl (id, input, expected, actual, pass, notes)
         eval/results/e2e_summary.json
"""
import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.config import get_settings
from app.embeddings import embedding_function
from app.ingest import ingest_all
from app.llm import FakeLLM, build_llm
from app.store import get_client
from eval.run_retrieval import CATEGORIES, QUESTIONS, RESULTS_DIR, load_questions, validate_gold

EMPLOYEE_FOR = {"acme": "E-ACME-001", "bolt": "E-BOLT-001", "crest": "E-CRST-001"}
DOC_FOR = {"acme": "acme-transport-policy", "bolt": "bolt-commute-policy",
           "crest": "crest-transport-guidelines"}


# ---------------------------------------------------------------- scoring --

def score(case: dict, http_status: int, body: dict, res) -> tuple[bool, dict, list[str]]:
    expected = case["expected_status"]
    status = body.get("status") or ("unavailable" if http_status == 503 else f"http_{http_status}")
    answer = body.get("answer") or body.get("message") or ""
    cited = [c["section"] for c in body.get("citations", [])]
    cited_docs = {c["doc_id"] for c in body.get("citations", [])}
    low = answer.lower()

    checks, notes = {}, []
    checks["status"] = status == expected
    if not checks["status"]:
        notes.append(f"status {status} (expected {expected})")

    leaked_docs = sorted(cited_docs - {DOC_FOR[case["company"]]})
    checks["no_leak"] = not leaked_docs
    if leaked_docs:
        notes.append(f"LEAK cited {leaked_docs}")

    forbidden = [s for s in case["must_not_include"] if s.lower() in low]
    checks["no_forbidden"] = not forbidden
    if forbidden:
        notes.append(f"forbidden text {forbidden}")

    if expected == "answered" and status == "answered":
        missing = [g for g in case["must_include"] if not any(alt.lower() in low for alt in g)]
        checks["facts"] = not missing
        if missing:
            notes.append(f"missing fact {[g[0] for g in missing]}")
        checks["citation"] = any(s in case["required"] for s in cited)
        if not checks["citation"]:
            notes.append(f"cited {cited}, none required {case['required']}")

    if res is not None:
        recall = (sum(s["section"] in case["required"] for s in res.retrieved) / len(case["required"])
                  if case["required"] else None)
        if recall is not None and recall < 1:
            got = [s["section"] for s in res.retrieved]
            notes.append(f"retrieval missed {[s for s in case['required'] if s not in got]}")
        if res.llm_calls == 2:
            notes.append(f"fallback used ({res.reason})")
        if res.reason.startswith("gate"):
            notes.append("stopped by gate")
        elif res.status == "not_covered" and res.reason and res.reason != "model said not covered":
            notes.append(res.reason)
    passed = all(checks.values())
    return passed, checks, notes


# ---------------------------------------------------------------- summary --

def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def summarise(rows: list[dict]) -> dict:
    def block(rs):
        ans_exp = [r for r in rs if r["expected"]["status"] == "answered"]
        both_ans = [r for r in ans_exp if r["actual"]["status"] == "answered"]
        return {
            "n": len(rs),
            "pass": sum(r["pass"] for r in rs),
            "status_correct": sum(r["_checks"]["status"] for r in rs),
            "facts_correct": sum(r["_checks"].get("facts", False) for r in both_ans),
            "answered_expected": len(ans_exp),
            "citation_correct": sum(r["_checks"].get("citation", False) for r in both_ans),
            "forbidden_hits": sum(not r["_checks"]["no_forbidden"] for r in rs),
            "leaks": sum(not r["_checks"]["no_leak"] for r in rs),
        }

    groups = {"by_category": defaultdict(list), "by_company": defaultdict(list),
              "by_difficulty": defaultdict(list)}
    for r in rows:
        groups["by_category"][r["expected"]["category"]].append(r)
        groups["by_company"][r["input"]["company"]].append(r)
        groups["by_difficulty"][r["expected"]["difficulty"]].append(r)

    conf = Counter((r["expected"]["status"], r["actual"]["status"]) for r in rows)
    lat = [r["actual"]["latency_ms"] for r in rows]
    lat_llm = [r["actual"]["latency_ms"] for r in rows if r["actual"]["llm_calls"] > 0]
    calls = Counter(r["actual"]["llm_calls"] for r in rows)
    return {
        "overall": block(rows),
        "by_category": {k: block(groups["by_category"][k]) for k in CATEGORIES
                        if k in groups["by_category"]},
        "by_company": {k: block(v) for k, v in sorted(groups["by_company"].items())},
        "by_difficulty": {k: block(groups["by_difficulty"][k]) for k in ("easy", "medium", "hard")
                          if k in groups["by_difficulty"]},
        "confusion": {f"{e}->{a}": n for (e, a), n in sorted(conf.items())},
        "llm_calls": {str(k): v for k, v in sorted(calls.items())},
        "max_llm_calls": max(calls),
        "max_citations": max(len(r["actual"]["citations"]) for r in rows),
        "latency_ms_all": {"p50": pct(lat, .5), "p95": pct(lat, .95), "max": round(max(lat), 1)},
        "latency_ms_with_llm": ({"p50": pct(lat_llm, .5), "p95": pct(lat_llm, .95),
                                 "max": round(max(lat_llm), 1)} if lat_llm else {}),
        "models_used": dict(Counter(m for r in rows for m in r["actual"]["models"])),
    }


def print_scorecard(s: dict, provider: str, models: list[str]) -> None:
    head = f"{'':16}{'n':>4}{'pass':>9}{'status':>9}{'facts':>9}{'citation':>10}{'forbid':>8}{'leaks':>7}"

    def line(name, b):
        ae = b["answered_expected"]
        return (f"{name:16}{b['n']:>4}{b['pass']:>5}/{b['n']:<3}{b['status_correct']:>5}/{b['n']:<3}"
                f"{(str(b['facts_correct']) + '/' + str(ae)) if ae else '-':>9}"
                f"{(str(b['citation_correct']) + '/' + str(ae)) if ae else '-':>10}"
                f"{b['forbidden_hits']:>8}{b['leaks']:>7}")

    print("\n" + "=" * len(head))
    print(f"END-TO-END SCORECARD   llm={provider} {models}")
    print("=" * len(head))
    print(head)
    print(line("OVERALL", s["overall"]))
    for title, key in (("by category", "by_category"), ("by company", "by_company"),
                       ("by difficulty", "by_difficulty")):
        print(f"-- {title}")
        for k, b in s[key].items():
            print(line(k, b))
    print(f"\nstatus confusion (expected->actual): {s['confusion']}")
    print(f"LLM calls per question: {s['llm_calls']}   max {s['max_llm_calls']} (limit 2)   "
          f"max citations {s['max_citations']} (limit 3)")
    print(f"models used: {s['models_used']}")
    print(f"latency, all questions : {s['latency_ms_all']}")
    print(f"latency, LLM-backed    : {s['latency_ms_with_llm']}   (target p95 < 3000 ms)")


# ------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use FakeLLM (no key, no quota)")
    ap.add_argument("--pace", type=float, default=7.5,
                    help="seconds to wait after each LLM-backed question (Groq free tier: 8K tokens/min)")
    ap.add_argument("--only", help="comma-separated case ids to run")
    args = ap.parse_args()

    s = get_settings()
    cases = load_questions(QUESTIONS)
    validate_gold(cases, s.policies_dir)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    llm = FakeLLM() if args.offline else build_llm(s)
    provider = "fake" if args.offline else s.llm_provider
    pace = 0 if provider == "fake" else args.pace

    # Record the pipeline's internal result for each request (calls, models, reason)
    # while still going through the real HTTP endpoint.
    captured = {}
    real_ask = main_module.ask

    def recording_ask(*a, **kw):
        res = real_ask(*a, **kw)
        captured["last"] = res
        return res

    main_module.ask = recording_ask
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        chroma = get_client(Path(tmp))
        ingest_all(chroma, s.policies_dir)
        embedding_function()(["warm up"])
        main_module.app.state.chroma, main_module.app.state.llm = chroma, llm
        client = TestClient(main_module.app, raise_server_exceptions=False)
        tokens = {co: client.post("/auth/token", json={"employee_id": e}).json()["access_token"]
                  for co, e in EMPLOYEE_FOR.items()}

        started = time.time()
        for i, case in enumerate(cases, 1):
            captured.pop("last", None)
            t0 = time.perf_counter()
            r = client.post("/ask", json={"question": case["question"]},
                            headers={"Authorization": f"Bearer {tokens[case['company']]}"})
            latency = (time.perf_counter() - t0) * 1000
            body = r.json()
            res = captured.get("last")
            passed, checks, notes = score(case, r.status_code, body, res)
            rows.append({
                "id": case["id"],
                "input": {"company": case["company"], "question": case["question"]},
                "expected": {"status": case["expected_status"], "required": case["required"],
                             "helpful": case["helpful"], "must_include": case["must_include"],
                             "must_not_include": case["must_not_include"],
                             "category": case["category"], "difficulty": case["difficulty"]},
                "actual": {
                    "http_status": r.status_code,
                    "status": body.get("status") or ("unavailable" if r.status_code == 503 else "error"),
                    "answer": body.get("answer") or body.get("message"),
                    "citations": [c["section"] for c in body.get("citations", [])],
                    "llm_calls": res.llm_calls if res else 0,
                    "models": res.models if res else [],
                    "retrieved": [x["section"] for x in res.retrieved] if res else [],
                    "latency_ms": round(latency, 1),
                },
                "pass": passed,
                "notes": "; ".join(notes) or "ok",
                "_checks": checks,
            })
            mark = "PASS" if passed else "FAIL"
            print(f"[{i:>2}/{len(cases)}] {mark} {case['id']:9} {r.status_code} {latency:6.0f} ms  "
                  f"{rows[-1]['actual']['status']:12} {rows[-1]['notes'][:70]}", flush=True)
            if res and res.llm_calls and i < len(cases):
                time.sleep(pace)
    main_module.ask = real_ask

    summary = summarise(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "_offline" if args.offline else ""
    with open(RESULTS_DIR / f"e2e{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_checks"}) + "\n")
    (RESULTS_DIR / f"e2e{suffix}_summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "llm_provider": provider,
         "models": [s.llm_model, s.llm_fallback_model] if provider != "fake" else ["fake"],
         "run_seconds": round(time.time() - started), **summary}, indent=2))

    print_scorecard(summary, provider, [s.llm_model, s.llm_fallback_model])
    failures = [r for r in rows if not r["pass"]]
    print(f"\nfailed cases ({len(failures)}):")
    for r in failures:
        print(f"  {r['id']:9} [{r['expected']['category']}] {r['input']['question'][:70]!r}")
        print(f"            -> {r['actual']['status']}: {(r['actual']['answer'] or '')[:150]!r}")
        print(f"               {r['notes']}")
    print(f"\nwrote {RESULTS_DIR / f'e2e{suffix}.jsonl'}")


if __name__ == "__main__":
    main()
