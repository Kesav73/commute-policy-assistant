"""Retrieval evaluation: does the retriever fetch the sections a human said it should?

    python -m eval.run_retrieval

Inputs   eval/questions.jsonl  - 90 hand-labelled questions (30 per company)
Outputs  eval/results/retrieval.jsonl          - one line per case, required format:
                                                  id, input, expected, actual, pass, notes
         eval/results/retrieval_summary.json   - the scorecard numbers

Gold label used here:
    required  the sections a complete answer needs (empty for "not covered" questions)

Metrics, over answerable questions only (required is non-empty):
    recall@k  share of the required sections that appear in the top k results,
              averaged over questions. For a question that needs 2 sections,
              recall@1 can be at most 0.5, by definition.
    pass      recall@3 == 1 (every required section is in the top 3)
Latency, over all questions:
    avg latency  mean wall time of retrieve(): embed the question + Chroma search,
                 in ms, measured after one warm-up query (model load not counted)

Not-covered questions have nothing to recall, so they are listed but not scored.
Safety invariant (not a metric): a chunk from another company must never be
returned. If one is, the case fails and the run says so loudly.
"""
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.policy_parser import parse_policy_file
from app.retriever import TOP_K, retrieve
from app.store import PolicyStore
from app.versioning import ingest_all

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
CATEGORIES = ["direct", "paraphrase", "same_section", "multi_hop", "edge",
              "cross_company", "injection", "not_covered"]
KS = (1, 3)


# --------------------------------------------------------------- gold set --

def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_gold(cases: list[dict], policies_dir: Path) -> None:
    """Check the labels themselves, so a typo in the gold set can't skew the score."""
    sections = {}
    for f in policies_dir.glob("*.md"):
        p = parse_policy_file(f)
        sections[p.company_id] = {c.section for c in p.chunks}

    problems, seen = [], set()
    for c in cases:
        cid = c["id"]
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        if c["company"] not in sections:
            problems.append(f"{cid}: unknown company {c['company']}")
            continue
        labelled = c["required"] + c["helpful"]
        for s in labelled:
            if s not in sections[c["company"]]:
                problems.append(f"{cid}: section {s} does not exist in {c['company']}'s policy")
        if len(labelled) > TOP_K or len(set(labelled)) != len(labelled):
            problems.append(f"{cid}: required+helpful must be <= {TOP_K} distinct sections")
        if c["expected_status"] == "answered" and not c["required"]:
            problems.append(f"{cid}: answered case needs at least one required section")
        if c["expected_status"] == "not_covered" and c["required"]:
            problems.append(f"{cid}: not_covered case must have no required sections")
        if c["category"] not in CATEGORIES:
            problems.append(f"{cid}: unknown category {c['category']}")
    if problems:
        raise SystemExit("gold set is invalid:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------- scoring --

def recall_at(k: int, required: list[str], ranked: list[str]) -> float:
    return sum(s in ranked[:k] for s in required) / len(required)


def score_case(case: dict, ranked: list[str], leaks: list[str], latency_ms: float) -> dict:
    """ranked: section ids in retrieval order. Returns actual, pass, notes."""
    required = case["required"]
    actual = {"retrieved": ranked, "leaks": leaks, "latency_ms": round(latency_ms, 2)}
    notes = []
    if required:
        for k in KS:
            actual[f"recall_at_{k}"] = recall_at(k, required, ranked)
        passed = actual["recall_at_3"] == 1.0
        if missing := [s for s in required if s not in ranked[:3]]:
            notes.append(f"missed {missing}")
        if ranked and ranked[0] not in required:
            notes.append(f"top-1 was {ranked[0]}")
    else:
        passed = True
        notes.append("not covered: nothing to recall")
    if leaks:
        passed = False
        notes.append(f"LEAK {leaks}")
    return {"actual": actual, "pass": passed, "notes": "; ".join(notes) or "ok"}


def mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else None


def block(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expected"]["required"]]
    out = {"n": len(rows), "answerable": len(ans),
           "pass": sum(r["pass"] for r in ans)}
    for k in KS:
        out[f"recall_at_{k}"] = mean(r["actual"][f"recall_at_{k}"] for r in ans)
    out["avg_latency_ms"] = mean(r["actual"]["latency_ms"] for r in rows)
    out["leaks"] = sum(len(r["actual"]["leaks"]) for r in rows)
    return out


def summarise(rows: list[dict]) -> dict:
    groups = {"category": defaultdict(list), "company": defaultdict(list)}
    for r in rows:
        groups["category"][r["expected"]["category"]].append(r)
        groups["company"][r["input"]["company"]].append(r)
    return {
        "overall": block(rows),
        "by_category": {k: block(groups["category"][k]) for k in CATEGORIES if k in groups["category"]},
        "by_company": {k: block(v) for k, v in sorted(groups["company"].items())},
    }


# ---------------------------------------------------------------- report --

def print_scorecard(summary: dict) -> None:
    head = (f"{'':16}{'answerable':>11}{'pass':>9}{'recall@1':>10}{'recall@3':>10}"
            f"{'avg ms':>9}{'leaks':>7}")

    def fmt(x):
        return "    -" if x is None else f"{x:5.2f}"

    def line(name, b):
        p = f"{b['pass']}/{b['answerable']}" if b["answerable"] else "-"
        return (f"{name:16}{b['answerable']:>11}{p:>9}{fmt(b['recall_at_1']):>10}"
                f"{fmt(b['recall_at_3']):>10}{b['avg_latency_ms']:>9.1f}{b['leaks']:>7}")

    print("\n" + "=" * len(head) + "\nRETRIEVAL SCORECARD\n" + "=" * len(head))
    print(head)
    print(line("OVERALL", summary["overall"]))
    for title, key in (("by category", "by_category"), ("by company", "by_company")):
        print(f"-- {title}")
        for k, b in summary[key].items():
            print(line(k, b))


def main() -> None:
    s = get_settings()
    cases = load_questions()
    validate_gold(cases, s.policies_dir)

    # Fresh, throwaway store: the score never depends on leftover local data.
    with tempfile.TemporaryDirectory() as tmp:
        store = PolicyStore(Path(tmp))
        ingest_all(store, s.policies_dir)
        retrieve(store, cases[0]["company"], ["warm up"])   # load the model before timing
        rows = []
        for c in cases:
            t0 = time.perf_counter()
            r = retrieve(store, c["company"], [c["question"]])
            latency_ms = (time.perf_counter() - t0) * 1000
            ranked = [ch.section for ch in r.chunks]
            distances = [round(ch.distance, 3) for ch in r.chunks]
            leaks = [ch.id for ch in r.chunks if ch.company_id != c["company"]]
            res = score_case(c, ranked, leaks, latency_ms)
            res["actual"]["distances"] = distances   # for tuning the "not covered" gate
            rows.append({
                "id": c["id"],
                "input": {"company": c["company"], "question": c["question"]},
                "expected": {"required": c["required"], "category": c["category"]},
                **res,
            })

    summary = summarise(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "retrieval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (RESULTS_DIR / "retrieval_summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "top_k": TOP_K, **summary}, indent=2))

    print_scorecard(summary)
    failures = [r for r in rows if not r["pass"]]
    print(f"\nfailed cases ({len(failures)}):")
    for r in failures:
        print(f"  {r['id']:9} [{r['expected']['category']}] {r['input']['question'][:62]!r}")
        print(f"            want {r['expected']['required']}  got {r['actual']['retrieved']}  -> {r['notes']}")
    if summary["overall"]["leaks"]:
        print("\n!! CROSS-COMPANY LEAK: chunks from another company were returned !!")
    print(f"\nwrote {RESULTS_DIR / 'retrieval.jsonl'} and retrieval_summary.json")


if __name__ == "__main__":
    main()
