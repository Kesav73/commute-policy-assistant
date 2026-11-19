"""Retrieval evaluation: does the pipeline fetch the chunks a human said it should?

    python -m eval.run_retrieval            # score, print scorecard, write results

Inputs   eval/questions.jsonl  - 90 hand-labelled questions (30 per company)
Outputs  eval/results/retrieval.jsonl          - one line per case, required format:
                                                  id, input, expected, actual, pass, notes
         eval/results/retrieval_summary.json   - the numbers in the scorecard

Gold labels per question:
    required  chunks that MUST be in the top 3 for a complete answer
    helpful   relevant but optional (count towards precision, never a failure)

Metrics (answerable questions = required is non-empty):
    pass         every required chunk is in the top 3 AND the evidence gate let it through
    recall@3     share of required chunks found in the top 3
    hit@1        the top chunk is a required chunk
    MRR          1 / rank of the first required chunk (0 if none in the top 3)
    precision@3  share of the top 3 that are required or helpful
    gate miss    the gate said "not covered" for an answerable question (a real failure)
Not-covered questions (required is empty):
    pass         no chunk from another company was retrieved (isolation)
    gate catch   the gate already said "not covered" (0 LLM calls); the rest are left
                 to the LLM, so this is reported but is not a failure at this stage
All questions:
    leaks        chunks from any other company - must always be 0
"""
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.embeddings import embedding_function
from app.ingest import ingest_all
from app.policy_parser import parse_policy_file
from app.retriever import TOP_K, retrieve
from app.store import get_client

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
CATEGORIES = ["direct", "paraphrase", "same_section", "multi_hop", "edge",
              "cross_company", "injection", "not_covered"]


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

def score_case(case: dict, r) -> dict:
    got = [c.section for c in r.chunks]
    required, helpful = case["required"], case["helpful"]
    leaks = [c.id for c in r.chunks if c.company_id != case["company"]]
    answerable = bool(required)

    row = {
        "retrieved": [{"section": c.section, "title": c.section_title, "distance": round(c.distance, 3)}
                      for c in r.chunks],
        "has_evidence": r.has_evidence,
        "best_distance": round(r.best_distance, 3) if r.chunks else None,
        "leaks": leaks,
        "latency_ms": round(sum(r.timings_ms.values()), 1),
    }
    notes = []
    if answerable:
        found = [s for s in required if s in got]
        ranks = [got.index(s) + 1 for s in required if s in got]
        row.update(
            recall_at_3=len(found) / len(required),
            hit_at_1=bool(got) and got[0] in required,
            mrr=1 / min(ranks) if ranks else 0.0,
            precision_at_3=sum(s in required + helpful for s in got) / len(got) if got else 0.0,
            gate_miss=not r.has_evidence,
        )
        passed = len(found) == len(required) and r.has_evidence and not leaks
        if missing := [s for s in required if s not in got]:
            notes.append(f"missed required {missing}")
        if not r.has_evidence:
            notes.append(f"gate wrongly said not covered (best {row['best_distance']})")
        if got and got[0] not in required:
            notes.append(f"top-1 was §{got[0]}")
    else:
        row["gate_caught"] = not r.has_evidence
        passed = not leaks
        notes.append("gate: not covered (0 LLM calls)" if not r.has_evidence
                     else f"gate passed it on (best {row['best_distance']}); LLM must decide")
    if leaks:
        notes.append(f"LEAK {leaks}")
    return {"passed": passed, "row": row, "notes": "; ".join(notes) or "ok"}


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(scored: list[tuple[dict, dict]]) -> dict:
    def block(items):
        ans = [s["row"] for c, s in items if c["required"]]
        nc = [s["row"] for c, s in items if not c["required"]]
        return {
            "n": len(items),
            "pass": sum(s["passed"] for _, s in items),
            "answerable": len(ans),
            "recall_at_3": mean(r["recall_at_3"] for r in ans),
            "hit_at_1": mean(r["hit_at_1"] for r in ans),
            "mrr": mean(r["mrr"] for r in ans),
            "precision_at_3": mean(r["precision_at_3"] for r in ans),
            "gate_misses": sum(r["gate_miss"] for r in ans),
            "not_covered": len(nc),
            "gate_caught": sum(r["gate_caught"] for r in nc),
            "leaks": sum(len(s["row"]["leaks"]) for _, s in items),
        }

    by_cat, by_co, by_diff = defaultdict(list), defaultdict(list), defaultdict(list)
    for c, s in scored:
        by_cat[c["category"]].append((c, s))
        by_co[c["company"]].append((c, s))
        by_diff[c["difficulty"]].append((c, s))
    lat = [s["row"]["latency_ms"] for _, s in scored]
    return {
        "overall": block(scored),
        "by_category": {k: block(by_cat[k]) for k in CATEGORIES if k in by_cat},
        "by_company": {k: block(v) for k, v in sorted(by_co.items())},
        "by_difficulty": {k: block(by_diff[k]) for k in ("easy", "medium", "hard") if k in by_diff},
        "latency_ms": {"p50": round(statistics.median(lat), 1),
                       "p95": round(statistics.quantiles(lat, n=20)[18], 1),
                       "max": round(max(lat), 1)},
    }


def gate_sweep(scored: list[tuple[dict, dict]]) -> list[dict]:
    """How each candidate threshold trades answerable questions lost vs not-covered caught."""
    ans = [s["row"]["best_distance"] for c, s in scored if c["required"]]
    nc = [s["row"]["best_distance"] for c, s in scored if not c["required"]]
    rows = []
    for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        rows.append({"threshold": t,
                     "answerable_wrongly_blocked": sum(d > t for d in ans),
                     "not_covered_caught": sum(d > t for d in nc)})
    return rows


# ---------------------------------------------------------------- report --

def fmt(x) -> str:
    return "  -  " if x != x else f"{x:5.2f}"  # NaN -> "-"


def print_scorecard(summary: dict, sweep: list[dict], n_ans: int, n_nc: int) -> None:
    head = (f"{'':16}{'n':>4}{'pass':>7}{'recall@3':>10}{'hit@1':>8}{'MRR':>7}{'prec@3':>8}"
            f"{'gate-miss':>11}{'gate-catch':>12}{'leaks':>7}")

    def line(name, b):
        catch = f"{b['gate_caught']}/{b['not_covered']}" if b["not_covered"] else "-"
        miss = f"{b['gate_misses']}" if b["answerable"] else "-"
        return (f"{name:16}{b['n']:>4}{b['pass']:>4}/{b['n']:<3}{fmt(b['recall_at_3']):>9}"
                f"{fmt(b['hit_at_1']):>8}{fmt(b['mrr']):>7}{fmt(b['precision_at_3']):>8}"
                f"{miss:>11}{catch:>12}{b['leaks']:>7}")

    print("\n" + "=" * len(head) + "\nRETRIEVAL SCORECARD\n" + "=" * len(head))
    print(head)
    print(line("OVERALL", summary["overall"]))
    for title, key in (("by category", "by_category"), ("by company", "by_company"),
                       ("by difficulty", "by_difficulty")):
        print(f"-- {title}")
        for k, b in summary[key].items():
            print(line(k, b))
    lat = summary["latency_ms"]
    print(f"\nretrieval latency (embed + search): p50 {lat['p50']} ms, p95 {lat['p95']} ms, max {lat['max']} ms")
    print(f"\nevidence-gate sweep ({n_ans} answerable, {n_nc} not covered):")
    print(f"  {'threshold':>9}  {'answerable wrongly blocked':>27}  {'not-covered caught':>19}")
    for r in sweep:
        print(f"  {r['threshold']:>9.2f}  {r['answerable_wrongly_blocked']:>27}  {r['not_covered_caught']:>19}")


def main() -> None:
    s = get_settings()
    cases = load_questions()
    validate_gold(cases, s.policies_dir)

    # Fresh, throwaway store: the score never depends on leftover local data.
    with tempfile.TemporaryDirectory() as tmp:
        client = get_client(Path(tmp))
        ingest_all(client, s.policies_dir)
        embedding_function()(["warm up"])

        scored, lines = [], []
        for c in cases:
            r = retrieve(client, c["company"], c["question"])
            res = score_case(c, r)
            scored.append((c, res))
            lines.append({
                "id": c["id"],
                "input": {"company": c["company"], "question": c["question"]},
                "expected": {"status": c["expected_status"], "required": c["required"],
                             "helpful": c["helpful"], "category": c["category"],
                             "difficulty": c["difficulty"]},
                "actual": res["row"],
                "pass": res["passed"],
                "notes": res["notes"],
            })

    summary = summarise(scored)
    sweep = gate_sweep(scored)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "retrieval.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    (RESULTS_DIR / "retrieval_summary.json").write_text(json.dumps(
        {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "gate_threshold": s.retrieval_max_distance, "top_k": TOP_K,
         **summary, "gate_sweep": sweep}, indent=2))

    print_scorecard(summary, sweep, summary["overall"]["answerable"], summary["overall"]["not_covered"])
    failures = [l for l in lines if not l["pass"]]
    print(f"\nfailed cases ({len(failures)}):")
    for l in failures:
        top = " ".join(f"§{x['section']}({x['distance']})" for x in l["actual"]["retrieved"])
        print(f"  {l['id']:9} [{l['expected']['category']}] {l['input']['question'][:60]!r}")
        print(f"            want {l['expected']['required']}  got {top}  -> {l['notes']}")
    print(f"\nwrote {RESULTS_DIR / 'retrieval.jsonl'} and retrieval_summary.json")


if __name__ == "__main__":
    main()
