"""The ask flow: question -> (rewrite) -> retrieve -> gate -> answer -> checked answer.

    1. pin the caller's company's ACTIVE policy version for the whole request
    With QUERY_REWRITE=on (default, EVALUATION.md):
    2. call 1: rewrite the question into 1-3 search queries (policy outline as context)
       and judge it (input guard). A pure attack / off-topic question stops here.
       If the rewrite fails, the original question is searched instead.
    3. ONE hybrid retrieval for those queries, merged round-robin, top 3
    4. evidence gate: nothing close enough -> not_covered, no answer call
    5. call 2: answer. No fallback model is left: if it fails -> "try again".
    With QUERY_REWRITE=off: retrieve the original question, gate, then call 1 answers
    and call 2 is the fallback model.
    Never a 3rd call (PS.md C2). Every stage has a hard time limit (EVALUATION.md).

Try it:  python -m app.pipeline acme "Can I book a cab on Saturday?"
"""
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field


from app import messages
from app.answerer import SYSTEM_PROMPT, Citation, InvalidOutput, build_user_prompt, parse_answer
from app.auth import known_companies
from app.config import Settings, get_settings
from app.guardrails import echoed_injection, mentions_other_company, other_company_terms
from app.llm import LLM, LLMError, build_llm
from app.retriever import retrieve
from app.rewriter import rewrite_query
from app.store import PolicyStore
from app.timeouts import StageTimeout, run_with_timeout

log = logging.getLogger("policy_assistant")

MAX_LLM_CALLS = 2            # PS.md C2
MIN_TIME_FOR_CALL_S = 0.6    # don't start a call that can't plausibly finish before the deadline


@dataclass
class AskResult:
    status: str                       # "answered" | "not_covered" | "unavailable"
    answer: str                       # always safe to show the employee
    citations: list[Citation]
    request_id: str
    llm_calls: int = 0
    models: list[str] = field(default_factory=list)
    reason: str = ""                  # internal: why this outcome (logs / eval notes)
    retrieved: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    policy_version: int | None = None  # the ONE version this whole request used


def ask(store: PolicyStore, llm: LLM, company_id: str, question: str,
        settings: Settings | None = None) -> AskResult:
    s = settings or get_settings()
    rid = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()
    deadline = t_start + s.answer_deadline_s
    question = " ".join(question.split())  # collapse whitespace / newlines
    version: int | None = None   # pinned in step 1

    def done(status: str, answer: str, citations: list[Citation], reason: str,
             calls: int = 0, models: list[str] | None = None) -> AskResult:
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        log.info("ask rid=%s company=%s v=%s status=%s calls=%d models=%s reason=%s ms=%s",
                 rid, company_id, version, status, calls, models or [], reason, timings)
        return AskResult(status, answer, citations, rid, calls, models or [], reason,
                         retrieved, timings, version)

    timings: dict = {}
    retrieved: list[dict] = []
    models_used, failures = [], []

    # 1. Pin the company's ACTIVE policy version once. Every step below (rewrite
    #    outline, retrieval, citations) uses it, so a switch to a new version in the
    #    middle of this request can't mix two versions (PS.md R2).
    try:
        version = store.active_version(company_id)
    except Exception:
        log.exception("store unavailable rid=%s", rid)
        return done("unavailable", messages.TRY_AGAIN, [], "store error")
    if version is None:
        return done("not_covered", messages.NOT_COVERED, [], "no policy")

    # 2. Query rewrite + input guard (LLM call 1 of 2) -----------------------------
    queries: list[str] | None = None          # None = search with the original question
    if s.query_rewrite:
        budget = min(s.rewrite_timeout_s, deadline - time.perf_counter() - MIN_TIME_FOR_CALL_S)
        if budget >= 0.3:
            models_used.append(f"{s.llm_model} (rewrite)")
            t0 = time.perf_counter()
            rw = rewrite_query(llm, store, company_id, question, s.llm_model, budget, version)
            timings["rewrite"] = round((time.perf_counter() - t0) * 1000, 1)
            if rw.ok and rw.verdict != "ok" and not rw.queries:
                # Guardrail (input): an attack / other-company / off-topic request with
                # no real transport question in it. Stop here: 1 call, no answer call.
                timings["rewrite_verdict"] = rw.verdict
                return done("not_covered", messages.GUARDRAIL if rw.verdict != "off_topic"
                            else messages.NOT_COVERED, [], f"guardrail: input {rw.verdict}",
                            len(models_used), models_used)
            if rw.ok and rw.verdict != "ok":
                failures.append(f"input flagged {rw.verdict}; answering the transport part only")
            if rw.ok and rw.queries:
                queries = rw.queries
            else:
                failures.append(f"rewrite: {rw.error or 'no queries'}; searching the original question")
        else:
            failures.append("rewrite skipped: no time")

    # 3. ONE retrieval: the rewritten queries (merged round-robin), or the original
    #    question if there are none. Hard time limit, leaving the answer call its time.
    try:
        limit = min(s.retrieval_timeout_s,
                    deadline - time.perf_counter() - (MIN_TIME_FOR_CALL_S if models_used else 0))
        if queries:
            r = run_with_timeout(retrieve, limit, store, company_id, question, rewrites=queries,
                                 include_original=False, fusion="round_robin", version=version)
        else:
            r = run_with_timeout(retrieve, limit, store, company_id, question, version=version)
    except StageTimeout as e:
        log.warning("retrieval timed out rid=%s: %s", rid, e)
        return done("unavailable", messages.TRY_AGAIN, [], "retrieval timeout",
                    len(models_used), models_used)
    except Exception:
        log.exception("retrieval failed rid=%s", rid)
        return done("unavailable", messages.TRY_AGAIN, [], "retrieval error",
                    len(models_used), models_used)
    timings.update({f"retrieve_{k}": v for k, v in r.timings_ms.items()})
    retrieved = [{"section": c.section, "distance": round(c.distance, 3)} for c in r.chunks]

    # 4. Evidence gate: nothing close enough -> "not covered", no answer call.
    if not r.has_evidence:
        return done("not_covered", messages.NOT_COVERED, [],
                    "; ".join(failures + [f"gate (best distance {r.best_distance:.3f})"
                                          if r.chunks else "no policy"]),
                    len(models_used), models_used)

    # 5. Answer (the remaining LLM calls) -------------------------------------------
    user_prompt = build_user_prompt(question, r.chunks)
    answer_models = (s.llm_model, s.llm_fallback_model)[:MAX_LLM_CALLS - len(models_used)]
    for model in answer_models:
        remaining = deadline - time.perf_counter()
        if remaining < MIN_TIME_FOR_CALL_S:
            failures.append(f"no time left for {model} ({remaining:.2f}s)")
            break
        models_used.append(model)
        t0 = time.perf_counter()
        try:
            raw = llm.complete_json(SYSTEM_PROMPT, user_prompt, model,
                                    timeout_s=min(s.llm_timeout_s, remaining))
            result = parse_answer(raw, r.chunks)
        except (LLMError, InvalidOutput) as e:
            timings[f"llm_{len(models_used)}"] = round((time.perf_counter() - t0) * 1000, 1)
            kind = getattr(e, "kind", "invalid_output")
            failures.append(f"{model}: {kind}")
            log.warning("llm call failed rid=%s model=%s kind=%s detail=%s", rid, model, kind, e)
            continue
        timings[f"llm_{len(models_used)}"] = round((time.perf_counter() - t0) * 1000, 1)

        if result.status == "blocked":
            return done("not_covered", messages.GUARDRAIL, [],
                        "; ".join(failures + ["guardrail: answer call refused"]),
                        len(models_used), models_used)
        if result.status == "answered":
            # Guardrail (output), code backstop: the model can be talked into things,
            # so these checks have the final word. On a hit: plain "not covered".
            hit = echoed_injection(result.text)
            if hit:
                reason = f"guardrail: output repeats injected text {hit!r}"
            elif other := mentions_other_company(
                    result.text, other_company_terms(store, company_id, known_companies())):
                reason = f"guardrail: output names another company {other!r}"
            else:
                return done("answered", result.text, result.citations,
                            "; ".join(failures) or "ok", len(models_used), models_used)
            log.warning("output guardrail rid=%s: %s", rid, reason)
            return done("not_covered", messages.NOT_COVERED, [], "; ".join(failures + [reason]),
                        len(models_used), models_used)
        reason = result.downgraded or "model said not covered"
        return done("not_covered", messages.NOT_COVERED, [],
                    "; ".join(failures + [reason]), len(models_used), models_used)

    return done("unavailable", messages.TRY_AGAIN, [], "; ".join(failures),
                len(models_used), models_used)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: python -m app.pipeline <company_id> "<question>"')
    from app.embeddings import embed_query
    from app.store import open_store

    s = get_settings()
    embed_query("warm up")
    res = ask(open_store(s.data_dir), build_llm(s), sys.argv[1], sys.argv[2], s)
    print(f"\n[{sys.argv[1]}] {sys.argv[2]}")
    print(f"status   : {res.status}")
    print(f"answer   : {res.answer}")
    for c in res.citations:
        print(f"citation : {c.title}, §{c.section} {c.section_title}")
    print(f"llm calls: {res.llm_calls} {res.models}   provider: {s.llm_provider}")
    print(f"reason   : {res.reason}")
    print(f"retrieved: {res.retrieved}")
    print(f"timings  : {res.timings_ms}")


if __name__ == "__main__":
    main()
