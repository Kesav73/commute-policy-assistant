"""The ask flow: question -> retrieve -> gate -> LLM (max 2 calls) -> checked answer.

    1. retrieve from the caller's company only, with the ORIGINAL question
    2. evidence gate: nothing close enough -> not_covered, 0 LLM calls
    With QUERY_REWRITE=on (default, docs/journey/13):
    3. call 1: rewrite the question into 1-3 search queries (policy outline as context);
       retrieve again with those queries, merged round-robin, top 3.
       If the rewrite fails, keep the step-1 chunks (harmless).
    4. call 2: answer (primary model). No fallback model is left: if it fails -> "try again".
    With QUERY_REWRITE=off:
    3. call 1: answer with the primary model (Qwen)
    4. if call 1 failed for ANY reason and there is time left -> call 2: fallback model
    Never a 3rd call (PS.md C2).

Try it:  python -m app.pipeline acme "Can I book a cab on Saturday?"
"""
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field


from app import messages
from app.answerer import SYSTEM_PROMPT, Citation, InvalidOutput, build_user_prompt, parse_answer
from app.config import Settings, get_settings
from app.llm import LLM, LLMError, build_llm
from app.retriever import retrieve
from app.rewriter import rewrite_query
from app.store import PolicyStore

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


def ask(store: PolicyStore, llm: LLM, company_id: str, question: str,
        settings: Settings | None = None) -> AskResult:
    s = settings or get_settings()
    rid = uuid.uuid4().hex[:12]
    t_start = time.perf_counter()
    deadline = t_start + s.answer_deadline_s
    question = " ".join(question.split())  # collapse whitespace / newlines

    def done(status: str, answer: str, citations: list[Citation], reason: str,
             calls: int = 0, models: list[str] | None = None) -> AskResult:
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        log.info("ask rid=%s company=%s status=%s calls=%d models=%s reason=%s ms=%s",
                 rid, company_id, status, calls, models or [], reason, timings)
        return AskResult(status, answer, citations, rid, calls, models or [], reason,
                         retrieved, timings)

    # 1-2. Retrieval + evidence gate --------------------------------------------
    timings: dict = {}
    retrieved: list[dict] = []
    try:
        r = retrieve(store, company_id, question)
    except Exception:
        log.exception("retrieval failed rid=%s", rid)
        return done("unavailable", messages.TRY_AGAIN, [], "retrieval error")
    timings.update({f"retrieve_{k}": v for k, v in r.timings_ms.items()})
    retrieved = [{"section": c.section, "distance": round(c.distance, 3)} for c in r.chunks]

    if not r.has_evidence:
        return done("not_covered", messages.NOT_COVERED, [],
                    f"gate (best distance {r.best_distance:.3f})" if r.chunks else "no policy")

    models_used, failures = [], []

    # 3. Query rewrite (LLM call 1 of 2) ------------------------------------------
    if s.query_rewrite:
        budget = min(s.rewrite_timeout_s, deadline - time.perf_counter() - MIN_TIME_FOR_CALL_S)
        if budget >= 0.3:
            models_used.append(f"{s.llm_model} (rewrite)")
            t0 = time.perf_counter()
            rw = rewrite_query(llm, store, company_id, question, s.llm_model, budget)
            timings["rewrite"] = round((time.perf_counter() - t0) * 1000, 1)
            if rw.ok and rw.queries:
                try:
                    r = retrieve(store, company_id, question, rewrites=rw.queries,
                                 include_original=False, fusion="round_robin")
                    timings["retrieve_rewritten"] = r.timings_ms.get("multi_query")
                    retrieved = [{"section": c.section, "distance": round(c.distance, 3)}
                                 for c in r.chunks]
                except Exception:
                    log.exception("rewritten retrieval failed rid=%s; using original chunks", rid)
                    failures.append("rewritten retrieval error")
            else:
                failures.append(f"rewrite: {rw.error or 'no queries'}")
        else:
            failures.append("rewrite skipped: no time")

    # 4. Answer (the remaining LLM calls) -------------------------------------------
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

        if result.status == "answered":
            return done("answered", result.text, result.citations,
                        "; ".join(failures) or "ok", len(models_used), models_used)
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
