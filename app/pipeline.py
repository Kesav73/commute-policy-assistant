"""The /ask flow: question -> checked answer with citations. At most 2 LLM calls.

    1. pin the caller's company's LIVE policy version (used by every step below)
    2. LLM call 1 (primary model): rewrite the question into 1-3 search queries and
       judge it. A pure attack or off-topic question stops here with a plain message.
       If the call fails, we search the original question instead, and if the MODEL
       itself failed (timeout, outage), call 2 goes to the fallback model.
    3. one retrieval for those queries -> top 3 chunks
    4. evidence gate: nothing close enough -> "not covered" (no answer call)
    5. LLM call 2: answer from the 3 excerpts
    6. checks in code: valid citations, no other company, no injected text

Never a 3rd call. Every step has a time limit, and the whole flow a deadline, so a
slow model or search gives "please try again", never a hang.
"""
import logging
import time
import uuid
from dataclasses import dataclass, field

from app import messages
from app.answerer import SYSTEM_PROMPT, Citation, InvalidOutput, build_user_prompt, parse_answer
from app.auth import known_companies
from app.config import FALLBACK_MODEL, PRIMARY_MODEL, Settings, get_settings
from app.guardrails import echoed_injection, mentions_other_company, other_company_terms
from app.llm import LLMError
from app.retriever import retrieve
from app.rewriter import rewrite_query
from app.store import PolicyStore
from app.timeouts import StageTimeout, run_with_timeout

log = logging.getLogger("policy_assistant")

MAX_LLM_CALLS = 2            # the brief's limit
MIN_TIME_FOR_CALL_S = 0.6    # don't start an LLM call that can't finish before the deadline


@dataclass
class AskResult:
    status: str                       # "answered" | "not_covered" | "unavailable"
    answer: str                       # always safe to show the employee
    citations: list[Citation]
    request_id: str
    policy_version: int | None = None
    llm_calls: int = 0
    models: list[str] = field(default_factory=list)
    reason: str = ""                  # why this outcome (logs and evaluation notes)
    retrieved: list[dict] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


class _Trace:
    """Collects what happened during one request, and builds the final AskResult."""

    def __init__(self, company_id: str):
        self.company_id = company_id
        self.request_id = uuid.uuid4().hex[:12]
        self.started = time.perf_counter()
        self.version = None
        self.models = []       # one entry per LLM call made
        self.notes = []        # anything worth knowing about this request
        self.retrieved = []
        self.timings = {}

    def finish(self, status: str, answer: str, citations=None, note: str = "") -> AskResult:
        if note:
            self.notes.append(note)
        self.timings["total"] = round((time.perf_counter() - self.started) * 1000, 1)
        reason = "; ".join(self.notes) or "ok"
        # One log line per request. Never the question text: it may contain personal details.
        log.info("ask rid=%s company=%s v=%s status=%s calls=%d models=%s reason=%s ms=%s",
                 self.request_id, self.company_id, self.version, status, len(self.models),
                 self.models, reason, self.timings)
        return AskResult(status, answer, citations or [], self.request_id, self.version,
                         len(self.models), self.models, reason, self.retrieved, self.timings)


def ask(store: PolicyStore, llm, company_id: str, question: str, settings: Settings | None = None) -> AskResult:
    s = settings or get_settings()
    trace = _Trace(company_id)
    deadline = trace.started + s.answer_deadline_s
    question = " ".join(question.split())     # collapse extra spaces and newlines

    # 1. Pin the live policy version once, so an update in the middle of this request
    #    can't mix two versions.
    try:
        trace.version = store.active_version(company_id)
    except Exception:
        log.exception("store unavailable rid=%s", trace.request_id)
        return trace.finish("unavailable", messages.TRY_AGAIN, note="store error")
    if trace.version is None:
        return trace.finish("not_covered", messages.NOT_COVERED, note="no policy for this company")

    # 2. LLM call 1: rewrite + input guardrail.
    queries = [question]
    answer_model = PRIMARY_MODEL
    rewrite_budget = min(s.rewrite_timeout_s, deadline - time.perf_counter() - MIN_TIME_FOR_CALL_S)
    if rewrite_budget >= 0.3:
        trace.models.append(f"{PRIMARY_MODEL} (rewrite)")
        t0 = time.perf_counter()
        rewrite = rewrite_query(llm, store, company_id, question, PRIMARY_MODEL, rewrite_budget, trace.version)
        trace.timings["rewrite"] = round((time.perf_counter() - t0) * 1000, 1)

        if rewrite.ok and rewrite.verdict != "ok" and not rewrite.queries:
            # An attack / another company's policy / off-topic, with no real question in it.
            message = messages.NOT_COVERED if rewrite.verdict == "off_topic" else messages.GUARDRAIL
            return trace.finish("not_covered", message, note=f"guardrail: input {rewrite.verdict}")
        if rewrite.ok and rewrite.verdict != "ok":
            trace.notes.append(f"input flagged {rewrite.verdict}; answering the transport part only")
        if rewrite.ok and rewrite.queries:
            queries = rewrite.queries
        else:
            trace.notes.append(f"rewrite: {rewrite.error or 'no queries'}; searching the original question")
        if rewrite.model_failed:
            answer_model = FALLBACK_MODEL     # the primary model just failed: don't rely on it again
    else:
        trace.notes.append("rewrite skipped: no time")

    # 3. Retrieval (one search per query), with a hard time limit.
    try:
        time_left = deadline - time.perf_counter() - MIN_TIME_FOR_CALL_S
        result = run_with_timeout(retrieve, min(s.retrieval_timeout_s, time_left),
                                  store, company_id, queries, version=trace.version)
    except StageTimeout as e:
        log.warning("retrieval timed out rid=%s: %s", trace.request_id, e)
        return trace.finish("unavailable", messages.TRY_AGAIN, note="retrieval timeout")
    except Exception:
        log.exception("retrieval failed rid=%s", trace.request_id)
        return trace.finish("unavailable", messages.TRY_AGAIN, note="retrieval error")
    trace.timings["retrieve"] = result.timings_ms["retrieve"]
    trace.retrieved = [{"section": c.section, "distance": round(c.distance, 3)} for c in result.chunks]

    # 4. Evidence gate.
    if not result.has_evidence:
        if result.chunks:
            gate_note = f"gate (best distance {result.best_distance:.3f})"
        else:
            gate_note = "no policy chunks"
        return trace.finish("not_covered", messages.NOT_COVERED, note=gate_note)

    # 5. LLM call 2: the answer. If call 1 was skipped, a failed answer may be retried
    #    once on the fallback model (still at most 2 calls in total).
    calls_left = MAX_LLM_CALLS - len(trace.models)
    models_to_try = [answer_model]
    if calls_left >= 2 and answer_model != FALLBACK_MODEL:
        models_to_try.append(FALLBACK_MODEL)
    prompt = build_user_prompt(question, result.chunks)

    for model in models_to_try:
        time_left = deadline - time.perf_counter()
        if time_left < MIN_TIME_FOR_CALL_S:
            trace.notes.append(f"no time left for {model}")
            break
        trace.models.append(model)
        t0 = time.perf_counter()
        try:
            raw = llm.complete_json(SYSTEM_PROMPT, prompt, model, timeout_s=min(s.llm_timeout_s, time_left))
            answer = parse_answer(raw, result.chunks)
        except (LLMError, InvalidOutput) as e:
            kind = getattr(e, "kind", "invalid_output")
            trace.notes.append(f"{model}: {kind}")
            log.warning("answer call failed rid=%s model=%s kind=%s: %s", trace.request_id, model, kind, e)
            continue
        finally:
            trace.timings[f"answer_{len(trace.models)}"] = round((time.perf_counter() - t0) * 1000, 1)
        return _check_answer(trace, answer, store, company_id)

    return trace.finish("unavailable", messages.TRY_AGAIN)


def _check_answer(trace: _Trace, answer, store: PolicyStore, company_id: str) -> AskResult:
    """6. The model's answer is untrusted: these code checks have the final word."""
    if answer.status == "blocked":
        return trace.finish("not_covered", messages.GUARDRAIL, note="guardrail: answer call refused")
    if answer.status != "answered":
        return trace.finish("not_covered", messages.NOT_COVERED,
                            note=answer.downgraded or "model said not covered")

    echoed = echoed_injection(answer.text)
    if echoed:
        return trace.finish("not_covered", messages.NOT_COVERED,
                            note=f"guardrail: output repeats injected text {echoed!r}")
    other = mentions_other_company(answer.text, other_company_terms(store, company_id, known_companies()))
    if other:
        return trace.finish("not_covered", messages.NOT_COVERED,
                            note=f"guardrail: output names another company {other!r}")
    return trace.finish("answered", answer.text, answer.citations)
