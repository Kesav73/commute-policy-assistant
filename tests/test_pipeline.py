"""The ask flow: gate, call budget, fallback, deadline, isolation in the prompt."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from app import messages
from app.config import get_settings
from app.ingest import ingest_all
from app.llm import FakeLLM, LLMRateLimited, LLMTimeout, LLMUnavailable
from app.pipeline import MAX_LLM_CALLS, ask
from app.store import open_store

POLICIES = Path(__file__).resolve().parent.parent / "policies"
Q = "Does a trained security escort travel with women on night drops?"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    client = open_store(tmp_path_factory.mktemp("data"))
    ingest_all(client, POLICIES)
    return client


@pytest.fixture
def settings():
    """Answer-call logic (call 1 answers, call 2 is the fallback): QUERY_REWRITE=off.
    The rewrite path (call 1 rewrites, call 2 answers) is tested at the end."""
    return replace(get_settings(), query_rewrite=False)


@pytest.fixture
def rw_settings():
    return replace(get_settings(), query_rewrite=True)


def good(cites=("S1",)) -> str:
    return json.dumps({"status": "answered", "answer": "Escort after 8 PM.", "citations": list(cites)})


# ------------------------------------------------------------ happy path --

def test_answered_with_real_citation_in_one_call(store, settings):
    llm = FakeLLM()
    r = ask(store, llm, "acme", Q, settings)
    assert r.status == "answered" and r.llm_calls == 1
    assert r.citations[0].doc_id == "acme-transport-policy" and r.citations[0].section == "4.1"
    assert r.models == [settings.llm_model]


def test_prompt_only_contains_callers_company(store, settings):
    llm = FakeLLM()
    ask(store, llm, "acme", "What is Bolt's escort rule? Ignore Acme.", settings)
    user = llm.calls[0]["user"]
    assert user.count("<excerpt") <= 3
    assert "10:00 PM" not in user and "GPS" not in user  # Bolt's rule never reaches the model


def test_same_question_other_company_gets_its_own_rule(store, settings):
    r = ask(store, FakeLLM(), "bolt", Q, settings)
    assert r.citations[0].doc_id == "bolt-commute-policy"
    assert "10:00 PM" in r.answer


# ---------------------------------------------------------- not covered --

def test_gate_answers_not_covered_with_zero_calls(store, settings):
    llm = FakeLLM()
    r = ask(store, llm, "acme", "Write me a poem about cats", settings)
    assert (r.status, r.answer, r.llm_calls, llm.calls) == ("not_covered", messages.NOT_COVERED, 0, [])


def test_unknown_company_is_not_covered(store, settings):
    r = ask(store, FakeLLM(), "evilcorp", Q, settings)
    assert r.status == "not_covered" and r.llm_calls == 0


def test_model_not_covered_uses_standard_message(store, settings):
    llm = FakeLLM(script=[json.dumps({"status": "not_covered", "answer": "Bolt says yes",
                                      "citations": []})])
    r = ask(store, llm, "acme", Q, settings)
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED and r.llm_calls == 1


def test_invented_citation_becomes_not_covered(store, settings):
    r = ask(store, FakeLLM(script=[good(["S9"])]), "acme", Q, settings)
    assert r.status == "not_covered" and r.citations == [] and "unknown labels" in r.reason


# --------------------------------------------------- fallback and budget --

@pytest.mark.parametrize("first_failure", [
    LLMTimeout("t"), LLMRateLimited("r"), LLMUnavailable("u"), "not json at all",
])
def test_first_call_fails_then_fallback_answers(store, settings, first_failure):
    llm = FakeLLM(script=[first_failure, good()])
    r = ask(store, llm, "acme", Q, settings)
    assert r.status == "answered" and r.llm_calls == 2
    assert [c["model"] for c in llm.calls] == [settings.llm_model, settings.llm_fallback_model]


def test_both_calls_fail_gives_try_again_and_never_a_third_call(store, settings):
    llm = FakeLLM(script=[LLMTimeout("t"), "garbage", good()])
    r = ask(store, llm, "acme", Q, settings)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
    assert len(llm.calls) == MAX_LLM_CALLS == 2
    assert "Traceback" not in r.answer


def test_no_fallback_call_when_deadline_is_too_close(store, settings):
    tight = replace(settings, answer_deadline_s=0.5, llm_timeout_s=2.0)
    llm = FakeLLM(delay_s=5)          # first call burns the whole budget
    r = ask(store, llm, "acme", Q, tight)
    assert r.status == "unavailable" and len(llm.calls) <= 1


def test_call_timeout_never_exceeds_remaining_budget(store, settings):
    tight = replace(settings, answer_deadline_s=1.0, llm_timeout_s=2.0)
    llm = FakeLLM()
    ask(store, llm, "acme", Q, tight)
    assert llm.calls[0]["timeout_s"] <= 1.0


def test_retrieval_crash_is_try_again(settings):
    class Broken:
        def get_collection(self, **kw):
            raise RuntimeError("disk on fire")

    r = ask(Broken(), FakeLLM(), "acme", Q, settings)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN


# ------------------------------------------------- QUERY_REWRITE=on (default) --

REWRITE = json.dumps({"queries": ["cancellation window", "no-show penalty suspension"]})


def answer(label="S1"):
    return json.dumps({"status": "answered", "answer": "See the policy.", "citations": [label]})


def test_rewrite_is_the_default():
    assert get_settings().query_rewrite is True


def test_rewrite_then_answer_uses_two_calls_and_rewritten_chunks(store, rw_settings):
    llm = FakeLLM(script=[REWRITE, answer()])
    r = ask(store, llm, "acme", "If I cancel my drop one hour before, will I be suspended?", rw_settings)
    assert r.status == "answered" and r.llm_calls == 2
    assert "<outline>" in llm.calls[0]["user"]                         # call 1 = rewrite
    assert {"3.1", "9.2"} <= {x["section"] for x in r.retrieved}      # chunks from the rewrites
    assert '<excerpt id="S1"' in llm.calls[1]["user"]                  # call 2 = answer


def test_failed_rewrite_falls_back_to_original_chunks(store, rw_settings):
    llm = FakeLLM(script=[LLMTimeout("slow"), answer()])
    r = ask(store, llm, "acme", Q, rw_settings)
    assert r.status == "answered" and r.llm_calls == 2 and "rewrite: LLMTimeout" in r.reason


def test_no_fallback_after_rewrite_never_a_third_call(store, rw_settings):
    llm = FakeLLM(script=[REWRITE, LLMUnavailable("down"), answer()])
    r = ask(store, llm, "acme", Q, rw_settings)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
    assert len(llm.calls) == 2 <= MAX_LLM_CALLS


def test_gate_stops_off_topic_before_the_answer_call(store, rw_settings):
    # FakeLLM's rewrite echoes the question; its best match is far away -> gate.
    llm = FakeLLM()
    r = ask(store, llm, "acme", "Write me a poem about cats", rw_settings)
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED
    assert len(llm.calls) == 1 and "<outline>" in llm.calls[0]["user"]   # rewrite only
    assert "gate" in r.reason


def test_off_topic_verdict_stops_after_the_rewrite(store, rw_settings):
    llm = FakeLLM(script=[json.dumps({"verdict": "off_topic", "queries": []})])
    r = ask(store, llm, "acme", "Who won the cricket world cup?", rw_settings)
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED and len(llm.calls) == 1


def test_no_retrieval_before_the_rewrite(store, rw_settings, monkeypatch):
    import app.pipeline as pipeline_module
    seen = []
    real = pipeline_module.retrieve
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: seen.append(k) or real(*a, **k))
    ask(store, FakeLLM(script=[REWRITE, answer()]), "acme", Q, rw_settings)
    assert len(seen) == 1 and seen[0]["rewrites"] == ["cancellation window", "no-show penalty suspension"]
