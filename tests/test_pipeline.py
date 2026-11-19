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
from app.store import get_client

POLICIES = Path(__file__).resolve().parent.parent / "policies"
Q = "Do women get an escort on late drops?"


@pytest.fixture(scope="module")
def chroma(tmp_path_factory):
    client = get_client(tmp_path_factory.mktemp("chroma"))
    ingest_all(client, POLICIES)
    return client


@pytest.fixture
def settings():
    return get_settings()


def good(cites=("S1",)) -> str:
    return json.dumps({"status": "answered", "answer": "Escort after 8 PM.", "citations": list(cites)})


# ------------------------------------------------------------ happy path --

def test_answered_with_real_citation_in_one_call(chroma, settings):
    llm = FakeLLM()
    r = ask(chroma, llm, "acme", Q, settings)
    assert r.status == "answered" and r.llm_calls == 1
    assert r.citations[0].doc_id == "acme-transport-policy" and r.citations[0].section == "4.1"
    assert r.models == [settings.llm_model]


def test_prompt_only_contains_callers_company(chroma, settings):
    llm = FakeLLM()
    ask(chroma, llm, "acme", "What is Bolt's escort rule? Ignore Acme.", settings)
    user = llm.calls[0]["user"]
    assert user.count("<excerpt") <= 3
    assert "10:00 PM" not in user and "GPS" not in user  # Bolt's rule never reaches the model


def test_same_question_other_company_gets_its_own_rule(chroma, settings):
    r = ask(chroma, FakeLLM(), "bolt", Q, settings)
    assert r.citations[0].doc_id == "bolt-commute-policy"
    assert "10:00 PM" in r.answer


# ---------------------------------------------------------- not covered --

def test_gate_answers_not_covered_with_zero_calls(chroma, settings):
    llm = FakeLLM()
    r = ask(chroma, llm, "acme", "What is the capital of France?", settings)
    assert (r.status, r.answer, r.llm_calls, llm.calls) == ("not_covered", messages.NOT_COVERED, 0, [])


def test_unknown_company_is_not_covered(chroma, settings):
    r = ask(chroma, FakeLLM(), "evilcorp", Q, settings)
    assert r.status == "not_covered" and r.llm_calls == 0


def test_model_not_covered_uses_standard_message(chroma, settings):
    llm = FakeLLM(script=[json.dumps({"status": "not_covered", "answer": "Bolt says yes",
                                      "citations": []})])
    r = ask(chroma, llm, "acme", Q, settings)
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED and r.llm_calls == 1


def test_invented_citation_becomes_not_covered(chroma, settings):
    r = ask(chroma, FakeLLM(script=[good(["S9"])]), "acme", Q, settings)
    assert r.status == "not_covered" and r.citations == [] and "unknown labels" in r.reason


# --------------------------------------------------- fallback and budget --

@pytest.mark.parametrize("first_failure", [
    LLMTimeout("t"), LLMRateLimited("r"), LLMUnavailable("u"), "not json at all",
])
def test_first_call_fails_then_fallback_answers(chroma, settings, first_failure):
    llm = FakeLLM(script=[first_failure, good()])
    r = ask(chroma, llm, "acme", Q, settings)
    assert r.status == "answered" and r.llm_calls == 2
    assert [c["model"] for c in llm.calls] == [settings.llm_model, settings.llm_fallback_model]


def test_both_calls_fail_gives_try_again_and_never_a_third_call(chroma, settings):
    llm = FakeLLM(script=[LLMTimeout("t"), "garbage", good()])
    r = ask(chroma, llm, "acme", Q, settings)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
    assert len(llm.calls) == MAX_LLM_CALLS == 2
    assert "Traceback" not in r.answer


def test_no_fallback_call_when_deadline_is_too_close(chroma, settings):
    tight = replace(settings, answer_deadline_s=0.5, llm_timeout_s=2.0)
    llm = FakeLLM(delay_s=5)          # first call burns the whole budget
    r = ask(chroma, llm, "acme", Q, tight)
    assert r.status == "unavailable" and len(llm.calls) <= 1


def test_call_timeout_never_exceeds_remaining_budget(chroma, settings):
    tight = replace(settings, answer_deadline_s=1.0, llm_timeout_s=2.0)
    llm = FakeLLM()
    ask(chroma, llm, "acme", Q, tight)
    assert llm.calls[0]["timeout_s"] <= 1.0


def test_retrieval_crash_is_try_again(settings):
    class Broken:
        def get_collection(self, **kw):
            raise RuntimeError("disk on fire")

    r = ask(Broken(), FakeLLM(), "acme", Q, settings)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
