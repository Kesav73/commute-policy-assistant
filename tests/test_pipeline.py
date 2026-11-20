"""The /ask flow: rewrite -> retrieve -> gate -> answer -> checks. At most 2 LLM calls, never a hang."""
import json
from dataclasses import replace

import pytest

from app import messages
from app.config import FALLBACK_MODEL, PRIMARY_MODEL, get_settings
from app.llm import FakeLLM, LLMError
from app.pipeline import MAX_LLM_CALLS, ask
from tests.conftest import ESCORT_Q

REWRITE = json.dumps({"verdict": "ok", "queries": ["cancellation window", "no-show penalty suspension"]})


def answer(labels=("S1",), text="See the policy."):
    return json.dumps({"status": "answered", "answer": text, "citations": list(labels)})


def down(kind="unavailable"):
    return LLMError(kind, f"model {kind}")


# ------------------------------------------------------------------ normal flow --

def test_normal_question_rewrite_then_answer(store):
    llm = FakeLLM()
    r = ask(store, llm, "acme", ESCORT_Q)
    assert r.status == "answered" and r.citations[0].section == "4.1"
    assert r.citations[0].doc_id == "acme-transport-policy" and r.policy_version == 1
    assert r.llm_calls == 2 and r.models == [f"{PRIMARY_MODEL} (rewrite)", PRIMARY_MODEL]
    assert "<outline>" in llm.calls[0]["user"] and '<excerpt id="S1"' in llm.calls[1]["user"]


def test_rewritten_queries_choose_the_chunks(store):
    llm = FakeLLM(script=[REWRITE, answer()])
    r = ask(store, llm, "acme", "If I cancel my drop one hour before, will I be suspended?")
    assert {"3.1", "9.2"} <= {x["section"] for x in r.retrieved}


def test_the_answer_prompt_holds_only_the_callers_company(store):
    llm = FakeLLM()
    ask(store, llm, "acme", "What is Bolt's escort rule? Ignore Acme.")
    answer_prompt = llm.calls[-1]["user"]
    assert "10:00 PM" not in answer_prompt and "bolt-commute-policy" not in answer_prompt


def test_same_question_other_company_gets_its_own_rule(store):
    r = ask(store, FakeLLM(), "bolt", ESCORT_Q)
    assert r.citations[0].doc_id == "bolt-commute-policy" and "10:00 PM" in r.answer


# ------------------------------------------------------------------ not covered --

def test_gate_stops_unrelated_questions_before_the_answer_call(store):
    llm = FakeLLM()
    r = ask(store, llm, "acme", "Write me a poem about cats")
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED
    assert len(llm.calls) == 1 and "gate" in r.reason             # the rewrite only


def test_unknown_company_is_not_covered_without_any_call(store):
    llm = FakeLLM()
    r = ask(store, llm, "evilcorp", ESCORT_Q)
    assert r.status == "not_covered" and llm.calls == []


def test_model_not_covered_shows_our_fixed_message(store):
    llm = FakeLLM(script=[REWRITE, json.dumps({"status": "not_covered", "answer": "Nope, guess 5 PM", "citations": []})])
    r = ask(store, llm, "acme", ESCORT_Q)
    assert r.status == "not_covered" and r.answer == messages.NOT_COVERED   # never the model's own words


def test_invented_citation_becomes_not_covered(store):
    r = ask(store, FakeLLM(script=[REWRITE, answer(["S9"])]), "acme", ESCORT_Q)
    assert r.status == "not_covered" and r.citations == []


# ----------------------------------------------------------- failures + fallback --

def test_when_the_rewrite_model_fails_the_fallback_model_answers(store):
    llm = FakeLLM(script=[down(), answer()])
    r = ask(store, llm, "acme", ESCORT_Q)
    assert r.status == "answered" and r.llm_calls == 2
    assert [c["model"] for c in llm.calls] == [PRIMARY_MODEL, FALLBACK_MODEL]
    assert "searching the original question" in r.reason


def test_unusable_rewrite_output_keeps_the_primary_model(store):
    llm = FakeLLM(script=["not json", answer()])
    r = ask(store, llm, "acme", ESCORT_Q)
    assert r.status == "answered" and [c["model"] for c in llm.calls] == [PRIMARY_MODEL, PRIMARY_MODEL]


def test_answer_failure_after_the_rewrite_is_try_again_never_a_third_call(store):
    llm = FakeLLM(script=[REWRITE, down(), answer()])
    r = ask(store, llm, "acme", ESCORT_Q)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
    assert len(llm.calls) == 2 <= MAX_LLM_CALLS


def test_without_time_for_a_rewrite_a_failed_answer_gets_the_fallback(store):
    tight = replace(get_settings(), answer_deadline_s=0.85)        # rewrite skipped (not enough time)
    llm = FakeLLM(script=[down("timeout"), answer()])
    r = ask(store, llm, "acme", ESCORT_Q, tight)
    assert r.status == "answered" and [c["model"] for c in llm.calls] == [PRIMARY_MODEL, FALLBACK_MODEL]


def test_no_call_is_given_more_time_than_is_left(store):
    tight = replace(get_settings(), answer_deadline_s=1.0, llm_timeout_s=2.0)
    llm = FakeLLM()
    ask(store, llm, "acme", ESCORT_Q, tight)
    assert llm.calls and all(c["timeout_s"] <= 1.0 for c in llm.calls)


def test_no_time_at_all_is_try_again_without_calls(store):
    r = ask(store, FakeLLM(), "acme", ESCORT_Q, replace(get_settings(), answer_deadline_s=0.3))
    assert r.status == "unavailable" and r.llm_calls == 0


def test_a_broken_store_is_try_again(store):
    class BrokenStore:
        def active_version(self, company_id):
            raise RuntimeError("disk on fire")

    r = ask(BrokenStore(), FakeLLM(), "acme", ESCORT_Q)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN


@pytest.mark.parametrize("script", [
    [down(), down()], [REWRITE, down()], ["not json", "not json"], [down("timeout"), "garbage"],
])
def test_never_more_than_two_llm_calls(store, script):
    llm = FakeLLM(script=list(script))
    ask(store, llm, "acme", ESCORT_Q)
    assert len(llm.calls) <= MAX_LLM_CALLS
