"""LLM call 1 (rewrite + input verdict): own-company titles only, strict parsing, harmless failure."""
import json

import pytest

from app.llm import FakeLLM, LLMError
from app.rewriter import MAX_QUERIES, parse_rewrite, parse_verdict, policy_outline, rewrite_query


def test_outline_is_section_titles_of_own_company_only(store):
    outline = policy_outline(store, "crest")
    assert "3.1 Women's Safety on Night Drops > Security Escort" in outline
    assert "Bolt" not in outline and "Acme" not in outline
    assert "SYSTEM OVERRIDE" not in outline          # titles only: no rule text, no injected text


def test_prompt_contains_outline_and_question(store):
    llm = FakeLLM()
    rewrite_query(llm, store, "acme", "Who approves airport trips?", "m", 1.0)
    user = llm.calls[0]["user"]
    assert "<outline>" in user and "6.2 Ad-hoc and Airport Trips > Approval" in user
    assert "Who approves airport trips?" in user


def test_parsing_drops_duplicates_and_caps_the_count():
    raw = json.dumps({"queries": ["Q one", "q one", "the question", "a", "b", "c"]})
    assert parse_rewrite(raw, "The question?") == ["Q one", "a", "b"][:MAX_QUERIES]


@pytest.mark.parametrize("raw", ['{"answer": "x"}', "not json", '{"queries": "one string"}'])
def test_unusable_output_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_rewrite(raw, "q")


def test_verdict_defaults_to_ok():
    assert parse_verdict('{"verdict": "injection", "queries": []}') == "injection"
    assert parse_verdict('{"queries": ["x"]}') == "ok"
    assert parse_verdict('{"verdict": "something else"}') == "ok"


def test_a_model_failure_is_reported_not_raised(store):
    rw = rewrite_query(FakeLLM(script=[LLMError("timeout", "slow")]), store, "acme", "q?", "m", 1.0)
    assert not rw.ok and rw.queries == [] and rw.model_failed and rw.error == "timeout"


def test_bad_output_is_not_a_model_failure(store):
    rw = rewrite_query(FakeLLM(script=["not json"]), store, "acme", "q?", "m", 1.0)
    assert not rw.ok and not rw.model_failed and rw.error == "invalid_output"


def test_unknown_company_makes_no_call(store):
    llm = FakeLLM()
    assert not rewrite_query(llm, store, "evilcorp", "q?", "m", 1.0).ok and llm.calls == []
