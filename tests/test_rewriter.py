"""Query rewriter: company-only context, strict output parsing, harmless failure."""
import json
from pathlib import Path

import pytest

from app.ingest import ingest_all
from app.llm import FakeLLM, LLMTimeout
from app.retriever import retrieve
from app.rewriter import MAX_QUERIES, parse_rewrite, policy_outline, rewrite_query
from app.store import open_store

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = open_store(tmp_path_factory.mktemp("data"))
    ingest_all(s, POLICIES)
    return s


def test_outline_is_titles_of_own_company_only(store):
    outline = policy_outline(store, "crest")
    assert "3.1 Women's Safety on Night Drops > Security Escort" in outline
    assert "Bolt" not in outline and "Acme" not in outline
    assert "SYSTEM OVERRIDE" not in outline          # titles only: no rule text, no injection


def test_prompt_contains_outline_and_question(store):
    llm = FakeLLM()
    rewrite_query(llm, store, "acme", "Who approves airport trips?", "m", 1.0)
    user = llm.calls[0]["user"]
    assert "<outline>" in user and "6.2 Ad-hoc and Airport Trips > Approval" in user
    assert "Who approves airport trips?" in user


def test_parse_dedups_caps_and_drops_the_original():
    raw = json.dumps({"queries": ["Q one", "q one", "the question", "a", "b", "c"]})
    assert parse_rewrite(raw, "The question?") == ["Q one", "a", "b"][:MAX_QUERIES]


@pytest.mark.parametrize("raw", ['{"answer": "x"}', "not json", '{"queries": "one string"}'])
def test_parse_rejects_bad_output(raw):
    with pytest.raises(ValueError):
        parse_rewrite(raw, "q")


def test_failure_is_harmless(store):
    rw = rewrite_query(FakeLLM(script=[LLMTimeout("slow")]), store, "acme", "q?", "m", 1.0)
    assert not rw.ok and rw.queries == [] and rw.error == "LLMTimeout"


def test_unknown_company_gets_no_call(store):
    llm = FakeLLM()
    assert not rewrite_query(llm, store, "evilcorp", "q?", "m", 1.0).ok and llm.calls == []


# ------------------------------------------------- retrieval with rewrites --

def test_rewrites_only_and_original_plus_rewrites(store):
    q = "If I cancel my drop one hour before, will I be suspended?"
    rw = ["cancellation window penalty", "no-show penalty suspension"]
    a = retrieve(store, "acme", q, rewrites=rw, include_original=False)
    b = retrieve(store, "acme", q, rewrites=rw, include_original=True)
    assert a.timings_ms["queries"] == 2 and b.timings_ms["queries"] == 3
    for r in (a, b):
        assert len(r.chunks) == 3 and {c.company_id for c in r.chunks} == {"acme"}


def test_gate_uses_the_best_distance_of_the_searched_queries(store):
    from app.retriever import rank_candidates
    qs = ["Write me a poem about cats", "escort for women on night drops"]
    r = retrieve(store, "acme", "Write me a poem about cats", rewrites=qs, include_original=False)
    best = min(rank_candidates(store, "acme", q).best_distance for q in qs)
    assert r.best_distance == pytest.approx(best) and r.has_evidence
    off = retrieve(store, "acme", "x", rewrites=["Write me a poem about cats"], include_original=False)
    assert not off.has_evidence

def test_round_robin_gives_each_subquery_its_best_hit():
    from app.retriever import round_robin, rrf
    cancel = ["3.1", "3.2", "2.2"]       # sub-query 1: cancellation
    penalty = ["9.2", "3.2", "9.1"]      # sub-query 2: no-show penalty
    assert round_robin([cancel, penalty])[:3] == ["3.1", "9.2", "3.2"]
    assert rrf([cancel, penalty])[0] == "3.2"   # RRF favours the id both agree on


def test_round_robin_fusion_in_retrieve(store):
    r = retrieve(store, "acme", "If I cancel my drop one hour before, will I be suspended?",
                 rewrites=["cancellation window", "no-show penalty suspension"],
                 include_original=False, fusion="round_robin")
    assert {"3.1", "9.2"} <= {c.section for c in r.chunks}
