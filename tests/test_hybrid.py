"""Hybrid retrieval: BM25 keywords + vectors, fused with weighted RRF."""
from pathlib import Path

import pytest

from app.bm25 import BM25Index, tokenize
from app.config import ConfigError, get_settings
from app.ingest import ingest_all
from app.retriever import retrieve, rrf
from app.store import open_store

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = open_store(tmp_path_factory.mktemp("data"))
    ingest_all(s, POLICIES)
    return s


# ------------------------------------------------------------------ BM25 --

def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("Who approves the booking?") == ["approves", "booking"]
    assert tokenize("Who approves the booking?", stem=True) == ["approv", "book"]
    assert tokenize("no-show") == ["no-show"]          # hyphenated terms stay whole


def test_bm25_prefers_rare_matching_words():
    idx = BM25Index(["a", "b", "c"], ["cab cab cab", "airport approval by HR Head", "cab booking"])
    s = idx.scores("HR Head approval")
    assert max(s, key=s.get) == "b" and s["a"] == 0.0


def test_stemming_matches_word_forms():
    idx = BM25Index(["x", "y"], ["Trips must be booked early.", "Guests are not allowed."], stem=True)
    assert idx.scores("booking")["x"] > 0
    assert BM25Index(["x"], ["Trips must be booked early."]).scores("booking")["x"] == 0


# ------------------------------------------------------------------- RRF --

def test_rrf_rewards_agreement():
    # "b" is 1st in one list and 2nd in the other; "a" and "c" are top of one list
    # but last of the other. Being near the top of BOTH lists wins.
    assert rrf([["a", "b", "c", "d"], ["b", "d", "c", "a"]])[0] == "b"


def test_rrf_weights_favour_the_heavier_ranking():
    assert rrf([["v", "k"], ["k", "v"]], weights=[2.0, 1.0])[0] == "v"
    assert rrf([["v", "k"], ["k", "v"]], weights=[1.0, 2.0])[0] == "k"


# ------------------------------------------------------------- retrieval --

@pytest.mark.parametrize("company", ["acme", "bolt", "crest"])
def test_hybrid_returns_only_callers_company(store, company):
    r = retrieve(store, company, "escort for women on night drops", mode="hybrid")
    assert len(r.chunks) == 3 and {c.company_id for c in r.chunks} == {company}
    assert "keyword" in r.timings_ms


def test_hybrid_gate_uses_best_vector_distance(store):
    v = retrieve(store, "acme", "Write me a poem about cats", mode="vector")
    h = retrieve(store, "acme", "Write me a poem about cats", mode="hybrid")
    assert h.best_distance == pytest.approx(v.best_distance)
    assert not h.has_evidence


def test_default_mode_is_hybrid():
    assert get_settings().retrieval_mode == "hybrid"


def test_bad_mode_is_a_config_error(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "magic")
    get_settings.cache_clear()
    try:
        with pytest.raises(ConfigError, match="RETRIEVAL_MODE"):
            get_settings()
    finally:
        monkeypatch.delenv("RETRIEVAL_MODE")
        get_settings.cache_clear()
