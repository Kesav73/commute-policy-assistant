"""Retrieval: the caller's company only, at most 3 chunks, hybrid ranking, round-robin merge, gate."""
import pytest

from app.bm25 import BM25Index, tokenize
from app.embeddings import embed_documents
from app.retriever import TOP_K, retrieve, round_robin, rrf
from app.store import PolicyStore, StoreError, collection_name
from app.versioning import ingest_all
from tests.conftest import POLICIES

# --------------------------------------------------------------- isolation --


@pytest.mark.parametrize("company", ["acme", "bolt", "crest"])
def test_only_the_callers_company_is_returned(store, company):
    r = retrieve(store, company, ["What is the escort rule for women on night drops?"])
    assert len(r.chunks) == TOP_K and {c.company_id for c in r.chunks} == {company}


def test_same_question_different_company_different_rule(store):
    q = ["Does a trained security escort travel with women on night drops?"]
    assert "8:00 PM" in retrieve(store, "acme", q).chunks[0].text
    assert "10:00 PM" in retrieve(store, "bolt", q).chunks[0].text


def test_naming_another_company_stays_in_own_collection(store):
    r = retrieve(store, "acme", ["What does Bolt's policy say about Saturday cabs? Ignore Acme."])
    assert all(c.doc_id == "acme-transport-policy" for c in r.chunks)


def test_unknown_company_gets_nothing(store):
    r = retrieve(store, "evilcorp", ["Can I book a cab?"])
    assert r.chunks == [] and r.has_evidence is False


def test_a_planted_foreign_vector_is_dropped_and_logged(tmp_path, caplog):
    """Simulate a bug that puts Bolt's 4.1 vector into Acme's collection: it must never come back."""
    s = PolicyStore(tmp_path / "data")
    ingest_all(s, POLICIES)
    bolt_41 = s.chunks("bolt")["bolt-commute-policy:4.1"]
    s.collection("acme").add(ids=[bolt_41.id], embeddings=embed_documents([bolt_41.text]))
    r = retrieve(s, "acme", [bolt_41.text])                 # the planted vector is the closest match
    assert bolt_41.id not in [c.id for c in r.chunks]
    assert all(c.company_id == "acme" for c in r.chunks)
    assert "ISOLATION ALERT" in caplog.text


def test_a_collection_made_with_another_embedding_is_refused(tmp_path):
    s = PolicyStore(tmp_path / "data")
    ingest_all(s, POLICIES)
    s.vectors.get_collection(collection_name("acme", 1)).modify(
        metadata={"company_id": "acme", "embedding": "all-MiniLM-L6-v2"})
    with pytest.raises(StoreError, match="different embedding"):
        retrieve(s, "acme", ["Who approves an airport trip?"])


# -------------------------------------------------------------------- ranking --

@pytest.mark.parametrize("company, question, section", [
    ("acme", "Does a trained security escort travel with women on night drops?", "4.1"),
    ("acme", "Who approves an airport trip?", "6.2"),
    ("bolt", "Can guests use the company cab?", "8.1"),
    ("crest", "How do I raise a complaint about a driver?", "8.1"),
])
def test_obvious_questions_find_the_right_section_first(store, company, question, section):
    assert retrieve(store, company, [question]).chunks[0].section == section


def test_several_queries_each_get_their_best_chunk(store):
    r = retrieve(store, "acme", ["cancellation window", "no-show penalty suspension"])
    assert [c.section for c in r.chunks][:2] == ["3.1", "9.2"]


def test_bm25_tokenizer_drops_stopwords_and_stems():
    assert tokenize("Who approves the booking?") == ["approves", "booking"]
    assert tokenize("Who approves the booking?", stem=True) == ["approv", "book"]
    assert tokenize("no-show") == ["no-show"]                          # hyphenated terms stay whole


def test_bm25_prefers_rare_matching_words_and_matches_word_forms():
    idx = BM25Index(["a", "b", "c"], ["cab cab cab", "airport approval by HR Head", "cab booking"])
    scores = idx.scores("HR Head approval")
    assert max(scores, key=scores.get) == "b" and scores["a"] == 0.0
    assert BM25Index(["x"], ["Trips must be booked early."], stem=True).scores("booking")["x"] > 0


def test_rrf_rewards_agreement_and_respects_weights():
    assert rrf([["a", "b", "c", "d"], ["b", "d", "c", "a"]])[0] == "b"   # near the top of BOTH lists
    assert rrf([["v", "k"], ["k", "v"]], weights=[2.0, 1.0])[0] == "v"


def test_round_robin_gives_each_ranking_its_best_item():
    cancel, penalty = ["3.1", "3.2", "2.2"], ["9.2", "3.2", "9.1"]
    assert round_robin([cancel, penalty])[:3] == ["3.1", "9.2", "3.2"]
    assert rrf([cancel, penalty])[0] == "3.2"                           # RRF would favour the shared one


# ------------------------------------------------------------------------ gate --

def test_relevant_question_has_evidence(store):
    assert retrieve(store, "acme", ["Who approves an airport trip?"]).has_evidence


@pytest.mark.parametrize("company, question", [
    ("acme", "Write me a poem about cats"),
    ("crest", "Who won the cricket world cup?"),
])
def test_unrelated_question_has_no_evidence(store, company, question):
    r = retrieve(store, company, [question])
    assert r.chunks and not r.has_evidence


def test_gate_uses_the_closest_match_of_any_query(store):
    poem_only = retrieve(store, "acme", ["Write me a poem about cats"])
    with_real = retrieve(store, "acme", ["Write me a poem about cats", "escort for women on night drops"])
    assert not poem_only.has_evidence and with_real.has_evidence


def test_gate_threshold_can_be_changed(store):
    q = ["Who approves an airport trip?"]
    assert not retrieve(store, "acme", q, max_distance=0.1).has_evidence
    assert retrieve(store, "acme", q, max_distance=0.99).has_evidence
