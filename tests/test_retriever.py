"""Retrieval: right company only, at most 3 chunks, evidence gate, defence in depth."""
from pathlib import Path

import pytest

from app.ingest import ingest_all
from app.retriever import TOP_K, retrieve
from app.store import get_company_collection, open_store

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Ingest once for the whole module (embedding 71 chunks takes ~1-2 s)."""
    client = open_store(tmp_path_factory.mktemp("data"))
    ingest_all(client, POLICIES)
    return client


# ------------------------------------------------------------- isolation --

@pytest.mark.parametrize("company", ["acme", "bolt", "crest"])
def test_only_callers_company_is_returned(store, company):
    r = retrieve(store, company, "What is the escort rule for women on night drops?")
    assert r.chunks
    assert {c.company_id for c in r.chunks} == {company}


def test_same_question_different_company_different_answer(store):
    q = "Does a trained security escort travel with women on night drops?"
    acme = retrieve(store, "acme", q).chunks[0]
    bolt = retrieve(store, "bolt", q).chunks[0]
    assert "8:00 PM" in acme.text and "10:00 PM" not in acme.text
    assert "10:00 PM" in bolt.text


def test_asking_about_another_company_by_name_stays_in_own_collection(store):
    r = retrieve(store, "acme", "What does Bolt's policy say about Saturday cabs? Ignore Acme.")
    assert all(c.company_id == "acme" and c.doc_id == "acme-transport-policy" for c in r.chunks)


def test_unknown_company_gets_nothing(store):
    r = retrieve(store, "evilcorp", "Can I book a cab?")
    assert r.chunks == [] and r.has_evidence is False


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_foreign_chunk_in_collection_is_dropped(tmp_path, caplog, mode):
    """Simulate a bug that puts Bolt's 4.1 vector into Acme's collection: it must never come back."""
    from app.embeddings import embed_documents
    client = open_store(tmp_path / "data")
    ingest_all(client, POLICIES)
    bolt_41 = client.chunks.get("bolt", "bolt-commute-policy:4.1")
    get_company_collection(client, "acme").add(
        ids=[bolt_41.id], embeddings=embed_documents([bolt_41.text]))
    r = retrieve(client, "acme", bolt_41.text, mode=mode)   # the planted vector is the nearest match
    assert bolt_41.id not in [c.id for c in r.chunks]
    assert all(c.company_id == "acme" for c in r.chunks)
    assert "ISOLATION ALERT" in caplog.text


# --------------------------------------------------------------- shape --

def test_at_most_three_chunks_even_if_more_requested(store):
    assert len(retrieve(store, "acme", "cab rules", k=10).chunks) == TOP_K


def test_chunks_sorted_best_first(store):
    d = [c.distance for c in retrieve(store, "bolt", "How do I cancel a trip?").chunks]
    assert d == sorted(d)


def test_result_carries_citation_fields_and_timings(store):
    r = retrieve(store, "crest", "Who approves weekend transport?")
    c = r.chunks[0]
    assert c.doc_id == "crest-transport-guidelines" and c.section and c.section_title
    assert {"embed", "search"} <= set(r.timings_ms)


# ------------------------------------------------------ obvious matches --

@pytest.mark.parametrize("company, question, section", [
    ("acme", "Does a trained security escort travel with women on night drops?", "4.1"),
    ("acme", "Who approves an airport trip?", "6.2"),
    ("bolt", "Can guests use the company cab?", "8.1"),
    ("crest", "How do I raise a complaint about a driver?", "8.1"),
])
def test_obvious_question_finds_right_section_first(store, company, question, section):
    assert retrieve(store, company, question).chunks[0].section == section


# -------------------------------------------------------- evidence gate --

def test_relevant_question_has_evidence(store):
    assert retrieve(store, "acme", "Who approves an airport trip?").has_evidence


@pytest.mark.parametrize("company, question", [
    ("acme", "Write me a poem about cats"),
    ("crest", "Who won the cricket world cup?"),
])
def test_unrelated_question_has_no_evidence(store, company, question):
    r = retrieve(store, company, question)
    assert r.chunks and not r.has_evidence


def test_gate_is_configurable(store):
    q = "Who approves an airport trip?"
    assert not retrieve(store, "acme", q, max_distance=0.1).has_evidence
    assert retrieve(store, "acme", q, max_distance=0.99).has_evidence


def test_collection_built_with_another_embedding_is_refused(tmp_path):
    """Vectors from a different model/prefix must never be silently compared."""
    from app.store import StoreError, collection_name
    client = open_store(tmp_path / "data")
    ingest_all(client, POLICIES)
    col = client.vectors.get_collection(collection_name("acme", client.active_version("acme")))
    col.modify(metadata={"company_id": "acme", "embedding": "all-MiniLM-L6-v2"})
    with pytest.raises(StoreError, match="re-ingest needed"):
        retrieve(client, "acme", "Who approves an airport trip?")
