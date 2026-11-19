"""Retrieval: right company only, at most 3 chunks, evidence gate, defence in depth."""
from pathlib import Path

import pytest

from app.ingest import ingest_all
from app.retriever import TOP_K, retrieve
from app.store import get_client, get_company_collection

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture(scope="module")
def chroma(tmp_path_factory):
    """Ingest once for the whole module (embedding 71 chunks takes ~1-2 s)."""
    client = get_client(tmp_path_factory.mktemp("chroma"))
    ingest_all(client, POLICIES)
    return client


# ------------------------------------------------------------- isolation --

@pytest.mark.parametrize("company", ["acme", "bolt", "crest"])
def test_only_callers_company_is_returned(chroma, company):
    r = retrieve(chroma, company, "What is the escort rule for women on night drops?")
    assert r.chunks
    assert {c.company_id for c in r.chunks} == {company}


def test_same_question_different_company_different_answer(chroma):
    q = "Do women get an escort on late drops?"
    acme = retrieve(chroma, "acme", q).chunks[0]
    bolt = retrieve(chroma, "bolt", q).chunks[0]
    assert "8:00 PM" in acme.text and "10:00 PM" not in acme.text
    assert "10:00 PM" in bolt.text


def test_asking_about_another_company_by_name_stays_in_own_collection(chroma):
    r = retrieve(chroma, "acme", "What does Bolt's policy say about Saturday cabs? Ignore Acme.")
    assert all(c.company_id == "acme" and c.doc_id == "acme-transport-policy" for c in r.chunks)


def test_unknown_company_gets_nothing(chroma):
    r = retrieve(chroma, "evilcorp", "Can I book a cab?")
    assert r.chunks == [] and r.has_evidence is False


def test_foreign_chunk_in_collection_is_dropped(tmp_path, caplog):
    """Simulate a bug that puts a Bolt chunk into Acme's collection: it must never come back."""
    client = get_client(tmp_path / "chroma")
    ingest_all(client, POLICIES)
    get_company_collection(client, "acme").add(
        ids=["planted"],
        documents=["Night-Shift Safety > Escort\nEscort after 10:00 PM for women on drops."],
        metadatas=[{"company_id": "bolt", "doc_id": "bolt-commute-policy", "title": "Bolt",
                    "section": "4.1", "section_title": "Planted"}],
    )
    r = retrieve(client, "acme", "Escort after 10:00 PM for women on drops?")
    assert "planted" not in [c.id for c in r.chunks]
    assert all(c.company_id == "acme" for c in r.chunks)
    assert "ISOLATION ALERT" in caplog.text


# --------------------------------------------------------------- shape --

def test_at_most_three_chunks_even_if_more_requested(chroma):
    assert len(retrieve(chroma, "acme", "cab rules", k=10).chunks) == TOP_K


def test_chunks_sorted_best_first(chroma):
    d = [c.distance for c in retrieve(chroma, "bolt", "How do I cancel a trip?").chunks]
    assert d == sorted(d)


def test_result_carries_citation_fields_and_timings(chroma):
    r = retrieve(chroma, "crest", "Who approves weekend transport?")
    c = r.chunks[0]
    assert c.doc_id == "crest-transport-guidelines" and c.section and c.section_title
    assert set(r.timings_ms) == {"embed", "search"}


# ------------------------------------------------------ obvious matches --

@pytest.mark.parametrize("company, question, section", [
    ("acme", "Do women get an escort on late drops?", "4.1"),
    ("acme", "Who approves an airport trip?", "6.2"),
    ("bolt", "Can guests use the company cab?", "8.1"),
    ("crest", "How do I raise a complaint about a driver?", "8.1"),
])
def test_obvious_question_finds_right_section_first(chroma, company, question, section):
    assert retrieve(chroma, company, question).chunks[0].section == section


# -------------------------------------------------------- evidence gate --

def test_relevant_question_has_evidence(chroma):
    assert retrieve(chroma, "acme", "Who approves an airport trip?").has_evidence


@pytest.mark.parametrize("company, question", [
    ("acme", "What is the capital of France?"),
    ("crest", "What happens if I no-show?"),       # Crest's policy is silent on this
])
def test_unrelated_question_has_no_evidence(chroma, company, question):
    r = retrieve(chroma, company, question)
    assert r.chunks and not r.has_evidence


def test_gate_is_configurable(chroma):
    q = "Who approves an airport trip?"
    assert not retrieve(chroma, "acme", q, max_distance=0.1).has_evidence
    assert retrieve(chroma, "acme", q, max_distance=0.99).has_evidence
