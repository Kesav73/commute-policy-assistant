"""Ingestion into a real (temporary) chunk store + Chroma with the real embedding model."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.embeddings import embedding_function
from app.ingest import IngestError, ingest_all
from app.store import chunk_counts, get_company_collection, open_store

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture
def store(tmp_path):
    return open_store(tmp_path / "data")


@pytest.fixture
def policies_copy(tmp_path) -> Path:
    d = tmp_path / "policies"
    shutil.copytree(POLICIES, d)
    return d


def test_ingest_creates_one_collection_per_company(store):
    assert ingest_all(store, POLICIES) == {"acme": 27, "bolt": 25, "crest": 19}
    assert chunk_counts(store) == {"acme": 27, "bolt": 25, "crest": 19}


def test_each_company_holds_only_its_own_chunks(store):
    ingest_all(store, POLICIES)
    for company in ("acme", "bolt", "crest"):
        chunks = store.chunks.all(company)
        assert {c.company_id for c in chunks.values()} == {company}
        # the vector index and the chunk store hold exactly the same ids
        assert set(get_company_collection(store, company).get(include=[])["ids"]) == set(chunks)


def test_vector_index_holds_only_ids_and_vectors(store):
    ingest_all(store, POLICIES)
    acme = get_company_collection(store, "acme")
    assert acme.configuration_json["hnsw"]["space"] == "cosine"
    assert acme.metadata == {"company_id": "acme", "embedding_model": "all-MiniLM-L6-v2"}
    got = acme.get(ids=["acme-transport-policy:4.1"], include=["documents", "metadatas", "embeddings"])
    assert got["documents"] == [None] and got["metadatas"] == [None]   # no content in Chroma
    assert len(got["embeddings"][0]) == 384


def test_chunk_store_holds_content_and_citation_metadata(store, tmp_path):
    ingest_all(store, POLICIES)
    c = store.chunks.get("acme", "acme-transport-policy:4.1")
    assert c.section == "4.1" and c.section_title == "Night-Shift Safety > Escort Requirement"
    assert c.doc_id == "acme-transport-policy" and "8:00 PM" in c.text
    assert (tmp_path / "data" / "chunks" / "acme.jsonl").exists()
    assert store.chunks.get("acme", "bolt-commute-policy:4.1") is None   # not Acme's


def test_ingest_twice_does_not_duplicate(store):
    ingest_all(store, POLICIES)
    assert ingest_all(store, POLICIES) == {"acme": 27, "bolt": 25, "crest": 19}


def test_removed_section_disappears_after_reingest(store, policies_copy):
    ingest_all(store, policies_copy)
    f = policies_copy / "crest.md"
    f.write_text(f.read_text().replace(
        "### 8.2 Response Time\nThe transport desk acknowledges", "The transport desk acknowledges"))
    ingest_all(store, policies_copy)
    assert get_company_collection(store, "crest").get(ids=["crest-transport-guidelines:8.2"])["ids"] == []
    assert store.chunks.get("crest", "crest-transport-guidelines:8.2") is None


def test_bad_file_stops_ingest_before_anything_is_written(store, policies_copy):
    ingest_all(store, policies_copy)
    (policies_copy / "bolt.md").write_text("no front matter here")
    with pytest.raises(IngestError, match="bolt.md"):
        ingest_all(store, policies_copy)
    assert chunk_counts(store) == {"acme": 27, "bolt": 25, "crest": 19}  # untouched


def test_unknown_company_is_rejected(policies_copy, store):
    f = policies_copy / "crest.md"
    f.write_text(f.read_text().replace("company_id: crest", "company_id: evilcorp"))
    with pytest.raises(IngestError, match="not in the employee directory"):
        ingest_all(store, policies_copy)


def test_two_files_for_one_company_are_rejected(policies_copy, store):
    shutil.copy(policies_copy / "acme.md", policies_copy / "acme_copy.md")
    with pytest.raises(IngestError, match="already has a policy"):
        ingest_all(store, policies_copy)


def test_unknown_company_has_no_collection(store):
    ingest_all(store, POLICIES)
    assert get_company_collection(store, "evilcorp") is None


def test_fast_embedding_matches_chroma_default():
    # Our padding fix must not change the vectors, only the speed.
    texts = ["Can I book a cab on Saturday?", "A much longer piece of text " * 10]
    ours, theirs = np.array(embedding_function()(texts)), np.array(DefaultEmbeddingFunction()(texts))
    assert np.abs(ours - theirs).max() < 1e-5
