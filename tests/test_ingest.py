"""Ingestion into a real (temporary) Chroma with the real embedding model."""
import shutil
from pathlib import Path

import numpy as np
import pytest
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from app.embeddings import embedding_function
from app.ingest import IngestError, ingest_all
from app.store import chunk_counts, get_client, get_company_collection

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@pytest.fixture
def chroma(tmp_path):
    return get_client(tmp_path / "chroma")


@pytest.fixture
def policies_copy(tmp_path) -> Path:
    d = tmp_path / "policies"
    shutil.copytree(POLICIES, d)
    return d


def test_ingest_creates_one_collection_per_company(chroma):
    assert ingest_all(chroma, POLICIES) == {"acme": 27, "bolt": 25, "crest": 19}
    assert chunk_counts(chroma) == {"acme": 27, "bolt": 25, "crest": 19}


def test_each_collection_holds_only_its_own_company(chroma):
    ingest_all(chroma, POLICIES)
    for company in ("acme", "bolt", "crest"):
        metas = get_company_collection(chroma, company).get()["metadatas"]
        assert {m["company_id"] for m in metas} == {company}


def test_collection_uses_cosine_and_stores_citation_metadata(chroma):
    ingest_all(chroma, POLICIES)
    acme = get_company_collection(chroma, "acme")
    assert acme.configuration_json["hnsw"]["space"] == "cosine"
    got = acme.get(ids=["acme-transport-policy:4.1"], include=["metadatas", "embeddings"])
    assert got["metadatas"][0]["section_title"] == "Night-Shift Safety > Escort Requirement"
    assert len(got["embeddings"][0]) == 384


def test_ingest_twice_does_not_duplicate(chroma):
    ingest_all(chroma, POLICIES)
    assert ingest_all(chroma, POLICIES) == {"acme": 27, "bolt": 25, "crest": 19}


def test_removed_section_disappears_after_reingest(chroma, policies_copy):
    ingest_all(chroma, policies_copy)
    f = policies_copy / "crest.md"
    f.write_text(f.read_text().replace(
        "### 8.2 Response Time\nThe transport desk acknowledges", "The transport desk acknowledges"))
    ingest_all(chroma, policies_copy)
    assert get_company_collection(chroma, "crest").get(ids=["crest-transport-guidelines:8.2"])["ids"] == []


def test_bad_file_stops_ingest_before_anything_is_written(chroma, policies_copy):
    ingest_all(chroma, policies_copy)
    (policies_copy / "bolt.md").write_text("no front matter here")
    with pytest.raises(IngestError, match="bolt.md"):
        ingest_all(chroma, policies_copy)
    assert chunk_counts(chroma) == {"acme": 27, "bolt": 25, "crest": 19}  # untouched


def test_unknown_company_is_rejected(policies_copy, chroma):
    f = policies_copy / "crest.md"
    f.write_text(f.read_text().replace("company_id: crest", "company_id: evilcorp"))
    with pytest.raises(IngestError, match="not in the employee directory"):
        ingest_all(chroma, policies_copy)


def test_two_files_for_one_company_are_rejected(policies_copy, chroma):
    shutil.copy(policies_copy / "acme.md", policies_copy / "acme_copy.md")
    with pytest.raises(IngestError, match="already has a policy"):
        ingest_all(chroma, policies_copy)


def test_unknown_company_has_no_collection(chroma):
    ingest_all(chroma, POLICIES)
    assert get_company_collection(chroma, "evilcorp") is None


def test_fast_embedding_matches_chroma_default():
    # Our padding fix must not change the vectors, only the speed.
    texts = ["Can I book a cab on Saturday?", "A much longer piece of text " * 10]
    ours, theirs = np.array(embedding_function()(texts)), np.array(DefaultEmbeddingFunction()(texts))
    assert np.abs(ours - theirs).max() < 1e-5
