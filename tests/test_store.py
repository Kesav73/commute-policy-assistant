"""Storage and embeddings: content in chunk files, ids + vectors in Chroma, one company per collection."""
import shutil

import numpy as np
import pytest

from app.embeddings import DOCUMENT_PREFIX, EMBEDDING_ID, _embed, embed_documents, embed_query
from app.store import PolicyStore, collection_name
from app.versioning import PolicyRejected, ingest_all
from tests.conftest import POLICIES


@pytest.fixture
def policies_copy(tmp_path):
    folder = tmp_path / "policies"
    shutil.copytree(POLICIES, folder)
    return folder


def test_publishing_creates_one_collection_per_company(store):
    assert store.summary() == {"versions": {"acme": 1, "bolt": 1, "crest": 1},
                               "chunks": {"acme": 27, "bolt": 25, "crest": 18}}   # Crest 7 is quarantined
    assert store.collection("acme").name == collection_name("acme", 1) == "policy__acme__v1"


def test_each_company_holds_only_its_own_chunks(store):
    for company in ("acme", "bolt", "crest"):
        chunks = store.chunks(company)
        assert {c.company_id for c in chunks.values()} == {company}
        assert set(store.collection(company).get(include=[])["ids"]) == set(chunks)   # same ids in both stores


def test_chroma_holds_only_ids_and_vectors(store):
    acme = store.collection("acme")
    assert acme.configuration_json["hnsw"]["space"] == "cosine"
    assert acme.metadata == {"company_id": "acme", "version": 1, "embedding": EMBEDDING_ID}
    got = acme.get(ids=["acme-transport-policy:4.1"], include=["documents", "metadatas", "embeddings"])
    assert got["documents"] == [None] and got["metadatas"] == [None]       # no content in Chroma
    assert len(got["embeddings"][0]) == 768


def test_chunk_files_hold_content_and_citation_fields(store):
    c = store.chunks("acme")["acme-transport-policy:4.1"]
    assert c.section == "4.1" and c.section_title == "Night-Shift Safety > Escort Requirement"
    assert c.doc_id == "acme-transport-policy" and "8:00 PM" in c.text
    assert (store.chunks_dir / "acme" / "v1.jsonl").exists()
    assert "bolt-commute-policy:4.1" not in store.chunks("acme")            # not Acme's


def test_unknown_company_has_nothing(store):
    assert store.collection("evilcorp") is None and store.chunks("evilcorp") == {}


def test_bad_file_stops_publishing_before_anything_is_written(tmp_path, policies_copy):
    (policies_copy / "bolt.md").write_text("no front matter here")
    s = PolicyStore(tmp_path / "data")
    with pytest.raises(PolicyRejected, match="bolt.md"):
        ingest_all(s, policies_copy)
    assert s.active_companies() == []                                        # nothing written


def test_unknown_company_or_two_files_for_one_company_are_rejected(tmp_path, policies_copy):
    shutil.copy(policies_copy / "acme.md", policies_copy / "acme_copy.md")
    with pytest.raises(PolicyRejected, match="already has a policy"):
        ingest_all(PolicyStore(tmp_path / "a"), policies_copy)
    (policies_copy / "acme_copy.md").unlink()
    f = policies_copy / "crest.md"
    f.write_text(f.read_text().replace("company_id: crest", "company_id: evilcorp"))
    with pytest.raises(PolicyRejected, match="not in the employee directory"):
        ingest_all(PolicyStore(tmp_path / "b"), policies_copy)


# ---------------------------------------------------------------- embeddings --

def test_embeddings_are_unit_length_and_768_numbers():
    v = embed_documents(["Escort after 8 PM.", "A much longer piece of text " * 20])
    assert v.shape == (2, 768) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_question_and_chunk_prefixes_give_different_vectors():
    text = "Who approves an airport trip?"
    assert not np.allclose(embed_query(text)[0], embed_documents([text])[0], atol=1e-3)


def test_embedding_id_names_model_and_both_prefixes():
    assert EMBEDDING_ID == "nomic-ai/nomic-embed-text-v1.5|search_document:|search_query:"


def test_cached_chunk_vectors_are_the_same_as_fresh_ones():
    texts = ["Cache check one.", "Cache check two.", "Cache check one."]
    first = embed_documents(texts)
    assert np.allclose(first, embed_documents(texts))
    assert np.allclose(first[1], _embed([DOCUMENT_PREFIX + texts[1]])[0], atol=1e-5)
