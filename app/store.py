"""Storage: a chunk store for content + a Chroma vector index, one collection per company.

    data/chunks/<company>.jsonl   chunk id -> text + metadata   (app/chunk_store.py)
    data/chroma/policy__<company> chunk id -> vector, nothing else

Isolation by construction: the collection name is built from a company_id, and a
query only ever opens that one collection. There is no shared collection that a
missing filter could leak from. Retrieved ids are then looked up in that same
company's chunk store, so an id that doesn't belong to the company has no content.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.chunk_store import ChunkStore
from app.embeddings import EMBEDDING_MODEL, embedding_function
from app.policy_parser import Chunk

COLLECTION_PREFIX = "policy__"
_COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class StoreError(RuntimeError):
    pass


@dataclass
class PolicyStore:
    vectors: ClientAPI   # Chroma: ids + vectors
    chunks: ChunkStore   # ids -> content


def open_store(root: Path) -> PolicyStore:
    """Open (or create) both stores under `root`. Embedded Chroma: in-process, no server."""
    (root / "chroma").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(root / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return PolicyStore(vectors=client, chunks=ChunkStore(root / "chunks"))


def collection_name(company_id: str) -> str:
    if not _COMPANY_RE.match(company_id):
        raise StoreError(f"invalid company_id {company_id!r}")
    return f"{COLLECTION_PREFIX}{company_id}"


def index_company(store: PolicyStore, company_id: str, chunks: list[Chunk]) -> int:
    """Replace the company's chunks and vectors with exactly these. Returns the count.

    Rebuild (delete + create) rather than update, so the store always matches the
    file: a sub-section removed from the policy disappears from search too.
    """
    if not chunks:
        raise StoreError(f"no chunks for {company_id}")
    if any(c.company_id != company_id for c in chunks):
        raise StoreError(f"chunk from another company passed to {company_id}")

    vectors = embedding_function()([c.text for c in chunks])   # embed BEFORE touching anything
    store.chunks.put(company_id, chunks)

    name = collection_name(company_id)
    try:
        store.vectors.delete_collection(name)
    except (NotFoundError, ValueError):
        pass  # first run: nothing to delete

    collection = store.vectors.create_collection(
        name=name,
        embedding_function=None,   # we always pass vectors ourselves
        # Cosine distance: 0 = same meaning, larger = less related. Easier to reason
        # about (and to set a "not covered" threshold on) than Chroma's default L2.
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"company_id": company_id, "embedding_model": EMBEDDING_MODEL},
    )
    collection.add(ids=[c.id for c in chunks], embeddings=vectors)

    # Both stores must hold exactly the same ids, or retrieval would return
    # vectors with no content (or content that can never be found).
    indexed = set(collection.get(include=[])["ids"])
    if indexed != set(store.chunks.all(company_id)):
        raise StoreError(f"{name}: vector ids and chunk store ids differ")
    return len(indexed)


def get_company_collection(store: PolicyStore, company_id: str) -> Collection | None:
    """The ONLY way to open a collection for reading. None if the company has no policy."""
    try:
        return store.vectors.get_collection(name=collection_name(company_id), embedding_function=None)
    except (NotFoundError, ValueError):
        return None


def chunk_counts(store: PolicyStore) -> dict[str, int]:
    """{"acme": 27, ...} for /health and the ingest summary."""
    counts = {}
    for col in store.vectors.list_collections():
        if col.name.startswith(COLLECTION_PREFIX):
            counts[col.name[len(COLLECTION_PREFIX):]] = store.vectors.get_collection(col.name).count()
    return dict(sorted(counts.items()))
