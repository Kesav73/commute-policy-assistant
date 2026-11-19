"""Chroma vector store: one collection per company (HLD §5.3).

Isolation by construction: the collection name is built from a company_id,
and a query only ever opens that one collection. There is no shared collection
that a missing filter could leak from.
"""
import re
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.embeddings import embedding_function
from app.policy_parser import Chunk

COLLECTION_PREFIX = "policy__"
_COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class StoreError(RuntimeError):
    pass


def get_client(path: Path) -> ClientAPI:
    """Embedded Chroma: runs in-process, persists to `path`. No separate server."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def collection_name(company_id: str) -> str:
    if not _COMPANY_RE.match(company_id):
        raise StoreError(f"invalid company_id {company_id!r}")
    return f"{COLLECTION_PREFIX}{company_id}"


def rebuild_company_collection(client: ClientAPI, company_id: str, chunks: list[Chunk]) -> int:
    """Replace the company's collection with exactly these chunks. Returns the count.

    Rebuild (delete + create) rather than update, so the store always matches the
    file: a sub-section removed from the policy disappears from search too.
    """
    if not chunks:
        raise StoreError(f"no chunks for {company_id}")
    if any(c.company_id != company_id for c in chunks):
        raise StoreError(f"chunk from another company passed to {company_id}")

    name = collection_name(company_id)
    try:
        client.delete_collection(name)
    except (NotFoundError, ValueError):
        pass  # first run: nothing to delete

    collection = client.create_collection(
        name=name,
        embedding_function=embedding_function(),
        # Cosine distance: 0 = same meaning, larger = less related. Easier to reason
        # about (and to set a "not covered" threshold on) than Chroma's default L2.
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"company_id": company_id},
    )
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],       # Chroma embeds these with embedding_function()
        metadatas=[c.metadata() for c in chunks],
    )

    count = collection.count()
    if count != len(chunks):
        raise StoreError(f"{name}: stored {count} chunks, expected {len(chunks)}")
    return count


def get_company_collection(client: ClientAPI, company_id: str) -> Collection | None:
    """The ONLY way to open a collection for reading. None if the company has no policy."""
    try:
        return client.get_collection(
            name=collection_name(company_id), embedding_function=embedding_function()
        )
    except (NotFoundError, ValueError):
        return None


def chunk_counts(client: ClientAPI) -> dict[str, int]:
    """{"acme": 27, ...} for /health and the ingest summary."""
    counts = {}
    for col in client.list_collections():
        if col.name.startswith(COLLECTION_PREFIX):
            counts[col.name[len(COLLECTION_PREFIX):]] = client.get_collection(col.name).count()
    return dict(sorted(counts.items()))
