"""Storage: chunk store (content) + Chroma (vectors) + the ACTIVE-version pointer.

    data/active.json                      {"acme": {"version": 2, ...}, ...}   <- the switch
    data/chunks/<company>/v<n>.jsonl      chunk id -> text + metadata          (app/chunk_store.py)
    data/chroma/  policy__<company>__v<n> chunk id -> vector, nothing else

Versions are never edited in place. A new version is built under new names next to
the live one; only when it is complete and verified does `set_active` rewrite
active.json - one atomic file rename (os.replace) - and from the next request on
only the new version answers. Anything not named in active.json is never served.

Isolation by construction: collection names are built from a company_id, a query
only opens that one company's collection, and retrieved ids are looked up in the
same company's chunk file.
"""
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.chunk_store import ChunkStore
from app.embeddings import EMBEDDING_ID, embed_documents
from app.policy_parser import Chunk

COLLECTION_PREFIX = "policy__"
_COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COLLECTION_RE = re.compile(r"^policy__([a-z0-9][a-z0-9_-]*?)__v(\d+)$")


class StoreError(RuntimeError):
    pass


@dataclass
class PolicyStore:
    root: Path
    vectors: ClientAPI                    # Chroma: ids + vectors
    chunks: ChunkStore = field(init=False)
    _active: dict = field(default_factory=dict)        # company -> info (mirror of active.json)
    _write_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.chunks = ChunkStore(self.root / "chunks", self.active_version)
        path = self.root / "active.json"
        if path.exists():
            self._active = json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------ active pointer --
    def active_version(self, company_id: str) -> int | None:
        info = self._active.get(company_id)
        return info["version"] if info else None

    def active_info(self, company_id: str) -> dict | None:
        return self._active.get(company_id)

    def active_companies(self) -> list[str]:
        return sorted(self._active)

    def set_active(self, company_id: str, info: dict) -> None:
        """THE switch: rewrite active.json via temp file + os.replace (atomic on the
        same filesystem). A reader sees the old file or the new one, never a mix."""
        with self._write_lock:
            new = {**self._active, company_id: info}
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".active.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(new, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.root / "active.json")
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            self._active = new   # in-memory pointer swap: one assignment


def open_store(root: Path) -> PolicyStore:
    """Open (or create) the stores under `root`. Embedded Chroma: in-process, no server."""
    (root / "chroma").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(root / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return PolicyStore(root=root, vectors=client)


def collection_name(company_id: str, version: int) -> str:
    if not _COMPANY_RE.match(company_id):
        raise StoreError(f"invalid company_id {company_id!r}")
    return f"{COLLECTION_PREFIX}{company_id}__v{int(version)}"


# ------------------------------------------------------------- build / delete --

def build_version(store: PolicyStore, company_id: str, version: int, chunks: list[Chunk]) -> int:
    """Build version `version` of a company's policy NEXT TO the live one: embed,
    write its chunk file, create its own collection, then verify. Doesn't touch the
    active pointer or any other version. Returns the chunk count; raises on any
    problem (the caller deletes the partial build)."""
    if not chunks:
        raise StoreError(f"no chunks for {company_id}")
    if any(c.company_id != company_id for c in chunks):
        raise StoreError(f"chunk from another company passed to {company_id}")
    if version == store.active_version(company_id):
        raise StoreError(f"{company_id} v{version} is the live version; build a new number")

    vectors = embed_documents([c.text for c in chunks])   # embed BEFORE writing anything
    store.chunks.put(company_id, version, chunks)

    name = collection_name(company_id, version)
    _drop_collection(store, name)   # a leftover from an earlier crashed build
    collection = store.vectors.create_collection(
        name=name,
        embedding_function=None,   # we always pass vectors ourselves
        # Cosine distance: 0 = same meaning, larger = less related.
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"company_id": company_id, "version": version, "embedding": EMBEDDING_ID},
    )
    collection.add(ids=[c.id for c in chunks], embeddings=vectors)
    verify_version(store, company_id, version, expected=len(chunks))
    return len(chunks)


def verify_version(store: PolicyStore, company_id: str, version: int, expected: int | None = None) -> None:
    """Both stores hold exactly the same ids, the expected number of them, vectors
    made by the current embedding, and a search returns a result."""
    col = get_company_collection(store, company_id, version)
    if col is None:
        raise StoreError(f"{company_id} v{version}: collection missing")
    if (col.metadata or {}).get("embedding") != EMBEDDING_ID:
        raise StoreError(f"{company_id} v{version}: built with a different embedding")
    stored = col.get(include=["embeddings"])
    ids = set(stored["ids"])
    if ids != set(store.chunks.all(company_id, version)):
        raise StoreError(f"{company_id} v{version}: vector ids and chunk store ids differ")
    if expected is not None and len(ids) != expected:
        raise StoreError(f"{company_id} v{version}: {len(ids)} chunks indexed, expected {expected}")
    probe = col.query(query_embeddings=[stored["embeddings"][0]], n_results=1, include=[])
    if not probe["ids"][0]:
        raise StoreError(f"{company_id} v{version}: test search returned nothing")


def delete_version(store: PolicyStore, company_id: str, version: int) -> None:
    """Remove a version's vectors and chunk file. Refuses the live version."""
    if version == store.active_version(company_id):
        raise StoreError(f"refusing to delete the live version {company_id} v{version}")
    _drop_collection(store, collection_name(company_id, version))
    store.chunks.delete(company_id, version)


def delete_later(store: PolicyStore, company_id: str, version: int, after_s: float) -> None:
    """Delete an old version after a grace period: a request that pinned it just
    before the switch (<= REQUEST_TIMEOUT_S) can still finish. It is never chosen
    by new requests, because the pointer no longer names it."""
    def _run():
        time.sleep(after_s)
        try:
            if version != store.active_version(company_id):
                delete_version(store, company_id, version)
        except Exception:   # noqa: BLE001 - cleanup is best effort; startup cleans up too
            pass
    threading.Thread(target=_run, name=f"delete-{company_id}-v{version}", daemon=True).start()


def _drop_collection(store: PolicyStore, name: str) -> None:
    try:
        store.vectors.delete_collection(name)
    except (NotFoundError, ValueError):
        pass


def vector_versions(store: PolicyStore) -> dict[str, list[int]]:
    """{company: [versions with a collection]} - used to find leftovers at startup."""
    out: dict[str, list[int]] = {}
    for col in store.vectors.list_collections():
        if m := _COLLECTION_RE.match(col.name):
            out.setdefault(m.group(1), []).append(int(m.group(2)))
    return {k: sorted(v) for k, v in out.items()}


def legacy_collections(store: PolicyStore) -> list[str]:
    """Collections from before versioning ("policy__acme"): never served, removed at startup."""
    return [c.name for c in store.vectors.list_collections()
            if c.name.startswith(COLLECTION_PREFIX) and not _COLLECTION_RE.match(c.name)]


# -------------------------------------------------------------------- read --

def get_company_collection(store: PolicyStore, company_id: str,
                           version: int | None = None) -> Collection | None:
    """The ONLY way to open a collection for reading: the company's given (default:
    ACTIVE) version. None if the company has no policy."""
    v = store.active_version(company_id) if version is None else version
    if v is None:
        return None
    try:
        return store.vectors.get_collection(name=collection_name(company_id, v), embedding_function=None)
    except (NotFoundError, ValueError):
        return None


def chunk_counts(store: PolicyStore) -> dict[str, int]:
    """{"acme": 27, ...}: chunks in each company's ACTIVE version (for /health)."""
    return {co: store.active_info(co)["chunks"] for co in store.active_companies()}


def active_versions(store: PolicyStore) -> dict[str, int]:
    return {co: store.active_version(co) for co in store.active_companies()}
