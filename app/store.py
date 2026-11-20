"""All storage, in one class: chunk text, vectors, and which policy version is live.

    data/active.json                       {"acme": {"version": 2, ...}, ...}   <- the switch
    data/chunks/<company>/v<n>.jsonl       chunk text + metadata (the source of truth)
    data/chroma/  policy__<company>__v<n>  chunk ids + vectors only (one collection per company + version)

* Content and vectors are kept apart: the LLM and the citations always read text from
  the chunk files; Chroma only finds ids.
* A new policy version is written under NEW names next to the live one, and only
  becomes live when active.json is switched (one atomic file rename). Anything not
  named in active.json is never used.
* Isolation by construction: every lookup is for ONE company's files and collection.
"""
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from app.embeddings import EMBEDDING_ID, embed_documents
from app.policy_parser import Chunk

COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
COLLECTION_RE = re.compile(r"^policy__([a-z0-9][a-z0-9_-]*?)__v(\d+)$")


class StoreError(RuntimeError):
    pass


def collection_name(company_id: str, version: int) -> str:
    if not COMPANY_RE.match(company_id):
        raise StoreError(f"invalid company_id {company_id!r}")
    return f"policy__{company_id}__v{int(version)}"


def _write_atomically(path: Path, text: str) -> None:
    """Write to a temp file, then rename it over `path`: a reader sees the old
    file or the new one, never half a file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class PolicyStore:
    def __init__(self, root: Path):
        self.root = root
        self.chunks_dir = root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        (root / "chroma").mkdir(parents=True, exist_ok=True)
        self.vectors = chromadb.PersistentClient(          # embedded: runs in-process, no server
            path=str(root / "chroma"), settings=ChromaSettings(anonymized_telemetry=False))

        self._active_file = root / "active.json"
        self._active = {}                                   # in-memory copy of active.json
        if self._active_file.exists():
            self._active = json.loads(self._active_file.read_text(encoding="utf-8"))
        self._chunk_cache = {}                              # (company, version) -> {id: Chunk}
        self._write_lock = threading.Lock()

    # ------------------------------------------------------------ the switch --
    def active_version(self, company_id: str) -> int | None:
        info = self._active.get(company_id)
        return info["version"] if info else None

    def active_info(self, company_id: str) -> dict | None:
        return self._active.get(company_id)

    def active_companies(self) -> list[str]:
        return sorted(self._active)

    def set_active(self, company_id: str, info: dict) -> None:
        """Make a version live: rewrite active.json atomically, then swap the in-memory
        copy in one assignment. A reader sees the old state or the new one, never a mix."""
        with self._write_lock:
            new_active = dict(self._active)
            new_active[company_id] = info
            _write_atomically(self._active_file, json.dumps(new_active, indent=2, sort_keys=True))
            self._active = new_active

    # ---------------------------------------------------------------- chunks --
    def _chunk_file(self, company_id: str, version: int) -> Path:
        if not COMPANY_RE.match(company_id):
            raise StoreError(f"invalid company_id {company_id!r}")
        return self.chunks_dir / company_id / f"v{int(version)}.jsonl"

    def chunks(self, company_id: str, version: int | None = None) -> dict[str, Chunk]:
        """{chunk id: Chunk} for one company's version (default: the live one).
        Empty if there is none. Cached in memory after the first read."""
        if version is None:
            version = self.active_version(company_id)
        if version is None:
            return {}
        key = (company_id, version)
        if key not in self._chunk_cache:
            path = self._chunk_file(company_id, version)
            if not path.exists():
                return {}
            loaded = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    chunk = Chunk(**json.loads(line))
                    loaded[chunk.id] = chunk
            self._chunk_cache[key] = loaded
        return self._chunk_cache[key]

    def _write_chunks(self, company_id: str, version: int, chunks: list[Chunk]) -> None:
        path = self._chunk_file(company_id, version)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(c), ensure_ascii=False) for c in chunks]
        _write_atomically(path, "\n".join(lines) + "\n")
        self._chunk_cache[(company_id, version)] = {c.id: c for c in chunks}

    # --------------------------------------------------------------- vectors --
    def collection(self, company_id: str, version: int | None = None):
        """One company's Chroma collection for a version (default: the live one), or None."""
        if version is None:
            version = self.active_version(company_id)
        if version is None:
            return None
        try:
            return self.vectors.get_collection(name=collection_name(company_id, version), embedding_function=None)
        except (NotFoundError, ValueError):
            return None

    def _drop_collection(self, name: str) -> None:
        try:
            self.vectors.delete_collection(name)
        except (NotFoundError, ValueError):
            pass

    # -------------------------------------------------------------- versions --
    def build_version(self, company_id: str, version: int, chunks: list[Chunk]) -> int:
        """Build a version NEXT TO the live one (its own chunk file and collection),
        then verify it. Doesn't touch the live version. Raises on any problem; the
        caller then deletes the partial build."""
        if not chunks:
            raise StoreError(f"no chunks for {company_id}")
        for c in chunks:
            if c.company_id != company_id:
                raise StoreError(f"chunk from another company passed to {company_id}")
        if version == self.active_version(company_id):
            raise StoreError(f"{company_id} v{version} is live; a new version needs a new number")

        vectors = embed_documents([c.text for c in chunks])   # embed first: the slow, risky step
        self._write_chunks(company_id, version, chunks)
        name = collection_name(company_id, version)
        self._drop_collection(name)                            # leftover of an earlier crashed build
        collection = self.vectors.create_collection(
            name=name,
            embedding_function=None,                           # we always pass vectors ourselves
            configuration={"hnsw": {"space": "cosine"}},       # cosine distance: 0 = same meaning
            metadata={"company_id": company_id, "version": version, "embedding": EMBEDDING_ID},
        )
        collection.add(ids=[c.id for c in chunks], embeddings=vectors)
        self.verify_version(company_id, version, expected_chunks=len(chunks))
        return len(chunks)

    def verify_version(self, company_id: str, version: int, expected_chunks: int | None = None) -> None:
        """Raise StoreError unless the version is complete: both stores hold the same ids,
        the expected number of them, made by the current embedding model, and a search works."""
        collection = self.collection(company_id, version)
        if collection is None:
            raise StoreError(f"{company_id} v{version}: collection missing")
        if (collection.metadata or {}).get("embedding") != EMBEDDING_ID:
            raise StoreError(f"{company_id} v{version}: built with a different embedding model")
        stored = collection.get(include=["embeddings"])
        ids = set(stored["ids"])
        if ids != set(self.chunks(company_id, version)):
            raise StoreError(f"{company_id} v{version}: vector ids and chunk ids differ")
        if expected_chunks is not None and len(ids) != expected_chunks:
            raise StoreError(f"{company_id} v{version}: {len(ids)} chunks indexed, expected {expected_chunks}")
        probe = collection.query(query_embeddings=[stored["embeddings"][0]], n_results=1, include=[])
        if not probe["ids"][0]:
            raise StoreError(f"{company_id} v{version}: test search returned nothing")

    def delete_version(self, company_id: str, version: int) -> None:
        """Remove a version's vectors and chunk file. Never the live one."""
        if version == self.active_version(company_id):
            raise StoreError(f"refusing to delete the live version {company_id} v{version}")
        self._drop_collection(collection_name(company_id, version))
        self._chunk_cache.pop((company_id, version), None)
        self._chunk_file(company_id, version).unlink(missing_ok=True)

    def delete_later(self, company_id: str, version: int, after_s: float) -> None:
        """Delete an old version after a grace period, in the background, so a request
        that started on it just before the switch can still finish."""
        def run():
            time.sleep(after_s)
            try:
                if version != self.active_version(company_id):
                    self.delete_version(company_id, version)
            except Exception:   # noqa: BLE001 - best effort; startup cleans up leftovers too
                pass
        threading.Thread(target=run, daemon=True).start()

    def versions_on_disk(self, company_id: str) -> set[int]:
        """Every version of a company that has a chunk file or a collection."""
        found = set()
        folder = self.chunks_dir / company_id
        if folder.exists():
            for path in folder.glob("v*.jsonl"):
                if path.stem[1:].isdigit():
                    found.add(int(path.stem[1:]))
        for col in self.vectors.list_collections():
            match = COLLECTION_RE.match(col.name)
            if match and match.group(1) == company_id:
                found.add(int(match.group(2)))
        return found

    def companies_on_disk(self) -> set[str]:
        found = {p.name for p in self.chunks_dir.iterdir() if p.is_dir() and COMPANY_RE.match(p.name)}
        for col in self.vectors.list_collections():
            match = COLLECTION_RE.match(col.name)
            if match:
                found.add(match.group(1))
        return found

    # ---------------------------------------------------------------- health --
    def summary(self) -> dict:
        """{"versions": {"acme": 2, ...}, "chunks": {"acme": 27, ...}} for GET /health."""
        companies = self.active_companies()
        return {"versions": {c: self.active_version(c) for c in companies},
                "chunks": {c: self._active[c]["chunks"] for c in companies}}
