"""Chunk store: the single source of truth for chunk CONTENT (text + metadata).

    data/chunks/<company_id>/v<version>.jsonl   one line per chunk, in policy order

Each policy version has its own file. The vector index (Chroma, app/store.py) holds
only chunk ids and vectors, one collection per company AND version. Everything a
user or the LLM sees - excerpt text, document title, section for the citation - is
read from here by id. So:
  * a new version is built in a NEW file next to the live one, never edited in place
    (the live version keeps answering until the switch);
  * the embedding model can change without touching the content;
  * the company check after retrieval uses this file, which the vector index cannot
    influence.

Writes are atomic: a temp file is written and renamed over the target, so a reader
never sees half a file.

`version=None` means "the company's ACTIVE version" (resolved through the callback
given by the store). A request pins one version at its start and passes it
explicitly, so a version switch mid-request can't mix two versions.
"""
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from app.policy_parser import Chunk

_COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ChunkStoreError(RuntimeError):
    pass


class ChunkStore:
    def __init__(self, root: Path, active_version: Callable[[str], int | None]) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._active = active_version
        self._cache: dict[tuple[str, int], dict[str, Chunk]] = {}  # (company, v) -> {id -> Chunk}

    # ---------------------------------------------------------------- paths --
    def _dir(self, company_id: str) -> Path:
        if not _COMPANY_RE.match(company_id):
            raise ChunkStoreError(f"invalid company_id {company_id!r}")
        return self.root / company_id

    def _path(self, company_id: str, version: int) -> Path:
        return self._dir(company_id) / f"v{int(version)}.jsonl"

    def _resolve(self, company_id: str, version: int | None) -> int | None:
        return self._active(company_id) if version is None else version

    # ---------------------------------------------------------------- write --
    def put(self, company_id: str, version: int, chunks: list[Chunk]) -> None:
        """Write this version's chunks (atomic rename). Never touches other versions."""
        if any(c.company_id != company_id for c in chunks):
            raise ChunkStoreError(f"chunk from another company passed to {company_id}")
        if len({c.id for c in chunks}) != len(chunks):
            raise ChunkStoreError(f"duplicate chunk ids for {company_id}")
        d = self._dir(company_id)
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=f".v{version}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            os.replace(tmp, self._path(company_id, version))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._cache[(company_id, version)] = {c.id: c for c in chunks}

    def delete(self, company_id: str, version: int) -> None:
        self._cache.pop((company_id, version), None)
        self._path(company_id, version).unlink(missing_ok=True)

    def delete_company(self, company_id: str) -> None:
        for key in [k for k in self._cache if k[0] == company_id]:
            del self._cache[key]
        shutil.rmtree(self._dir(company_id), ignore_errors=True)

    # ----------------------------------------------------------------- read --
    def versions_on_disk(self, company_id: str) -> list[int]:
        d = self._dir(company_id)
        if not d.exists():
            return []
        return sorted(int(p.stem[1:]) for p in d.glob("v*.jsonl") if p.stem[1:].isdigit())

    def companies_on_disk(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _COMPANY_RE.match(p.name))

    def all(self, company_id: str, version: int | None = None) -> dict[str, Chunk]:
        """{chunk id: Chunk} for the company's given (default: active) version; empty
        if there is none. Cached in memory."""
        v = self._resolve(company_id, version)
        if v is None:
            return {}
        key = (company_id, v)
        if key not in self._cache:
            path = self._path(company_id, v)
            if not path.exists():
                return {}
            chunks = [Chunk(**json.loads(line))
                      for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._cache[key] = {c.id: c for c in chunks}
        return self._cache[key]

    def get(self, company_id: str, chunk_id: str, version: int | None = None) -> Chunk | None:
        """The chunk, only if it belongs to THIS company's (given/active) version."""
        return self.all(company_id, version).get(chunk_id)
