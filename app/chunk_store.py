"""Chunk store: the single source of truth for chunk CONTENT (text + metadata).

    data/chunks/<company_id>.jsonl   one line per chunk, in policy order

The vector index (Chroma, app/store.py) holds only chunk ids and their vectors.
Everything a user or the LLM sees - excerpt text, document title, section number
for the citation - is read from here by id. So:
  * the embedding model can change (or several can be compared) without
    touching the content, and
  * the company check after retrieval uses this file, which the vector index
    cannot influence.

Writes are atomic: the new file is written to a temp file and renamed over the
old one, so a reader never sees half a file.
"""
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from app.policy_parser import Chunk

_COMPANY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ChunkStoreError(RuntimeError):
    pass


class ChunkStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Chunk]] = {}  # company -> {chunk id -> Chunk}

    def _path(self, company_id: str) -> Path:
        if not _COMPANY_RE.match(company_id):
            raise ChunkStoreError(f"invalid company_id {company_id!r}")
        return self.root / f"{company_id}.jsonl"

    def put(self, company_id: str, chunks: list[Chunk]) -> None:
        """Replace the company's chunks with exactly these (atomic rename)."""
        if any(c.company_id != company_id for c in chunks):
            raise ChunkStoreError(f"chunk from another company passed to {company_id}")
        if len({c.id for c in chunks}) != len(chunks):
            raise ChunkStoreError(f"duplicate chunk ids for {company_id}")
        path = self._path(company_id)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{company_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._cache[company_id] = {c.id: c for c in chunks}

    def all(self, company_id: str) -> dict[str, Chunk]:
        """{chunk id: Chunk} for the company; empty if it has no chunks. Cached in memory."""
        if company_id not in self._cache:
            path = self._path(company_id)
            if not path.exists():
                return {}
            chunks = [Chunk(**json.loads(line))
                      for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self._cache[company_id] = {c.id: c for c in chunks}
        return self._cache[company_id]

    def get(self, company_id: str, chunk_id: str) -> Chunk | None:
        """The chunk, only if it belongs to THIS company's store."""
        return self.all(company_id).get(chunk_id)
