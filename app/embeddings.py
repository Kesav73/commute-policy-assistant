"""Embedding model: nomic-embed-text-v1.5 (ONNX on CPU via fastembed, 768 dims).

Chosen by measurement (EVALUATION.md): on the 90-question gold
set it lifted recall@1 0.55 -> 0.64 and recall@3 0.82 -> 0.89 over all-MiniLM-L6-v2,
for ~30 ms more per question. Same library and model files as the benchmark, so the
app's vectors are exactly the measured ones.

Prefixes (from the model card; the model was trained with them):
    chunks     "search_document: " + text
    questions  "search_query: "    + text
Search is asymmetric - a short question vs a long rule - and the prefix tells the
model which side it is embedding. Both sides MUST use the same model and the
matching prefixes, so both live here and nowhere else, and EMBEDDING_ID is recorded
on every Chroma collection and checked at query time.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
# What a collection records about how its vectors were made. Change any part
# (model or prefixes) and existing collections no longer match -> re-ingest.
EMBEDDING_ID = f"{EMBEDDING_MODEL}|{DOCUMENT_PREFIX.strip()}|{QUERY_PREFIX.strip()}"

# Model files live here. The Docker image downloads them at BUILD time, so a running
# container never needs the network for embeddings.
CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", Path.home() / ".cache" / "fastembed"))


@lru_cache
def _model():
    """Load once per process (~1 s), then reuse. Import is lazy so tests that never
    embed don't pay for it."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))


def _embed(texts: list[str]) -> np.ndarray:
    v = np.asarray(list(_model().embed(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)   # unit length -> cosine-ready


# Chunk vectors already computed in this process, keyed by chunk text. Re-ingesting an
# unchanged policy (a restart-free re-ingest, every test that ingests, the eval) then
# skips the model. Bounded so a long-running process can't grow it without limit.
_DOC_CACHE: dict[str, np.ndarray] = {}
_DOC_CACHE_MAX = 5000


def embed_documents(texts: list[str]) -> np.ndarray:
    """Chunks at ingest time."""
    missing = list(dict.fromkeys(t for t in texts if t not in _DOC_CACHE))
    if missing:
        for t, v in zip(missing, _embed([DOCUMENT_PREFIX + t for t in missing])):
            if len(_DOC_CACHE) < _DOC_CACHE_MAX:
                _DOC_CACHE[t] = v
            else:
                return _embed([DOCUMENT_PREFIX + t for t in texts])   # cache full: just embed
    return np.stack([_DOC_CACHE[t] for t in texts])


def embed_query(text: str) -> np.ndarray:
    """One question at ask time. Shape (1, 768)."""
    return _embed([QUERY_PREFIX + text])
