"""Embeddings: nomic-embed-text-v1.5, run locally on the CPU (ONNX via fastembed), 768 numbers per text.

Chosen by measurement (EVALUATION.md): best of four models on our evaluation set.

The model was trained with a prefix that says which side of a search a text is on,
so we always use it:
    chunks     "search_document: " + text      (when a policy is published)
    questions  "search_query: "    + text      (when an employee asks)
Both sides must use the same model and matching prefixes. EMBEDDING_ID records that
on every collection, and a collection made differently is refused.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
EMBEDDING_ID = f"{EMBEDDING_MODEL}|{DOCUMENT_PREFIX.strip()}|{QUERY_PREFIX.strip()}"

# The Docker image downloads the model files here at BUILD time, so a running
# container never needs the network for embeddings.
CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", Path.home() / ".cache" / "fastembed"))

# Vectors of chunks already embedded in this process, by text. Publishing an
# unchanged section again (or ingesting in every test) then skips the model.
_chunk_vectors: dict[str, np.ndarray] = {}
MAX_CACHED_CHUNKS = 5000


@lru_cache
def _model():
    """Loaded once per process (~1 s), on first use."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=str(CACHE_DIR))


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.asarray(list(_model().embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # unit length, for cosine distance


def embed_query(question: str) -> np.ndarray:
    """One question -> array of shape (1, 768)."""
    return _embed([QUERY_PREFIX + question])


def embed_documents(texts: list[str]) -> np.ndarray:
    """Chunk texts -> array of shape (len(texts), 768)."""
    if len(_chunk_vectors) + len(texts) > MAX_CACHED_CHUNKS:
        _chunk_vectors.clear()              # keep memory bounded; the cache is only a speed-up
    new_texts = []
    for t in texts:
        if t not in _chunk_vectors and t not in new_texts:
            new_texts.append(t)
    if new_texts:
        new_vectors = _embed([DOCUMENT_PREFIX + t for t in new_texts])
        for t, v in zip(new_texts, new_vectors):
            _chunk_vectors[t] = v
    return np.stack([_chunk_vectors[t] for t in texts])
