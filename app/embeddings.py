"""Embedding model: all-MiniLM-L6-v2 (ONNX, CPU, 384-dim), same weights as Chroma's default.

Why a subclass instead of Chroma's DefaultEmbeddingFunction as-is:
Chroma pads EVERY text to 256 tokens. A 12-word question is then processed as
256 tokens - measured ~440 ms per question on our machine, a big slice of the
3 s p95 budget (PS.md C1). Here we pad only to the longest text in the batch.
Padding tokens are masked out of the mean-pooling, so the vectors are the same
(checked by tests/test_embeddings.py); only the wasted work is removed.
"""
from functools import cached_property, lru_cache
from typing import Any

import numpy as np
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

MAX_TOKENS = 256  # sentence-transformers' limit for this model; longer text is cut


class MiniLMEmbedding(ONNXMiniLM_L6_V2):
    def __init__(self) -> None:
        super().__init__(preferred_providers=["CPUExecutionProvider"])

    @cached_property
    def tokenizer(self) -> Any:
        tok = super().tokenizer
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")  # pad to longest in batch, not to 256
        return tok

    def _forward(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(documents), batch_size):
            encoded = self.tokenizer.encode_batch(documents[i:i + batch_size])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            hidden = self.model.run(None, {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            })[0]
            # Mean of the real (unmasked) token vectors, then unit length -> cosine-ready.
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(self._normalize(pooled).astype(np.float32))
        return np.concatenate(out)


@lru_cache
def embedding_function() -> MiniLMEmbedding:
    """The single shared instance. The SAME function embeds chunks at ingest and
    questions at query time; vectors from different models are not comparable."""
    return MiniLMEmbedding()
