# 01. Embedding speed: Chroma's fixed padding

**Date:** 2026-11-19 · **Commit:** `3c95b0e` (feat(ingest)) · **Code:** `app/embeddings.py`

## Question
The brief allows 3 s at p95 for the whole answer, LLM included. How much of that does embedding the question take?

## Finding
Chroma's built-in `DefaultEmbeddingFunction` (all-MiniLM-L6-v2, ONNX) pads **every** input to 256 tokens. A 12-word question is processed as if it were 256 tokens long.

| Embedding function | Time per question |
|---|---|
| Chroma default (pad to 256) | ~440 ms |
| `MiniLMEmbedding` (pad to longest in batch) | ~5.9 ms |

## Change
A small subclass, `MiniLMEmbedding`, keeps the same model weights and pooling but pads only to the longest text in the batch. Padding tokens are masked out of the mean pooling, so the vectors are the same; a test checks this against Chroma's own output.

## Decision
Kept. About 75× faster, no change in retrieval results, no new dependency. It frees about 0.4 s of the 3 s budget for the LLM.

Related: the first `docker compose up` used to download the model (~2 min). The Dockerfile now downloads it at build time.
