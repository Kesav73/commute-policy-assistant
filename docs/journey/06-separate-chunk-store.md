# 06. Separate chunk store: content apart from vectors

**Date:** 2026-11-19 · **Code:** `app/chunk_store.py`, `app/store.py`, `app/retriever.py`

## Question
Chunking stays as it is (one chunk per `### N.M` sub-section, headings prepended: document-aware). But where should the chunk content live? Until now Chroma held everything for each chunk: id, vector, text and metadata. Before comparing embedding models we want content and vectors apart, so a model swap never touches the content.

## Change
| Before | After |
|---|---|
| Chroma: id + vector + text + metadata | **Chunk store** `data/chunks/<company>.jsonl`: id → text + metadata (source of truth) |
| | **Vector index** Chroma `policy__<company>`: id + vector only; collection metadata records `company_id` and `embedding_model` |
| Company check reads the metadata Chroma returns | Company check looks the id up in **the caller's own chunk store**; an id that isn't there is dropped and logged |

- Ingest embeds first, then writes the chunk store (temp file + atomic rename), then rebuilds the collection with ids + vectors, and finally checks that both stores hold exactly the same ids.
- Chunk ids stay deterministic: `<doc_id>:<section>`, e.g. `acme-transport-policy:4.1`.
- Tests: Chroma holds no documents or metadata; both stores hold the same ids; a chunk is only found in its own company's store; a planted Bolt vector in Acme's collection is dropped (139 tests pass).

## Results
Same retrieval, so recall must not change; latency measured over several runs on the same machine.

| | recall@1 | recall@3 | avg latency |
|---|---|---|---|
| Before (entry 05) | 0.55 | 0.82 | 4.8–5.1 ms (3 runs) |
| **After** | **0.55** | **0.82** | **4.3–4.5 ms** (3 runs) |

Identical recall, as expected. About 10% faster, because Chroma no longer reads and returns text and metadata; the chunk lookup is an in-memory dictionary.

## Decision
Kept. Benefits: embedding models can be swapped or compared over one fixed set of chunks; the content is a plain file anyone can read; the isolation check no longer trusts data returned by the vector index. Cost: one more module (~70 lines) and two stores to keep in step, which ingest verifies.

Next: compare embedding models on this setup.
