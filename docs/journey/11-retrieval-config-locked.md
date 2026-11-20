# 11. Retrieval configuration locked: hybrid, 3 chunks read = 3 chunks citable

**Date:** 2026-11-20 · **Type:** decision (no new experiment)

## Question
Retrieval experiments are done (entries 07–10). One option was still open: let the LLM *read* 5 chunks while it may still *cite* at most 3 (the brief's C3 limits cited chunks). Take it, or keep reading = citing = 3?

## Decision
**The LLM gets exactly the 3 chunks it is allowed to cite.** Final retrieval configuration:

| Part | Choice | Evidence |
|---|---|---|
| Chunking | one chunk per `### N.M` sub-section, headings prepended | entries 05–06 |
| Storage | chunk store (content) + Chroma (ids + vectors) | entry 06 |
| Embeddings | nomic-embed-text-v1.5 with `search_document:` / `search_query:` prefixes | entries 07–08 |
| Search | hybrid: stemmed BM25 + vectors, weighted RRF 2:1 | entry 09 |
| Re-ranking | none | entry 10: best re-ranker only ties, at ~1 s per question |
| Chunks to the LLM | **3** (`TOP_K = 3`) | this entry |
| Result | recall@1 0.63, recall@3 0.92, 66/75, ~34 ms | entry 09 |

## Why 3 and not "read 5, cite 3"
- **Every answer stays checkable against what was read.** If the model reads 5 but cites 3, a fact could come from an uncited excerpt, and the employee couldn't verify it from the citations. With read = cite, every excerpt the answer could rely on is a valid citation.
- **The C3 explanation stays simple:** "the model only ever sees 3 excerpts, so it cannot cite more than 3", enforced by `TOP_K` in the retriever and again by the citation validator.
- **Less surface for injected text:** fewer excerpts in the prompt means fewer places for an instruction-like chunk (Crest §7) to reach the model.
- **Smaller prompt, steadier latency** against the 3 s p95.

**Accepted cost:** multi-section questions whose second section ranks 4th–6th stay partly unanswered (acme-18, 19, 20, 22, bolt-19). For these the model should answer what the 3 excerpts support and not guess the rest. This is checked when the end-to-end answer evaluation is rebuilt.
