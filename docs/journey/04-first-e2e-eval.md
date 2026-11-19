# 04. First end-to-end evaluation

**Date:** 2026-11-19 · **Commit:** `2679653` (eval: end-to-end) · **Status:** historical. The script was removed in entry 05, to be rebuilt in a more structured way.

## Question
With the whole system running (JWT → retrieval → Qwen → citation checks → `POST /ask`), how good are the answers, and where do the failures come from?

## Setup
All 90 gold questions through the real HTTP API with a real token per company and the real LLM (Groq, Qwen primary). Deterministic checks per answer: status (answered / not covered), required facts present, forbidden text absent (for example other companies' values), correct section cited, no cross-company leak.

## Results

| Metric | Result |
|---|---|
| Pass | 72/90 (73 after fixing a grader mistake on acme-17) |
| Status correct | 81/90 |
| Facts correct | 60/75 |
| Citation correct | 63/75 |
| Not-covered handled correctly | 15/15 |
| Leaks / forbidden text / injection obeyed | 0 / 0 / 0 |
| LLM calls | 89 questions used 1 call, 1 used 2; max 2 |
| Latency | p50 0.44 s, p95 0.83 s, max 1.3 s |

Confusion: answered→answered 66, answered→not covered 9, not covered→not covered 15, **not covered→answered 0**. When it was wrong, it was over-cautious, never over-confident.

## Where the 18 failures came from

| Cause | Count |
|---|---|
| Retrieval missed a required section | 14 |
| Model over-cautious (said not covered) | 2 (crest-29, bolt-29) |
| Answer incomplete (bolt-04 left out the Friday 5 PM deadline) | 1 |
| Grader mistake | 1 |

## Decision
**Retrieval is the bottleneck.** Improving it is the next priority, before any prompt work. That is why the following experiments focus on retrieval (embeddings, then chunking, hybrid search and re-ranking).
