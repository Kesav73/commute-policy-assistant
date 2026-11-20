# 18. Simpler pipeline: rewrite first, one retrieval

**Date:** 2026-11-20 · **Code:** `app/pipeline.py`, `app/retriever.py` · **Tests:** `tests/test_pipeline.py`, `tests/test_rewriter.py`, `tests/test_timeouts.py`

## Question
The pipeline retrieved **twice**: once with the original question (for the "not covered" gate and as a backup), and again with the rewritten queries. Is the first retrieval worth its complexity?

## Evidence
| Purpose of retrieval #1 | What the final run (entry 17) shows |
|---|---|
| Gate before any LLM call (off-topic costs 0 calls) | **0 of 90** questions stopped at the gate (`LLM calls: {1: 11, 2: 79}`); 11 stopped at the rewriter's verdict instead |
| Backup chunks if the rewrite fails | Needed once (acme-01: transient Groq error on the rewrite); searching the original question *at that point* does the same |
| Gate on the original question, so a rewrite can't make nonsense "covered" | Covered by the rewriter's `off_topic` verdict |

## Change
```
before: retrieve(original) → gate → rewrite → retrieve(rewrites) → answer
after:  rewrite → retrieve(rewrites, or original if the rewrite failed) → gate → answer
```
- One retrieval per question; the gate uses the best distance any searched query reached.
- Cost: a nonsense question ("poem about cats") now uses 1 LLM call (the rewrite) instead of 0.
- A retrieval timeout after the rewrite is now "try again" (there is no earlier set of chunks to fall back to).

## Verification (no new LLM spend)
The rewrite prompts and the chunks for the answer are the same as in the final run, so the end-to-end eval was rerun from the response cache (89 of 90 calls cached):

| | Final run (entry 17) | Simpler pipeline |
|---|---|---|
| Pass / status / facts / citation | 83/90 · 85/90 · 69/75 · 71/75 | **83/90 · 85/90 · 69/75 · 71/75** |
| LLM calls | {1: 11, 2: 79} | {1: 11, 2: 79} |

Every per-question outcome is the same. The only recorded differences: the 11 guardrail-stopped questions no longer show chunks from the unused retrieval #1, and acme-01 (whose rewrite had failed in the live run) now uses its rewritten chunks; it passes either way. Live latency is kept from entry 17 (p95 1.25 s), since cached calls take ~0 ms; the new flow can only be faster (one embedding pass fewer).

226 tests pass (new: no retrieval before the rewrite; off-topic verdict stops after one call; gate after a rewrite; retrieval timeout after a rewrite → "try again").

## Decision
Kept. Same results, one stage fewer, simpler to explain: **rewrite → retrieve → gate → answer**.
