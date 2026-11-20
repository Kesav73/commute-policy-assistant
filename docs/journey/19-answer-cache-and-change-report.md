# 19. Answer cache (per policy version) and change report on publish

**Date:** 2026-11-20 · **Code:** `app/answer_cache.py`, `app/main.py`, `app/versioning.py` (`compare_versions`) · **Tests:** `tests/test_answer_cache.py` (10), `tests/test_versioning.py` (+4)

## Why
- The brief: *"Employees ask the desk the same things every day."* Each repeat cost 2 LLM calls, ~1 s and ~1,500 tokens of a free quota that allows ~80–100 questions a day.
- An admin publishing a new version had no way to see what the upload actually changed.

## 1. Answer cache
**Key = (company, active policy version, normalised question).**
- **Company in the key:** Acme's cached answer can never reach a Bolt employee.
- **Version in the key:** once v2 is live, lookups use v2, so a v1 answer can never be returned. Correctness under updates needs no invalidation step; publishing also drops the company's entries just to free memory.
- **Exact match after normalising** case, spacing and punctuation. Deliberately not "similar questions": "drop *at* 9 PM" and "drop *after* 9 PM" must not share an answer.
- **Cached:** answered, not covered, guardrail refusals. **Never cached:** "try again".
- **Bounded:** 1,000 entries, least recently used out, 6 h time-to-live; in memory (a restart starts empty). `ANSWER_CACHE_SIZE=0` disables it.
- **At the endpoint, not in the pipeline,** so tests and the end-to-end eval always measure the real pipeline. The response carries `"cached": true|false`.

## 2. Change report on publish
`POST /admin/policies` now returns, for example:
```json
"changes": {"changed": ["4.1 Night-Shift Safety > Escort Requirement", "4.2 Night-Shift Safety > Routing Rules"],
            "added": [], "removed": [], "unchanged": 25}
```
Computed section by section (title + text) against the live version before the switch, and written to the log as an audit line. "26 removed" would immediately expose a wrong or truncated upload.

## Verification
- **Tests (14 new, 240 total):** normalisation; company and version isolation in the key; "try again" not cached; LRU bound and expiry; disabled at size 0. Through the API: a repeat makes 0 LLM calls; another company is never served a cached answer; **after publishing v2 the same question is not cached and answers 9:00 PM**; a failure isn't cached. Change report: edited, added, removed and first-version cases.
- **Live container** (offline mode):

| Step | Result |
|---|---|
| Ask escort question | v1, `cached=false`, 8:00 PM |
| Same question, different case/spacing | v1, **`cached=true`, 14 ms** |
| Publish acme_v2.md | `changed: 4.1, 4.2; added 0; removed 0; unchanged 25` |
| Same question | **v2, `cached=false`**, 9:00 PM |
| Again | v2, `cached=true`, 19 ms |

With the real LLM a fresh answer takes ~0.8–1.25 s against ~15 ms from the cache.

## Decision
Both kept: small, quota-free, and they tie together latency, cost and versioning.
