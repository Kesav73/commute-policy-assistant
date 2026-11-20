# 15. Never a hang: a hard time limit on every stage

**Date:** 2026-11-20 · **Code:** `app/timeouts.py`, `app/pipeline.py`, `app/llm.py`, `app/main.py`, `app/config.py` · **Tests:** `tests/test_timeouts.py` (6 tests)

## Question
The brief (R10): *"When retrieval is slow or the LLM fails, they get a clear 'try again' rather than a hang or a half answer."* The pipeline had a 2.8 s deadline, but it was only *checked before each LLM call*. Was every stage actually bounded?

## What the audit found
| Stage | Before | Problem |
|---|---|---|
| Retrieval (embedding + Chroma + BM25) | No limit | A stuck lock or stalled disk would hold the request forever; the deadline check was never reached |
| LLM HTTP call | `httpx.Timeout(t)` | Limits each *phase* (connect, each read) separately, so a server trickling bytes could keep the call open far past `t` |
| Whole request | No backstop | Any unforeseen stall = hang |
| Slow-but-finishing retrieval, LLM errors, quota, malformed output | Handled | — |

## Change
- **`run_with_timeout(fn, t)`** (`app/timeouts.py`): runs a blocking step in a bounded worker pool and stops *waiting* after `t`, raising `StageTimeout`. A thread can't be cancelled, and waiting for a cancellation is itself a hang, so the stuck worker is left to finish in the background.
- **Retrieval, original question:** limit `RETRIEVAL_TIMEOUT_S` = 1.0 s (about 10× normal) or what's left of the deadline → 503 "try again", 0 LLM calls.
- **Retrieval with rewritten queries:** same limit, and it must leave 0.6 s for the answer call → on timeout, answer from the original chunks.
- **LLM calls:** the HTTP call runs inside `run_with_timeout`, so its **total** time is capped → `LLMTimeout`, handled as before.
- **Whole request:** `/ask` runs the pipeline in its own thread pool and stops waiting after `REQUEST_TIMEOUT_S` = 3.0 s → 503 "try again". This is the last-resort backstop.

## A bug the tests caught
The first backstop was middleware using `asyncio.wait_for`. A forced hang showed it **didn't fire**: the response took the full 5 s. `wait_for` cancels the task and then *waits for the cancellation to finish*, and a sync endpoint's worker thread can't be cancelled. Moving the limit into the endpoint with `asyncio.wait` (stop waiting, don't cancel) fixed it: **503 in 0.50 s**. A second, test-only delay came from the loop's default thread pool being joined at loop shutdown; giving `/ask` its own pool removed it.

## Verification (each test forces one stage to hang for 5 s)
| Forced hang | Result |
|---|---|
| Retrieval | 503 "try again" in < 1 s, 0 LLM calls |
| Rewritten-query retrieval | Answered from the original chunks in < 1.5 s |
| LLM server never finishes responding | `LLMTimeout` at the 0.3 s limit (< 1 s) |
| The whole pipeline | `/ask` answers 503 + `Retry-After: 5` in 0.52 s |
| Other endpoints | Unaffected (`/health` fine) |

205 tests pass locally and in Docker. The offline end-to-end flow is unchanged (2 calls per question, p95 49 ms with the fake LLM).

## Decision
Every stage on the request path now has a hard limit, and the request as a whole has one too. The worst case is a clear "try again" within 3 s, never a hang.
