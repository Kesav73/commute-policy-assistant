"""Hard wall-clock limits for any blocking step (PS.md R10: never a hang).

Python can't kill a thread, so a step that might block (embedding + vector search,
an HTTP call to the LLM) runs in a worker thread and we stop WAITING for it after
`timeout_s`. The caller gets StageTimeout immediately and answers "try again";
the stuck worker finishes (or not) in the background without holding the request.

The pool is bounded. If every worker were stuck, new work would queue, and the
caller's wait would still time out: the request never hangs either way.
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, TypeVar

T = TypeVar("T")

_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="bounded")


class StageTimeout(TimeoutError):
    pass


def run_with_timeout(fn: Callable[..., T], timeout_s: float, *args, **kwargs) -> T:
    """fn(*args, **kwargs), or StageTimeout if it hasn't returned within timeout_s.
    Exceptions raised by fn are re-raised unchanged."""
    if timeout_s <= 0:
        raise StageTimeout("no time left")
    future = _POOL.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()   # no-op if already running; drops it if still queued
        raise StageTimeout(f"{getattr(fn, '__name__', 'step')} took longer than {timeout_s:.2f}s") from None
