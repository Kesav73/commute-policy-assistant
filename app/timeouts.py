"""A hard time limit for any step that could block (embedding + search, an LLM call).

Python can't stop a running thread, so the step runs in a worker thread and we
simply stop WAITING for it after `timeout_s`: the caller gets StageTimeout at once
and can answer "please try again". A stuck worker finishes (or not) in the
background without holding up the request.
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

_WORKERS = ThreadPoolExecutor(max_workers=16, thread_name_prefix="limited")


class StageTimeout(TimeoutError):
    pass


def run_with_timeout(func, timeout_s: float, *args, **kwargs):
    """Return func(*args, **kwargs), or raise StageTimeout if it takes longer than
    timeout_s. Errors raised by func are passed through unchanged."""
    if timeout_s <= 0:
        raise StageTimeout("no time left")
    future = _WORKERS.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()   # drops it if it hasn't started yet; a running one is left alone
        raise StageTimeout(f"{getattr(func, '__name__', 'step')} took longer than {timeout_s:.2f}s") from None
