"""Never a hang (PS.md R10): every blocking stage has a hard wall-clock limit.

Each test forces one stage to hang and checks the caller still gets a clear
answer, fast.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import app.main as main_module
import app.pipeline as pipeline_module
from app import messages
from app.config import get_settings
from app.ingest import ingest_all
from app.llm import FakeLLM, LLMTimeout, OpenAICompatLLM
from app.pipeline import ask
from app.store import open_store
from app.timeouts import StageTimeout, run_with_timeout
from tests.conftest import bearer

POLICIES = Path(__file__).resolve().parent.parent / "policies"
Q = "Does a trained security escort travel with women on night drops?"
HANG_S = 5.0


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = open_store(tmp_path_factory.mktemp("data"))
    ingest_all(s, POLICIES)
    return s


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ------------------------------------------------------------- primitive --

def test_run_with_timeout_returns_raises_and_propagates():
    assert run_with_timeout(lambda: 42, 1.0) == 42
    t0 = time.perf_counter()
    with pytest.raises(StageTimeout):
        run_with_timeout(time.sleep, 0.2, HANG_S)
    assert time.perf_counter() - t0 < 1.0
    with pytest.raises(ZeroDivisionError):
        run_with_timeout(lambda: 1 / 0, 1.0)
    with pytest.raises(StageTimeout):
        run_with_timeout(lambda: 1, 0)              # no time left at all


# ------------------------------------------------------------- retrieval --

def test_hanging_retrieval_gives_try_again_fast(store, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: time.sleep(HANG_S))
    s = replace(get_settings(), retrieval_timeout_s=0.3)
    llm = FakeLLM()
    r, took = timed(ask, store, llm, "acme", Q, s)
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN
    assert r.reason == "retrieval timeout" and llm.calls == []
    assert took < 1.0


def test_hanging_rewritten_retrieval_keeps_original_chunks(store, monkeypatch):
    real = pipeline_module.retrieve

    def slow_for_rewrites(*a, **k):
        if k.get("rewrites"):
            time.sleep(HANG_S)
        return real(*a, **k)

    monkeypatch.setattr(pipeline_module, "retrieve", slow_for_rewrites)
    s = replace(get_settings(), query_rewrite=True, retrieval_timeout_s=0.3)
    llm = FakeLLM(script=[json.dumps({"queries": ["escort"]}),
                          json.dumps({"status": "answered", "answer": "Escort after 8:00 PM.",
                                      "citations": ["S1"]})])
    r, took = timed(ask, store, llm, "acme", Q, s)
    assert r.status == "answered" and "rewritten retrieval timeout" in r.reason
    assert took < 1.5


# ------------------------------------------------------------------- LLM --

def test_llm_call_is_capped_in_total_time():
    """A server that never finishes responding: httpx's per-phase timeout alone could
    wait much longer; the total-time cap stops it at timeout_s."""
    def stuck(request):
        time.sleep(HANG_S)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    llm = OpenAICompatLLM("groq", "test-key", transport=httpx.MockTransport(stuck))
    t0 = time.perf_counter()
    with pytest.raises(LLMTimeout):
        llm.complete_json("sys", "user", "qwen/qwen3.8-27b", timeout_s=0.3)
    assert time.perf_counter() - t0 < 1.0


# ----------------------------------------------------------- whole request --

def test_request_backstop_returns_503_when_anything_hangs(client, token_for, monkeypatch):
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "0.5")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "ask", lambda *a, **k: time.sleep(HANG_S))
    main_module.app.state.store, main_module.app.state.llm = None, None
    try:
        headers = bearer(token_for("E-ACME-001"))
        t0 = time.perf_counter()
        r = client.post("/ask", json={"question": Q}, headers=headers)
        assert time.perf_counter() - t0 < 1.5
        assert r.status_code == 503 and r.json()["message"] == messages.TRY_AGAIN
        assert r.headers["Retry-After"] == "5"
    finally:
        del main_module.app.state.store, main_module.app.state.llm


def test_backstop_does_not_touch_other_endpoints(client):
    assert client.get("/health").status_code == 200
