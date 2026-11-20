"""Never a hang: each test forces one stage to hang for 5 s and checks the answer still comes fast."""
import json
import time
from dataclasses import replace

import pytest

import app.main as main_module
import app.pipeline as pipeline_module
from app import messages
from app.config import get_settings
from app.llm import FakeLLM
from app.pipeline import ask
from app.timeouts import StageTimeout, run_with_timeout
from tests.conftest import ESCORT_Q, bearer

HANG_S = 5.0


def timed(func, *args):
    started = time.perf_counter()
    result = func(*args)
    return result, time.perf_counter() - started


def test_run_with_timeout():
    assert run_with_timeout(lambda: 42, 1.0) == 42
    started = time.perf_counter()
    with pytest.raises(StageTimeout):
        run_with_timeout(time.sleep, 0.2, HANG_S)
    assert time.perf_counter() - started < 1.0
    with pytest.raises(ZeroDivisionError):                 # errors pass through unchanged
        run_with_timeout(lambda: 1 / 0, 1.0)
    with pytest.raises(StageTimeout):                      # no time left at all
        run_with_timeout(lambda: 1, 0)


def test_hanging_retrieval_gives_try_again_fast(store, monkeypatch):
    monkeypatch.setattr(pipeline_module, "retrieve", lambda *a, **k: time.sleep(HANG_S))
    llm = FakeLLM(script=[json.dumps({"queries": ["escort"]})])
    r, took = timed(ask, store, llm, "acme", ESCORT_Q, replace(get_settings(), retrieval_timeout_s=0.3))
    assert r.status == "unavailable" and r.answer == messages.TRY_AGAIN and r.reason == "retrieval timeout"
    assert len(llm.calls) == 1 and took < 1.5              # the rewrite only: no answer call


def test_hanging_model_gives_try_again_within_the_deadline(store):
    llm = FakeLLM(delay_s=HANG_S)                          # every call hangs
    r, took = timed(ask, store, llm, "acme", ESCORT_Q, get_settings())
    assert r.status == "unavailable" and took < get_settings().answer_deadline_s + 0.5


def test_the_whole_request_has_a_backstop(client, token_for, monkeypatch):
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "0.5")
    get_settings.cache_clear()
    monkeypatch.setattr(main_module, "ask", lambda *a, **k: time.sleep(HANG_S))
    main_module.app.state.store, main_module.app.state.llm = None, None
    try:
        headers = bearer(token_for("E-ACME-001"))
        started = time.perf_counter()
        r = client.post("/ask", json={"question": ESCORT_Q}, headers=headers)
        assert time.perf_counter() - started < 1.5
        assert r.status_code == 503 and r.json()["message"] == messages.TRY_AGAIN
        assert r.headers["Retry-After"] == "5"
    finally:
        del main_module.app.state.store, main_module.app.state.llm


def test_the_backstop_does_not_touch_other_endpoints(client):
    assert client.get("/health").status_code == 200
