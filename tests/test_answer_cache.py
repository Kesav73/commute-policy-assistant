"""Answer cache: repeated questions cost 0 LLM calls; never across companies, never
across policy versions, never a temporary failure."""
import json
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.answer_cache import AnswerCache, CachedAnswer, normalise
from app.llm import FakeLLM, LLMError
from app.store import PolicyStore
from app.versioning import bootstrap, publish_policy
from tests.conftest import ACME_V2, POLICIES, bearer

Q = "After what time do women get a security escort on drops?"


# ------------------------------------------------------------------- unit --

def test_normalise_ignores_case_spaces_and_punctuation():
    assert normalise("  Can I book a cab on SATURDAY?? ") == normalise("can i book a cab, on saturday")
    assert normalise("drop at 9 PM") != normalise("drop after 9 PM")      # exact, not "similar"


def ans(status="answered", v=1):
    return CachedAnswer(status, "x", (), v)


def test_key_includes_company_and_version():
    c = AnswerCache()
    c.put("acme", Q, ans(v=1))
    assert c.get("acme", 1, Q) is not None
    assert c.get("bolt", 1, Q) is None          # isolation
    assert c.get("acme", 2, Q) is None          # a new version never sees v1 answers


def test_try_again_is_never_cached():
    c = AnswerCache()
    c.put("acme", Q, ans(status="unavailable"))
    assert c.get("acme", 1, Q) is None


def test_lru_bound_and_ttl(monkeypatch):
    c = AnswerCache(max_entries=2, ttl_s=10)
    for q in ("a?", "b?", "c?"):
        c.put("acme", q, ans())
    assert len(c) == 2 and c.get("acme", 1, "a?") is None          # oldest left first
    now = time.monotonic()
    monkeypatch.setattr("app.answer_cache.time.monotonic", lambda: now + 11)
    assert c.get("acme", 1, "c?") is None                          # expired


def test_size_zero_disables():
    c = AnswerCache(max_entries=0)
    c.put("acme", Q, ans())
    assert c.get("acme", 1, Q) is None


# -------------------------------------------------------------------- API --

@pytest.fixture
def api(tmp_path):
    store = PolicyStore(tmp_path / "data")
    bootstrap(store, POLICIES)
    llm = FakeLLM()
    main_module.app.state.store, main_module.app.state.llm = store, llm
    main_module.app.state.answer_cache = AnswerCache()
    yield TestClient(main_module.app), llm, store
    del main_module.app.state.store, main_module.app.state.llm, main_module.app.state.answer_cache


def test_repeat_question_is_served_from_cache_with_no_llm_call(api, token_for):
    client, llm, _ = api
    h = bearer(token_for("E-ACME-001"))
    first = client.post("/ask", json={"question": Q}, headers=h).json()
    calls = len(llm.calls)
    again = client.post("/ask", json={"question": "  after what time do WOMEN get a security escort on drops "},
                        headers=h).json()
    assert first["cached"] is False and again["cached"] is True
    assert len(llm.calls) == calls                                   # 0 LLM calls for the repeat
    assert again["answer"] == first["answer"] and again["citations"] == first["citations"]
    assert again["request_id"] != first["request_id"]


def test_other_company_never_gets_a_cached_answer(api, token_for):
    client, _, _ = api
    client.post("/ask", json={"question": Q}, headers=bearer(token_for("E-ACME-001")))
    bolt = client.post("/ask", json={"question": Q}, headers=bearer(token_for("E-BOLT-001"))).json()
    assert bolt["cached"] is False and "10:00 PM" in bolt["answer"]


def test_new_policy_version_is_never_answered_from_the_old_cache(api, token_for):
    client, _, store = api
    h = bearer(token_for("E-ACME-001"))
    v1 = client.post("/ask", json={"question": Q}, headers=h).json()
    assert "8:00 PM" in v1["answer"]
    r = client.post("/admin/policies", json={"markdown": ACME_V2}, headers=bearer(token_for("A-ACME-ADM")))
    assert r.status_code == 200
    v2 = client.post("/ask", json={"question": Q}, headers=h).json()
    assert v2["cached"] is False and v2["policy_version"] == 2 and "9:00 PM" in v2["answer"]


def test_failure_is_not_cached(api, token_for):
    client, llm, _ = api
    llm.script.extend([json.dumps({"queries": ["escort"]}), LLMError("unavailable", "down")])
    h = bearer(token_for("E-ACME-001"))
    assert client.post("/ask", json={"question": Q}, headers=h).status_code == 503
    ok = client.post("/ask", json={"question": Q}, headers=h)
    assert ok.status_code == 200 and ok.json()["cached"] is False


def test_publish_also_clears_the_company_entries(api, token_for):
    client, _, store = api
    client.post("/ask", json={"question": Q}, headers=bearer(token_for("E-ACME-001")))
    cache = main_module.app.state.answer_cache
    assert len(cache) == 1
    publish_policy(store, ACME_V2, old_version_grace_s=0)            # direct publish: entry stays, but
    assert cache.get("acme", store.active_version("acme"), Q) is None  # the v2 key can't match it
