"""LLM client: request shape and error mapping (mock HTTP, never the real API)."""
import json

import httpx
import pytest

from app.config import ConfigError, get_settings
from app.llm import (FakeLLM, LLMRateLimited, LLMRequestError, LLMTimeout, LLMUnavailable,
                     OpenAICompatLLM, build_llm)


def client_returning(handler) -> tuple[OpenAICompatLLM, list]:
    seen = []

    def wrapped(request: httpx.Request):
        seen.append(request)
        return handler(request)

    return OpenAICompatLLM("groq", "test-key", transport=httpx.MockTransport(wrapped)), seen


def ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_request_shape_and_model_params():
    llm, seen = client_returning(lambda r: ok('{"status":"not_covered"}'))
    assert llm.complete_json("sys", "usr", "qwen/qwen3.8-27b", 2.0) == '{"status":"not_covered"}'
    body = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/chat/completions")
    assert seen[0].headers["Authorization"] == "Bearer test-key"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0
    assert body["reasoning_effort"] == "none"          # qwen: reasoning off
    assert [m["role"] for m in body["messages"]] == ["system", "user"]

    llm.complete_json("sys", "usr", "openai/gpt-oss-120b", 2.0)
    assert json.loads(seen[1].content)["reasoning_effort"] == "low"


@pytest.mark.parametrize("response, error", [
    (httpx.Response(429), LLMRateLimited),
    (httpx.Response(503), LLMUnavailable),
    (httpx.Response(400, text="bad model"), LLMRequestError),
    (httpx.Response(200, json={"unexpected": True}), LLMUnavailable),
])
def test_http_errors_are_mapped(response, error):
    llm, _ = client_returning(lambda r: response)
    with pytest.raises(error):
        llm.complete_json("s", "u", "m", 1.0)


def test_timeout_and_network_errors_are_mapped():
    def slow(r):
        raise httpx.ReadTimeout("slow", request=r)

    def down(r):
        raise httpx.ConnectError("down", request=r)

    with pytest.raises(LLMTimeout):
        client_returning(slow)[0].complete_json("s", "u", "m", 1.0)
    with pytest.raises(LLMUnavailable):
        client_returning(down)[0].complete_json("s", "u", "m", 1.0)


# ---------------------------------------------------------------- config --

def test_no_key_means_offline_fake(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    get_settings.cache_clear()
    s = get_settings()
    assert s.llm_provider == "fake"
    assert isinstance(build_llm(s), FakeLLM)


def test_groq_key_selects_groq_with_benchmarked_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    get_settings.cache_clear()
    s = get_settings()
    assert (s.llm_provider, s.llm_model, s.llm_fallback_model) == (
        "groq", "qwen/qwen3.8-27b", "openai/gpt-oss-120b")


def test_explicit_provider_without_key_fails_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        get_settings()


# ------------------------------------------------------------------ fake --

def test_fake_llm_default_answers_from_s1():
    user = '<excerpt id="S1" section="4.1">\nNight > Escort\nEscort after 8 PM. More text.\n</excerpt>'
    out = json.loads(FakeLLM().complete_json("s", user, "m", 1.0))
    assert out == {"status": "answered", "answer": "Escort after 8 PM.", "citations": ["S1"]}


def test_fake_llm_script_and_timeout():
    fake = FakeLLM(script=["raw text", LLMRateLimited("x")])
    assert fake.complete_json("s", "u", "m", 1.0) == "raw text"
    with pytest.raises(LLMRateLimited):
        fake.complete_json("s", "u", "m", 1.0)
    with pytest.raises(LLMTimeout):
        FakeLLM(delay_s=0.2).complete_json("s", "u", "m", timeout_s=0.05)
