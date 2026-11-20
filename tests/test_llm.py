"""LLM client: the request we send, and how failures are reported (fake HTTP, never the real API)."""
import json
import time

import httpx
import pytest

from app.config import FALLBACK_MODEL, PRIMARY_MODEL, ConfigError, get_settings
from app.llm import FakeLLM, GroqLLM, LLMError, build_llm


def groq_with(handler):
    """A GroqLLM whose network is replaced by `handler`. Returns (llm, list of requests sent)."""
    sent = []

    def record(request: httpx.Request):
        sent.append(request)
        return handler(request)

    return GroqLLM("test-key", transport=httpx.MockTransport(record)), sent


def reply(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_request_shape():
    llm, sent = groq_with(lambda r: reply('{"status":"not_covered"}'))
    assert llm.complete_json("sys", "usr", PRIMARY_MODEL, 2.0) == '{"status":"not_covered"}'
    body = json.loads(sent[0].content)
    assert sent[0].url.path.endswith("/chat/completions")
    assert sent[0].headers["Authorization"] == "Bearer test-key"
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
    assert body["reasoning_effort"] == "none"                         # Qwen: no hidden reasoning
    assert [m["role"] for m in body["messages"]] == ["system", "user"]

    llm.complete_json("sys", "usr", FALLBACK_MODEL, 2.0)
    assert json.loads(sent[1].content)["reasoning_effort"] == "low"   # gpt-oss: kept low


@pytest.mark.parametrize("response, kind", [
    (httpx.Response(429), "rate_limited"),
    (httpx.Response(503), "unavailable"),
    (httpx.Response(400, text="bad model"), "bad_request"),
    (httpx.Response(200, json={"unexpected": True}), "unavailable"),
])
def test_http_errors_become_llm_errors(response, kind):
    llm, _ = groq_with(lambda r: response)
    with pytest.raises(LLMError) as err:
        llm.complete_json("s", "u", "m", 1.0)
    assert err.value.kind == kind


def test_timeout_and_network_errors():
    def slow(r):
        raise httpx.ReadTimeout("slow", request=r)

    def down(r):
        raise httpx.ConnectError("down", request=r)

    with pytest.raises(LLMError) as err:
        groq_with(slow)[0].complete_json("s", "u", "m", 1.0)
    assert err.value.kind == "timeout"
    with pytest.raises(LLMError) as err:
        groq_with(down)[0].complete_json("s", "u", "m", 1.0)
    assert err.value.kind == "unavailable"


def test_a_call_is_capped_in_total_time():
    """A server that never finishes: the total-time limit stops it at timeout_s."""
    def stuck(r):
        time.sleep(5)
        return reply("{}")

    started = time.perf_counter()
    with pytest.raises(LLMError) as err:
        groq_with(stuck)[0].complete_json("s", "u", PRIMARY_MODEL, timeout_s=0.3)
    assert err.value.kind == "timeout" and time.perf_counter() - started < 1.0


# -------------------------------------------------------------------- config --

def test_no_key_means_offline_mode(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    get_settings.cache_clear()
    assert get_settings().llm_provider == "fake" and isinstance(build_llm(get_settings()), FakeLLM)


def test_a_key_selects_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    get_settings.cache_clear()
    assert get_settings().llm_provider == "groq" and isinstance(build_llm(get_settings()), GroqLLM)


def test_groq_without_a_key_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        get_settings()


# ---------------------------------------------------------------------- fake --

def test_fake_answers_from_s1_and_echoes_rewrites():
    answer_prompt = '<excerpt id="S1" section="4.1">\nNight > Escort\nEscort after 8 PM. More text.\n</excerpt>'
    assert json.loads(FakeLLM().complete_json("s", answer_prompt, "m", 1.0)) == {
        "status": "answered", "answer": "Escort after 8 PM.", "citations": ["S1"]}
    rewrite_prompt = "<outline>\n1.1 X\n</outline>\n\n<question>\nCan I?\n</question>"
    assert json.loads(FakeLLM().complete_json("s", rewrite_prompt, "m", 1.0)) == {"queries": ["Can I?"]}


def test_fake_script_and_timeout():
    fake = FakeLLM(script=["raw text", LLMError("rate_limited", "x")])
    assert fake.complete_json("s", "u", "m", 1.0) == "raw text"
    with pytest.raises(LLMError):
        fake.complete_json("s", "u", "m", 1.0)
    with pytest.raises(LLMError) as err:
        FakeLLM(delay_s=0.2).complete_json("s", "u", "m", timeout_s=0.05)
    assert err.value.kind == "timeout"
