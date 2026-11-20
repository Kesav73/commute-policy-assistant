"""The only module that talks to an LLM.

    GroqLLM   the real model, through Groq's OpenAI-compatible chat API (plain httpx)
    FakeLLM   a deterministic stand-in for tests and offline mode (no API key)

Both have the same method, complete_json(system, user, model, timeout_s) -> str.
Every failure is raised as LLMError with a short `kind` ("timeout", "rate_limited",
"unavailable", "bad_request"), so the pipeline never deals with HTTP details.
"""
import json
import re
import time
from collections import deque

import httpx

from app.config import Settings
from app.timeouts import StageTimeout, run_with_timeout

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_OUTPUT_TOKENS = 400

# Hidden "reasoning" tokens only added latency in our benchmark, so they are
# switched off (Qwen) or kept low (gpt-oss).
REASONING = {"qwen/": "none", "openai/gpt-oss": "low"}


class LLMError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class GroqLLM:
    provider = "groq"

    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None):
        # `transport` lets tests replace the network with a fake server.
        self._client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,                              # same input -> same output
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},     # the API guarantees valid JSON
        }
        for prefix, effort in REASONING.items():
            if model.startswith(prefix):
                body["reasoning_effort"] = effort

        try:
            # httpx's own timeout applies to each network step separately; the outer
            # limit caps the TOTAL time of the call.
            resp = run_with_timeout(self._client.post, timeout_s, GROQ_URL,
                                    json=body, timeout=httpx.Timeout(timeout_s))
        except (httpx.TimeoutException, StageTimeout) as e:
            raise LLMError("timeout", f"{model}: no response within {timeout_s:.1f}s") from e
        except httpx.HTTPError as e:
            raise LLMError("unavailable", f"{model}: {type(e).__name__}") from e

        if resp.status_code == 429:
            raise LLMError("rate_limited", f"{model}: rate limited")
        if resp.status_code >= 500:
            raise LLMError("unavailable", f"{model}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise LLMError("bad_request", f"{model}: HTTP {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError("unavailable", f"{model}: unexpected response") from e


_EXCERPT_RE = re.compile(r'<excerpt id="(S\d)"[^>]*>\n(.*?)\n</excerpt>', re.S)


class FakeLLM:
    """Deterministic stand-in for the real model.

    By default:
      * a rewrite prompt -> the question itself is the only search query
      * an answer prompt -> the first sentence of excerpt S1, citing S1
    Tests can pass `script`: a list of replies (strings) or errors (raised), used in order.
    `delay_s` makes it slow. Every call is recorded in `calls`.
    """
    provider = "fake"

    def __init__(self, script: list | None = None, delay_s: float = 0.0):
        self.script = deque(script or [])
        self.delay_s = delay_s
        self.calls: list[dict] = []

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        self.calls.append({"model": model, "system": system, "user": user, "timeout_s": timeout_s})
        if self.delay_s > timeout_s:
            time.sleep(timeout_s)
            raise LLMError("timeout", f"{model}: fake timeout")
        time.sleep(self.delay_s)

        if self.script:
            reply = self.script.popleft()
            if isinstance(reply, BaseException):
                raise reply
            return reply

        if "<outline>" in user:                       # rewrite prompt
            question = user.split("<question>")[1].split("</question>")[0].strip()
            return json.dumps({"queries": [question]})
        excerpts = _EXCERPT_RE.findall(user)          # answer prompt
        if not excerpts:
            return json.dumps({"status": "not_covered", "answer": "", "citations": []})
        label, text = excerpts[0]
        body = text.split("\n", 1)[-1]
        first_sentence = body.split(". ")[0].rstrip(".") + "."
        return json.dumps({"status": "answered", "answer": first_sentence, "citations": [label]})


def build_llm(settings: Settings):
    if settings.llm_provider == "fake":
        return FakeLLM()
    return GroqLLM(settings.llm_api_key)
