"""The only module that talks to an LLM.

    OpenAICompatLLM  Groq through its OpenAI-compatible chat endpoint (plain httpx)
    FakeLLM          deterministic stand-in: tests, and offline mode when no key is set

Every failure becomes one of a few LLMError types, so the pipeline can decide
"fall back" or "tell the user to try again" without knowing HTTP details.
"""
import json
import re
import time
from collections import deque
from typing import Protocol

import httpx

from app.config import Settings

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
}

# Per-model request settings, matched by prefix. Reasoning is switched off / kept low:
# the task is read-and-cite, and hidden reasoning tokens only add latency
# (gpt-oss-20b at effort=medium doubled p95 in eval/bench_llm.py).
MODEL_PARAMS = {
    "qwen/": {"reasoning_effort": "none"},
    "openai/gpt-oss": {"reasoning_effort": "low"},
}

MAX_OUTPUT_TOKENS = 400


# ------------------------------------------------------------------ errors --

class LLMError(Exception):
    """Base class. `kind` is a short label used in logs and eval notes."""
    kind = "error"


class LLMTimeout(LLMError):
    kind = "timeout"


class LLMRateLimited(LLMError):
    kind = "rate_limited"


class LLMUnavailable(LLMError):
    """Network failure or provider 5xx."""
    kind = "unavailable"


class LLMRequestError(LLMError):
    """Provider rejected the request (4xx other than 429): a bug or config problem."""
    kind = "bad_request"


# --------------------------------------------------------------- interface --

class LLM(Protocol):
    provider: str

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        """Return the model's raw text (expected to be a JSON object). Raise LLMError on failure."""
        ...


# ------------------------------------------------------------------- real --

class OpenAICompatLLM:
    def __init__(self, provider: str, api_key: str, transport: httpx.BaseTransport | None = None):
        self.provider = provider
        self._client = httpx.Client(
            base_url=BASE_URLS[provider],
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,  # tests inject a mock transport here
        )

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
        }
        for prefix, params in MODEL_PARAMS.items():
            if model.startswith(prefix):
                body.update(params)

        try:
            resp = self._client.post("/chat/completions", json=body,
                                     timeout=httpx.Timeout(timeout_s))
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"{model}: no response within {timeout_s:.1f}s") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"{model}: {type(e).__name__}") from e

        if resp.status_code == 429:
            raise LLMRateLimited(f"{model}: rate limited")
        if resp.status_code >= 500:
            raise LLMUnavailable(f"{model}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            # Never log the full body blindly; keep it short.
            raise LLMRequestError(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")

        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMUnavailable(f"{model}: unexpected response shape") from e


# ------------------------------------------------------------------- fake --

_EXCERPT_RE = re.compile(r'<excerpt id="(S\d)"[^>]*>\n(.*?)\n</excerpt>', re.S)


class FakeLLM:
    """Deterministic LLM for tests and offline mode.

    Default behaviour: answer with the first sentence of excerpt S1 and cite S1;
    if there are no excerpts, say not_covered.

    Tests can script exact behaviour with `script`: a queue of items, each either
    a string (returned as the raw model output) or an exception instance (raised).
    `delay_s` simulates a slow model. Every call is recorded in `calls`.
    """
    provider = "fake"

    def __init__(self, script: list | None = None, delay_s: float = 0.0):
        self.script = deque(script or [])
        self.delay_s = delay_s
        self.calls: list[dict] = []

    def complete_json(self, system: str, user: str, model: str, timeout_s: float) -> str:
        self.calls.append({"model": model, "system": system, "user": user, "timeout_s": timeout_s})
        if self.delay_s:
            if self.delay_s > timeout_s:
                time.sleep(timeout_s)
                raise LLMTimeout(f"{model}: fake timeout")
            time.sleep(self.delay_s)
        if self.script:
            item = self.script.popleft()
            if isinstance(item, BaseException):
                raise item
            return item

        excerpts = _EXCERPT_RE.findall(user)
        if not excerpts:
            return json.dumps({"status": "not_covered", "answer": "", "citations": []})
        label, text = excerpts[0]
        body = text.split("\n", 1)[-1]
        first_sentence = body.split(". ")[0].rstrip(".") + "."
        return json.dumps({"status": "answered", "answer": first_sentence, "citations": [label]})


# ----------------------------------------------------------------- factory --

def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()
    return OpenAICompatLLM(settings.llm_provider, settings.llm_api_key)
