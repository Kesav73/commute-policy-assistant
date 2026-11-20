"""Settings, read from environment variables.

JWT secret rules (HLD §7.3):
  1. If JWT_SECRET is set, use it - but refuse anything shorter than 32 characters.
  2. If not set, generate a random one ONCE, save it to <DATA_DIR>/jwt_secret
     (git-ignored, owner-only permissions) and reuse it on every restart.
This keeps `docker compose up` working with zero setup, without a secret in git.
"""
import logging
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("policy_assistant")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read .env (git-ignored) into the environment. Real environment variables win,
# so Docker / CI / tests can always override what's in the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)
MIN_SECRET_LENGTH = 32


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    jwt_secret: str
    jwt_algorithm: str
    token_ttl_minutes: int
    employees_file: Path
    data_dir: Path
    policies_dir: Path
    retrieval_max_distance: float
    retrieval_mode: str        # "vector" | "hybrid"
    query_rewrite: bool        # LLM call 1 rewrites the question into search queries
    llm_provider: str          # "groq" | "fake"
    llm_model: str             # primary
    llm_fallback_model: str    # used for the 2nd (and last) call
    llm_api_key: str
    llm_timeout_s: float       # per call
    answer_deadline_s: float   # whole /ask budget, kept under the 3 s p95 target
    rewrite_timeout_s: float   # cap for the rewrite call, so the answer call keeps its time


def _load_or_create_secret(data_dir: Path) -> str:
    from_env = os.getenv("JWT_SECRET", "").strip()
    if from_env:
        if len(from_env) < MIN_SECRET_LENGTH:
            raise ConfigError(
                f"JWT_SECRET is too short ({len(from_env)} chars); use at least "
                f"{MIN_SECRET_LENGTH}. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return from_env

    path = data_dir / "jwt_secret"
    if path.exists():
        return path.read_text().strip()

    data_dir.mkdir(parents=True, exist_ok=True)
    new_secret = secrets.token_urlsafe(32)  # 32 random bytes -> 43 URL-safe chars
    try:
        # O_EXCL: if two processes start at once, only one creates the file.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text().strip()
    with os.fdopen(fd, "w") as f:
        f.write(new_secret)
    log.warning("JWT_SECRET not set: generated a new one and saved it to %s", path)
    return new_secret


@lru_cache
def get_settings() -> Settings:
    """Built once per process. Tests call get_settings.cache_clear() after changing env."""
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    return Settings(
        jwt_secret=_load_or_create_secret(data_dir),
        jwt_algorithm="HS256",
        token_ttl_minutes=int(os.getenv("TOKEN_TTL_MINUTES", "60")),
        employees_file=Path(
            os.getenv("EMPLOYEES_FILE", PROJECT_ROOT / "sample_data" / "employees.json")
        ),
        data_dir=data_dir,
        policies_dir=Path(os.getenv("POLICIES_DIR", PROJECT_ROOT / "policies")),
        # Evidence gate: if even the best chunk is further than this (cosine distance),
        # answer "not covered" without calling the LLM. Conservative on purpose: only
        # clearly unrelated questions are cut here; borderline ones go to the LLM.
        # 0.50 = the tightest value that blocks none of the 75 answerable gold questions
        # with nomic-embed-text-v1.5 (worst real one: 0.489). See docs/journey/08.
        retrieval_max_distance=float(os.getenv("RETRIEVAL_MAX_DISTANCE", "0.50")),
        # hybrid (default) = embeddings + stemmed BM25 keywords, fused with weighted RRF;
        # vector = embeddings only. See docs/journey/09.
        retrieval_mode=_retrieval_mode(),
        # on (default): call 1 rewrites the question (docs/journey/13), call 2 answers;
        # off: call 1 answers, call 2 is the fallback model.
        query_rewrite=os.getenv("QUERY_REWRITE", "on").strip().lower() not in ("off", "0", "false", "no"),
        **_llm_settings(),
    )


# Defaults chosen from a latency benchmark on Groq: Qwen was fastest and resisted the planted
# injection; gpt-oss-120b is a different model with its own rate limits.
LLM_DEFAULTS = {
    "groq": ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "fake": ("fake-primary", "fake-fallback", ""),
}


def _retrieval_mode() -> str:
    mode = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
    if mode not in ("vector", "hybrid"):
        raise ConfigError(f"RETRIEVAL_MODE must be 'vector' or 'hybrid', got {mode!r}")
    return mode


def _llm_settings() -> dict:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        # No explicit choice: use Groq if a key exists, otherwise offline mode.
        provider = "groq" if os.getenv("GROQ_API_KEY", "").strip() else "fake"
    if provider not in LLM_DEFAULTS:
        raise ConfigError(f"LLM_PROVIDER must be one of {sorted(LLM_DEFAULTS)}, got {provider!r}")

    model, fallback, key_var = LLM_DEFAULTS[provider]
    key = os.getenv(key_var, "").strip() if key_var else ""
    if provider != "fake" and not key:
        raise ConfigError(f"LLM_PROVIDER={provider} needs {key_var} to be set")
    return {
        "llm_provider": provider,
        "llm_model": os.getenv("LLM_MODEL", "").strip() or model,
        "llm_fallback_model": os.getenv("LLM_FALLBACK_MODEL", "").strip() or fallback,
        "llm_api_key": key,
        "llm_timeout_s": float(os.getenv("LLM_TIMEOUT_S", "2.0")),
        "answer_deadline_s": float(os.getenv("ANSWER_DEADLINE_S", "2.8")),
        "rewrite_timeout_s": float(os.getenv("REWRITE_TIMEOUT_S", "1.2")),
    }
