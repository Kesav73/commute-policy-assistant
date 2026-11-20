"""Settings, read once from environment variables (and from .env if present).

Every setting has a working default, so `docker compose up` needs no setup:
  * no GROQ_API_KEY  -> offline mode with a fake LLM
  * no JWT_SECRET    -> a random secret is generated once and saved in the data folder
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

# Load .env into the environment. Variables that are already set win, so Docker
# and tests can always override what's in the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

# The two LLMs (Groq free tier). The fallback is only used when the primary model
# fails, so the service keeps answering if one model is down.
PRIMARY_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODEL = "openai/gpt-oss-120b"

MIN_SECRET_LENGTH = 32


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    # identity
    jwt_secret: str
    token_ttl_minutes: int
    employees_file: Path
    # files
    data_dir: Path
    policies_dir: Path
    # LLM
    llm_provider: str            # "groq" or "fake" (offline)
    llm_api_key: str
    # retrieval: "not covered" when even the closest chunk is farther than this
    retrieval_max_distance: float
    # time limits, in seconds (a request must never hang)
    retrieval_timeout_s: float   # one retrieval (embedding + search); normally ~30-90 ms
    rewrite_timeout_s: float     # LLM call 1, so the answer call keeps its time
    llm_timeout_s: float         # any single LLM call
    answer_deadline_s: float     # the whole pipeline, kept under the 3 s target
    request_timeout_s: float     # last-resort limit on a whole /ask request
    # answer cache (0 entries = off)
    answer_cache_size: int
    answer_cache_ttl_s: float


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _jwt_secret(data_dir: Path) -> str:
    """JWT_SECRET from the environment, or a secret generated once and saved to disk."""
    secret = _env("JWT_SECRET", "")
    if secret:
        if len(secret) < MIN_SECRET_LENGTH:
            raise ConfigError(
                f"JWT_SECRET is too short ({len(secret)} chars); use at least {MIN_SECRET_LENGTH}. "
                'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"')
        return secret

    path = data_dir / "jwt_secret"
    if path.exists():
        return path.read_text().strip()

    # First start without JWT_SECRET: create one. The file is readable only by its
    # owner, and O_EXCL makes sure two processes starting together can't both create it.
    data_dir.mkdir(parents=True, exist_ok=True)
    new_secret = secrets.token_urlsafe(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text().strip()
    with os.fdopen(fd, "w") as f:
        f.write(new_secret)
    log.warning("JWT_SECRET not set: generated a new one and saved it to %s", path)
    return new_secret


def _llm_provider() -> tuple[str, str]:
    """(provider, api key). Groq when a key is set, otherwise offline mode.
    LLM_PROVIDER=fake forces offline mode even with a key (tests use this)."""
    key = _env("GROQ_API_KEY", "")
    provider = _env("LLM_PROVIDER", "").lower() or ("groq" if key else "fake")
    if provider not in ("groq", "fake"):
        raise ConfigError(f"LLM_PROVIDER must be 'groq' or 'fake', got {provider!r}")
    if provider == "groq" and not key:
        raise ConfigError("LLM_PROVIDER=groq needs GROQ_API_KEY")
    return provider, key


@lru_cache
def get_settings() -> Settings:
    """Built once per process. Tests call get_settings.cache_clear() after changing env."""
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    provider, api_key = _llm_provider()
    return Settings(
        jwt_secret=_jwt_secret(data_dir),
        token_ttl_minutes=int(_env("TOKEN_TTL_MINUTES", "60")),
        employees_file=Path(_env("EMPLOYEES_FILE", str(PROJECT_ROOT / "sample_data" / "employees.json"))),
        data_dir=data_dir,
        policies_dir=Path(_env("POLICIES_DIR", str(PROJECT_ROOT / "policies"))),
        llm_provider=provider,
        llm_api_key=api_key,
        # 0.50 is the tightest value that blocks none of the 75 answerable evaluation
        # questions (the worst real one is 0.489). See EVALUATION.md.
        retrieval_max_distance=float(_env("RETRIEVAL_MAX_DISTANCE", "0.50")),
        retrieval_timeout_s=float(_env("RETRIEVAL_TIMEOUT_S", "1.0")),
        rewrite_timeout_s=float(_env("REWRITE_TIMEOUT_S", "1.2")),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S", "2.0")),
        answer_deadline_s=float(_env("ANSWER_DEADLINE_S", "2.8")),
        request_timeout_s=float(_env("REQUEST_TIMEOUT_S", "3.0")),
        answer_cache_size=int(_env("ANSWER_CACHE_SIZE", "1000")),
        answer_cache_ttl_s=float(_env("ANSWER_CACHE_TTL_S", "21600")),
    )
