"""Shared test setup. Tests never call a real LLM: no API key and no quota needed."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.store import PolicyStore
from app.versioning import ingest_all

TEST_SECRET = "test-secret-0123456789-abcdefghijklmnop"   # at least 32 characters
ROOT = Path(__file__).resolve().parent.parent
POLICIES = ROOT / "policies"
ACME_V2 = (ROOT / "sample_data" / "policy_updates" / "acme_v2.md").read_text()    # escort 8 PM -> 9 PM
ESCORT_Q = "Does a trained security escort travel with women on night drops?"


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
    """Every test: a known secret, its own empty data folder, and the fake LLM."""
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> PolicyStore:
    """A store with the three seed policies published (v1 each). Shared by a test module."""
    s = PolicyStore(tmp_path_factory.mktemp("store"))
    ingest_all(s, POLICIES)
    return s


@pytest.fixture
def fresh_store(tmp_path) -> PolicyStore:
    """Like `store`, but new for each test (for tests that change policies)."""
    s = PolicyStore(tmp_path / "fresh")
    ingest_all(s, POLICIES)
    return s


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def token_for(client):
    """token_for("E-ACME-001") -> a real token from our issuer endpoint."""
    def issue(employee_id: str) -> str:
        r = client.post("/auth/token", json={"employee_id": employee_id})
        assert r.status_code == 200, r.text
        return r.json()["access_token"]
    return issue


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
