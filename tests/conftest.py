import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

TEST_SECRET = "test-secret-0123456789-abcdefghijklmnop"  # >= 32 chars


@pytest.fixture(autouse=True)
def _test_env(monkeypatch, tmp_path):
    """Every test gets a known secret and its own empty data dir."""
    monkeypatch.setenv("JWT_SECRET", TEST_SECRET)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    # Tests never call a real LLM: no quota used, no key needed, deterministic (PS.md G2).
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    for var in ("LLM_MODEL", "LLM_FALLBACK_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def token_for(client):
    """token_for("E-ACME-001") -> a real token from our issuer endpoint."""
    def _issue(employee_id: str) -> str:
        r = client.post("/auth/token", json={"employee_id": employee_id})
        assert r.status_code == 200, r.text
        return r.json()["access_token"]
    return _issue


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
