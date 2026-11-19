"""JWT: identity comes from a token we signed, and nothing else (PS.md R4, R5)."""
import stat
import time

import jwt
import pytest

from app.config import ConfigError, get_settings
from tests.conftest import TEST_SECRET, bearer


def _forge(overrides: dict | None = None, secret: str = TEST_SECRET, drop: str | None = None) -> str:
    now = int(time.time())
    claims = {"sub": "E-ACME-001", "company_id": "acme", "role": "employee",
              "iat": now, "exp": now + 600, **(overrides or {})}
    if drop:
        claims.pop(drop)
    return jwt.encode(claims, secret, algorithm="HS256")


# ------------------------------------------------------------------ issue --

def test_token_carries_company_from_directory(client, token_for):
    r = client.get("/me", headers=bearer(token_for("E-BOLT-001")))
    assert r.status_code == 200
    assert r.json() == {"employee_id": "E-BOLT-001", "company_id": "bolt", "role": "employee"}


def test_token_response_shape(client):
    r = client.post("/auth/token", json={"employee_id": "E-ACME-001"})
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 3600


def test_unknown_employee_gets_no_token(client):
    assert client.post("/auth/token", json={"employee_id": "E-NOBODY"}).status_code == 404


def test_client_cannot_choose_company_at_login(client):
    r = client.post("/auth/token", json={"employee_id": "E-ACME-001", "company_id": "bolt"})
    assert r.status_code == 422


# ----------------------------------------------------------------- verify --

def test_query_param_and_header_cannot_change_company(client, token_for):
    r = client.get("/me?company_id=bolt",
                   headers={**bearer(token_for("E-ACME-001")), "X-Company-Id": "bolt"})
    assert r.json()["company_id"] == "acme"


def test_missing_token(client):
    r = client.get("/me")
    assert r.status_code == 401
    assert "sign in" in r.json()["message"].lower()


def test_garbage_token(client):
    assert client.get("/me", headers=bearer("not-a-jwt")).status_code == 401


def test_token_signed_with_other_secret(client):
    forged = _forge({"company_id": "bolt"}, secret="attacker-secret-that-is-long-enough-32b")
    assert client.get("/me", headers=bearer(forged)).status_code == 401


def test_tampered_payload(client, token_for):
    # Real Acme token, payload swapped for a Bolt one, original signature kept.
    header, _, sig = token_for("E-ACME-001").split(".")
    _, bolt_payload, _ = _forge({"company_id": "bolt"}).split(".")
    assert client.get("/me", headers=bearer(f"{header}.{bolt_payload}.{sig}")).status_code == 401


def test_alg_none_token(client):
    now = int(time.time())
    unsigned = jwt.encode({"sub": "x", "company_id": "bolt", "role": "employee",
                           "iat": now, "exp": now + 600}, key=None, algorithm="none")
    assert client.get("/me", headers=bearer(unsigned)).status_code == 401


def test_expired_token_has_clear_message(client):
    now = int(time.time())
    r = client.get("/me", headers=bearer(_forge({"iat": now - 7200, "exp": now - 3600})))
    assert r.status_code == 401
    assert "expired" in r.json()["message"].lower()


@pytest.mark.parametrize("claim", ["sub", "company_id", "role", "exp", "iat"])
def test_token_missing_required_claim(client, claim):
    assert client.get("/me", headers=bearer(_forge(drop=claim))).status_code == 401


def test_signed_token_for_unknown_company(client):
    assert client.get("/me", headers=bearer(_forge({"company_id": "evilcorp"}))).status_code == 401


def test_signed_token_with_unknown_role(client):
    assert client.get("/me", headers=bearer(_forge({"role": "superuser"}))).status_code == 401


# ----------------------------------------------------------- secret key --

def test_secret_auto_generated_once_and_persisted(monkeypatch, tmp_path):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "fresh"))
    get_settings.cache_clear()
    first = get_settings().jwt_secret

    secret_file = tmp_path / "fresh" / "jwt_secret"
    assert secret_file.exists()
    assert len(first) >= 32
    assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600  # owner-only

    get_settings.cache_clear()  # simulate a restart
    assert get_settings().jwt_secret == first


def test_short_secret_is_refused(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")
    get_settings.cache_clear()
    with pytest.raises(ConfigError):
        get_settings()
