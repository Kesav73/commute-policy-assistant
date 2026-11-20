"""JWT: the caller's identity comes from a token we signed, and nothing else."""
import stat
import time

import jwt
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth import Principal, current_principal, require_admin
from app.config import ConfigError, get_settings
from app.main import http_error
from tests.conftest import TEST_SECRET, bearer

# A tiny app with two protected routes, used only by these tests.
probe = FastAPI()
probe.add_exception_handler(HTTPException, http_error)   # same error shape as the app


@probe.get("/whoami")
def whoami(principal: Principal = Depends(current_principal)):
    return {"employee_id": principal.employee_id, "company_id": principal.company_id, "role": principal.role}


@probe.get("/admin-only")
def admin_only(principal: Principal = Depends(require_admin)):
    return {"ok": True}


@pytest.fixture
def probe_client():
    return TestClient(probe)


def forge(overrides: dict | None = None, secret: str = TEST_SECRET, drop: str | None = None) -> str:
    """A token with chosen claims, signed with `secret`."""
    now = int(time.time())
    claims = {"sub": "E-ACME-001", "company_id": "acme", "role": "employee", "iat": now, "exp": now + 600}
    claims.update(overrides or {})
    if drop:
        claims.pop(drop)
    return jwt.encode(claims, secret, algorithm="HS256")


# --------------------------------------------------------------------- issue --

def test_token_carries_company_and_role_from_the_directory(probe_client, token_for):
    r = probe_client.get("/whoami", headers=bearer(token_for("E-BOLT-001")))
    assert r.json() == {"employee_id": "E-BOLT-001", "company_id": "bolt", "role": "employee"}


def test_token_response_shape(client):
    body = client.post("/auth/token", json={"employee_id": "E-ACME-001"}).json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600 and body["access_token"]


def test_unknown_employee_gets_no_token(client):
    r = client.post("/auth/token", json={"employee_id": "E-NOBODY"})
    assert r.status_code == 404 and r.json()["error"] == "unknown_employee"


def test_client_cannot_choose_company_at_login(client):
    r = client.post("/auth/token", json={"employee_id": "E-ACME-001", "company_id": "bolt"})
    assert r.status_code == 422


def test_query_param_and_header_cannot_change_company(probe_client, token_for):
    r = probe_client.get("/whoami?company_id=bolt",
                         headers={**bearer(token_for("E-ACME-001")), "X-Company-Id": "bolt"})
    assert r.json()["company_id"] == "acme"


# -------------------------------------------------------------------- verify --

def test_missing_token(probe_client):
    r = probe_client.get("/whoami")
    assert r.status_code == 401 and "sign in" in r.json()["message"].lower()


@pytest.mark.parametrize("token", [
    "not-a-jwt",
    forge(secret="another-secret-that-is-long-enough-123"),            # signed by someone else
    forge({"company_id": "evilcorp"}),                                 # signed, but no such company
    forge({"role": "superuser"}),                                      # signed, but no such role
])
def test_bad_tokens_are_refused(probe_client, token):
    assert probe_client.get("/whoami", headers=bearer(token)).status_code == 401


def test_tampered_payload_is_refused(probe_client, token_for):
    header, _, sig = token_for("E-ACME-001").split(".")
    bolt_payload = forge({"company_id": "bolt"}).split(".")[1]        # swap in another payload
    assert probe_client.get("/whoami", headers=bearer(f"{header}.{bolt_payload}.{sig}")).status_code == 401


def test_alg_none_token_is_refused(probe_client):
    unsigned = jwt.encode({"sub": "E-ACME-001", "company_id": "acme", "role": "admin",
                           "iat": int(time.time()), "exp": int(time.time()) + 600}, None, algorithm="none")
    assert probe_client.get("/whoami", headers=bearer(unsigned)).status_code == 401


def test_expired_token_has_clear_message(probe_client):
    now = int(time.time())
    r = probe_client.get("/whoami", headers=bearer(forge({"iat": now - 7200, "exp": now - 3600})))
    assert r.status_code == 401 and "expired" in r.json()["message"]


@pytest.mark.parametrize("claim", ["sub", "company_id", "role", "exp"])
def test_token_missing_a_required_claim_is_refused(probe_client, claim):
    assert probe_client.get("/whoami", headers=bearer(forge(drop=claim))).status_code == 401


def test_admin_route_needs_admin_role(probe_client, token_for):
    assert probe_client.get("/admin-only", headers=bearer(token_for("E-ACME-001"))).status_code == 403
    assert probe_client.get("/admin-only", headers=bearer(token_for("A-ACME-ADM"))).status_code == 200


# ---------------------------------------------------------------- the secret --

def test_secret_is_generated_once_and_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "fresh"))
    get_settings.cache_clear()
    first = get_settings().jwt_secret
    secret_file = tmp_path / "fresh" / "jwt_secret"
    assert len(first) >= 32 and stat.S_IMODE(secret_file.stat().st_mode) == 0o600   # owner-only
    get_settings.cache_clear()                                                       # "restart"
    assert get_settings().jwt_secret == first


def test_short_secret_is_refused(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "too-short")
    get_settings.cache_clear()
    with pytest.raises(ConfigError):
        get_settings()
