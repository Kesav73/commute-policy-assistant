"""JWT: a mock issuer and a real verifier.

ISSUE  (simulated - stands in for the company's SSO):
    employee_id -> look up company/role in the directory -> sign a token.
VERIFY (the real requirement, PS.md R4/R5):
    every protected endpoint gets the caller's identity ONLY via current_principal(),
    which trusts nothing but a token signed with our secret.
"""
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings

ROLES = ("employee", "admin")
REQUIRED_CLAIMS = ["sub", "company_id", "role", "iat", "exp"]


@dataclass(frozen=True)
class Principal:
    """Who is calling. Only ever built from a verified token."""
    employee_id: str
    company_id: str
    role: str


class UnknownEmployeeError(Exception):
    pass


# ---------------------------------------------------------------- directory --

@lru_cache
def _load_directory(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {e["employee_id"]: e for e in data["employees"]}


def directory() -> dict[str, dict]:
    return _load_directory(get_settings().employees_file)


def known_companies() -> set[str]:
    return {e["company_id"] for e in directory().values()}


# ------------------------------------------------------------------- issue --

def issue_token(employee_id: str) -> tuple[str, int]:
    """Return (token, expires_in_seconds). Company comes from the directory, not the caller."""
    employee = directory().get(employee_id)
    if employee is None:
        raise UnknownEmployeeError(employee_id)

    s = get_settings()
    now = int(time.time())
    ttl = s.token_ttl_minutes * 60
    claims = {
        "sub": employee["employee_id"],
        "company_id": employee["company_id"],
        "role": employee["role"],
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(claims, s.jwt_secret, algorithm=s.jwt_algorithm), ttl


# ------------------------------------------------------------------ verify --

def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "unauthorized", "message": message},
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_token(token: str) -> Principal:
    """Signature, expiry, required claims, known company, known role. Any doubt -> 401."""
    s = get_settings()
    try:
        claims = jwt.decode(
            token,
            s.jwt_secret,
            algorithms=[s.jwt_algorithm],  # pinned: blocks "alg: none" and algorithm swaps
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Your session has expired. Please sign in again.")
    except jwt.InvalidTokenError:
        raise _unauthorized("Invalid sign-in token. Please sign in again.")

    company_id, role, sub = claims["company_id"], claims["role"], claims["sub"]
    # Defence in depth: even a correctly signed token must name a real company and role.
    if not isinstance(company_id, str) or company_id not in known_companies():
        raise _unauthorized("Invalid sign-in token. Please sign in again.")
    if role not in ROLES or not isinstance(sub, str):
        raise _unauthorized("Invalid sign-in token. Please sign in again.")

    return Principal(employee_id=sub, company_id=company_id, role=role)


_bearer = HTTPBearer(auto_error=False)


def current_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """FastAPI dependency. The ONLY way an endpoint learns who is calling."""
    if creds is None:
        raise _unauthorized("Please sign in: a Bearer token is required.")
    return decode_token(creds.credentials)


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    """For admin-only endpoints (policy upload, later)."""
    if principal.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "forbidden", "message": "This action needs an admin account."},
        )
    return principal
