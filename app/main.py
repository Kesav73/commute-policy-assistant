"""HTTP layer. Thin on purpose: endpoints wire things together, logic lives in modules.

Endpoints
    POST /auth/token   mock identity provider (stands in for company SSO)
    POST /ask          the product: question in, cited answer out
    GET  /me           shows what the server believes about the caller (demo helper)
    GET  /health       is it up, which LLM mode, how many chunks per company

Every error has one shape: {"error": "<code>", "message": "<plain text for a human>"}.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from app import messages
from app.auth import Principal, UnknownEmployeeError, current_principal, issue_token
from app.config import get_settings
from app.ingest import ingest_all
from app.llm import build_llm
from app.pipeline import ask
from app.store import chunk_counts, open_store

log = logging.getLogger("policy_assistant")

QUESTION_MIN, QUESTION_MAX = 3, 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything here runs BEFORE the first request is accepted, and any error stops
    # startup: better no service than one answering from half-loaded policies.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log.setLevel(logging.INFO)
    s = get_settings()                       # fails fast on bad JWT_SECRET / LLM config
    app.state.store = open_store(s.data_dir)
    counts = ingest_all(app.state.store, s.policies_dir)
    app.state.llm = build_llm(s)
    log.info("ready: chunks=%s llm=%s (%s, fallback %s)",
             counts, s.llm_provider, s.llm_model, s.llm_fallback_model)
    yield


app = FastAPI(title="Commute Policy Assistant", version="0.1.0", lifespan=lifespan)


# ----------------------------------------------------------------- models --

class StrictModel(BaseModel):
    # Unknown fields are rejected, so a client can't sneak in {"company_id": "bolt"}.
    model_config = ConfigDict(extra="forbid")


class TokenRequest(StrictModel):
    employee_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class AskRequest(StrictModel):
    """Only a question. There is deliberately NO company field: identity is the token's job."""
    question: str

    @field_validator("question")
    @classmethod
    def _check_question(cls, v: str) -> str:
        v = v.strip()
        if "\x00" in v or not (QUESTION_MIN <= len(v) <= QUESTION_MAX):
            raise ValueError(messages.QUESTION_LENGTH.format(min=QUESTION_MIN, max=QUESTION_MAX))
        return v


class CitationOut(BaseModel):
    doc_id: str
    title: str
    section: str
    section_title: str


class AskResponse(BaseModel):
    status: str                  # "answered" | "not_covered"
    answer: str
    citations: list[CitationOut]
    request_id: str


# ----------------------------------------------------------------- errors --

def error(status: int, code: str, message: str, headers: dict | None = None, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": code, "message": message, **extra})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Flatten our HTTPExceptions to the one error shape."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return error(exc.status_code, exc.detail["error"], exc.detail["message"], exc.headers)
    return error(exc.status_code, "error", str(exc.detail), exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Replace FastAPI's technical 422 with one sentence a person can act on."""
    first = exc.errors()[0] if exc.errors() else {}
    field = str(first.get("loc", ["", ""])[-1])
    kind = first.get("type", "")
    if kind == "extra_forbidden":
        msg = f"Unexpected field '{field}'. Send only the documented fields."
    elif kind == "missing":
        msg = f"Missing field '{field}'."
    elif kind == "value_error":
        msg = str(first.get("ctx", {}).get("error", first.get("msg", "")))
    else:
        msg = "The request body is not valid JSON in the expected format."
    return error(422, "invalid_request", msg)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence (PS.md R10): log details server-side, never show a stack trace."""
    log.exception("Unhandled error on %s", request.url.path)
    return error(500, "internal_error", messages.INTERNAL_ERROR)


# -------------------------------------------------------------- endpoints --

@app.get("/health")
def health(request: Request) -> dict:
    s = get_settings()
    store = getattr(request.app.state, "store", None)
    return {
        "status": "ok",
        "llm_provider": s.llm_provider,           # "fake" = offline mode
        "llm_models": [s.llm_model, s.llm_fallback_model],
        "chunks_loaded": chunk_counts(store) if store else {},
    }


@app.post("/auth/token", response_model=TokenResponse)
def login(body: TokenRequest) -> TokenResponse:
    """MOCK identity provider - stands in for the company's SSO. No password: demo only."""
    try:
        token, ttl = issue_token(body.employee_id)
    except UnknownEmployeeError:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_employee", "message": "No employee with that ID."},
        )
    return TokenResponse(access_token=token, expires_in=ttl)


@app.get("/me")
def me(principal: Principal = Depends(current_principal)) -> dict:
    """Protected endpoint: shows exactly what the server believes about the caller."""
    return {"employee_id": principal.employee_id,
            "company_id": principal.company_id,
            "role": principal.role}


# Own pool for /ask work (not the event loop's default one): a stuck request can't
# hold up anything that waits on the default pool, such as loop shutdown.
_ASK_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="ask")


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={401: {"description": "Missing or invalid token"},
               422: {"description": "Invalid question"},
               503: {"description": "Temporarily unavailable - try again"}},
)
async def ask_endpoint(body: AskRequest, request: Request,
                       principal: Principal = Depends(current_principal)):
    """Answer from the caller's own company policy, with citations.

    company_id comes ONLY from the verified token (principal).

    The pipeline is blocking (embedding, HTTP to the LLM), so it runs in a worker
    thread. Every stage inside it has its own time limit; on top of that, this
    endpoint stops WAITING after REQUEST_TIMEOUT_S and answers "try again" (R10:
    never a hang). It doesn't wait for the worker to finish or be cancelled - a
    thread can't be cancelled, and waiting for it is exactly the hang we avoid.
    """
    limit = get_settings().request_timeout_s
    loop = asyncio.get_running_loop()
    work = loop.run_in_executor(_ASK_POOL, ask, request.app.state.store, request.app.state.llm,
                                principal.company_id, body.question)
    done, _ = await asyncio.wait({work}, timeout=limit)
    if not done:
        work.add_done_callback(lambda f: f.exception())   # collect it later, silently
        log.error("request backstop fired: /ask exceeded %.1fs", limit)
        return error(503, "temporarily_unavailable", messages.TRY_AGAIN,
                     headers={"Retry-After": "5"})
    result = work.result()

    if result.status == "unavailable":
        return error(503, "temporarily_unavailable", result.answer,
                     headers={"Retry-After": "5"}, request_id=result.request_id)

    return AskResponse(
        status=result.status,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        request_id=result.request_id,
    )
