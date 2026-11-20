"""HTTP layer: the four endpoints. Thin on purpose - the logic lives in the other modules.

    POST /auth/token       mock identity provider (stands in for the company's SSO)
    POST /ask              the product: question in, cited answer out
    POST /admin/policies   publish a new version of the admin's company policy
    GET  /health           is it up, which LLM mode, which policy version per company

Every error has one shape: {"error": "<code>", "message": "<plain text for a person>"}.
"""
import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import messages
from app.answer_cache import AnswerCache, CachedAnswer
from app.auth import Principal, UnknownEmployeeError, current_principal, issue_token, require_admin
from app.config import FALLBACK_MODEL, PRIMARY_MODEL, get_settings
from app.llm import build_llm
from app.pipeline import ask
from app.retriever import warm_up
from app.store import PolicyStore
from app.versioning import MAX_POLICY_CHARS, PolicyRejected, PublishFailed, bootstrap, publish_policy

log = logging.getLogger("policy_assistant")

QUESTION_MIN, QUESTION_MAX = 3, 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once, BEFORE the first request is accepted. Any error stops startup:
    better no service than one answering from half-loaded policies."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log.setLevel(logging.INFO)
    settings = get_settings()                          # fails fast on a bad configuration
    app.state.store = PolicyStore(settings.data_dir)
    versions = bootstrap(app.state.store, settings.policies_dir)   # seed / keep versions, clean up
    log.info("warm-up (ms): %s", warm_up(app.state.store))        # load the model now, not on question 1
    app.state.llm = build_llm(settings)
    app.state.answer_cache = AnswerCache(settings.answer_cache_size, settings.answer_cache_ttl_s)
    log.info("ready: policy versions=%s llm=%s", versions, settings.llm_provider)
    yield


app = FastAPI(title="Commute Policy Assistant", version="1.0", lifespan=lifespan)


# --------------------------------------------------------------- request / response --

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
    """Only a question. There is deliberately NO company field: the token says who you are."""
    question: str

    @field_validator("question")
    @classmethod
    def check_question(cls, value: str) -> str:
        value = value.strip()
        if "\x00" in value or not (QUESTION_MIN <= len(value) <= QUESTION_MAX):
            raise ValueError(messages.QUESTION_LENGTH.format(min=QUESTION_MIN, max=QUESTION_MAX))
        return value


class CitationOut(BaseModel):
    doc_id: str
    title: str
    section: str
    section_title: str


class AskResponse(BaseModel):
    status: str                          # "answered" | "not_covered"
    answer: str
    citations: list[CitationOut]
    request_id: str
    policy_version: int | None = None    # which version of the company's policy answered
    cached: bool = False                 # answered from the cache (0 LLM calls)


class PolicyUpload(StrictModel):
    """A complete new version of the admin's OWN company policy, in the same markdown
    format as policies/*.md. The company is read from the file and must match the token."""
    markdown: str = Field(min_length=1, max_length=MAX_POLICY_CHARS)


class PublishResponse(BaseModel):
    company_id: str
    version: int
    previous_version: int | None
    chunks: int
    quarantined: list[str]
    effective_date: str
    changes: dict          # {"changed": [...], "added": [...], "removed": [...], "unchanged": n}


# ------------------------------------------------------------------------- errors --

def error(status: int, code: str, message: str, headers: dict | None = None, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, headers=headers, content={"error": code, "message": message, **extra})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Give every HTTPException the one error shape."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return error(exc.status_code, exc.detail["error"], exc.detail["message"], exc.headers)
    return error(exc.status_code, "error", str(exc.detail), exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Replace FastAPI's technical 422 with one sentence a person can act on."""
    first = exc.errors()[0] if exc.errors() else {}
    field_name = str(first.get("loc", ["", ""])[-1])
    kind = first.get("type", "")
    if kind == "extra_forbidden":
        text = f"Unexpected field '{field_name}'. Send only the documented fields."
    elif kind == "missing":
        text = f"Missing field '{field_name}'."
    elif kind == "value_error":
        text = str(first.get("ctx", {}).get("error", first.get("msg", "")))
    else:
        text = "The request body is not valid JSON in the expected format."
    return error(422, "invalid_request", text)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last line of defence: details go to the server log, never a stack trace to the user."""
    log.exception("unhandled error on %s", request.url.path)
    return error(500, "internal_error", messages.INTERNAL_ERROR)


# ---------------------------------------------------------------------- endpoints --

@app.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    store = getattr(request.app.state, "store", None)
    summary = store.summary() if store else {"versions": {}, "chunks": {}}
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,          # "fake" = offline mode
        "llm_models": [PRIMARY_MODEL, FALLBACK_MODEL] if settings.llm_provider == "groq" else ["fake (offline)"],
        "chunks_loaded": summary["chunks"],
        "policy_versions": summary["versions"],
    }


@app.post("/auth/token", response_model=TokenResponse)
def login(body: TokenRequest) -> TokenResponse:
    """MOCK identity provider, standing in for the company's SSO. No password: demo only."""
    try:
        token, ttl = issue_token(body.employee_id)
    except UnknownEmployeeError:
        raise HTTPException(404, detail={"error": "unknown_employee", "message": "No employee with that ID."})
    return TokenResponse(access_token=token, expires_in=ttl)


# /ask work runs in its own thread pool, so a stuck request never blocks anything else.
_ASK_WORKERS = ThreadPoolExecutor(max_workers=32, thread_name_prefix="ask")


@app.post("/ask", response_model=AskResponse,
          responses={401: {"description": "Missing or invalid token"},
                     422: {"description": "Invalid question"},
                     503: {"description": "Temporarily unavailable - please try again"}})
async def ask_endpoint(body: AskRequest, request: Request, principal: Principal = Depends(current_principal)):
    """Answer from the caller's own company policy, with citations.
    The company comes ONLY from the verified token (principal)."""
    store = request.app.state.store
    cache = getattr(request.app.state, "answer_cache", None)
    company = principal.company_id

    # Same question, same company, same live policy version? Answer from the cache.
    if cache is not None:
        try:
            hit = cache.get(company, store.active_version(company), body.question)
        except Exception:   # noqa: BLE001 - a cache problem must never fail the request
            hit = None
        if hit is not None:
            log.info("ask company=%s v=%s status=%s cached=true", company, hit.policy_version, hit.status)
            return AskResponse(status=hit.status, answer=hit.answer,
                               citations=[CitationOut(**c.__dict__) for c in hit.citations],
                               request_id=uuid.uuid4().hex[:12], policy_version=hit.policy_version, cached=True)

    # Run the pipeline in a worker thread and stop WAITING after REQUEST_TIMEOUT_S.
    # (A thread can't be cancelled; waiting for it to stop would itself be a hang.)
    limit = get_settings().request_timeout_s
    loop = asyncio.get_running_loop()
    work = loop.run_in_executor(_ASK_WORKERS, ask, store, request.app.state.llm, company, body.question)
    done, _ = await asyncio.wait({work}, timeout=limit)
    if not done:
        work.add_done_callback(lambda f: f.exception())   # collect its result later, silently
        log.error("request backstop fired: /ask took longer than %.1fs", limit)
        return error(503, "temporarily_unavailable", messages.TRY_AGAIN, headers={"Retry-After": "5"})
    result = work.result()

    if result.status == "unavailable":   # never cached: a temporary failure
        return error(503, "temporarily_unavailable", result.answer,
                     headers={"Retry-After": "5"}, request_id=result.request_id)
    if cache is not None and result.policy_version is not None:
        cache.put(company, body.question,
                  CachedAnswer(result.status, result.answer, tuple(result.citations), result.policy_version))
    return AskResponse(status=result.status, answer=result.answer,
                       citations=[CitationOut(**c.__dict__) for c in result.citations],
                       request_id=result.request_id, policy_version=result.policy_version)


@app.post("/admin/policies", response_model=PublishResponse,
          responses={400: {"description": "Invalid policy - nothing was changed"},
                     401: {"description": "Missing or invalid token"},
                     403: {"description": "Not an admin, or not the admin's company"},
                     503: {"description": "Building the new version failed - the previous version is still active"}})
def publish_endpoint(body: PolicyUpload, request: Request, principal: Principal = Depends(require_admin)):
    """Publish a NEW VERSION of the admin's own company policy on the running service.
    Built next to the live version, verified, then switched in one atomic step; if
    anything fails, the previous version stays live."""
    try:
        res = publish_policy(request.app.state.store, body.markdown, source="upload",
                             expected_company=principal.company_id)
    except PolicyRejected as e:
        if "you can only update" in str(e):
            return error(403, "forbidden", f"{e}. Nothing was changed.")
        return error(400, "invalid_policy", f"{e}. Nothing was changed.")
    except PublishFailed as e:
        return error(503, "publish_failed", str(e), headers={"Retry-After": "30"})

    log.info("admin %s published %s v%d", principal.employee_id, res.company_id, res.version)
    cache = getattr(request.app.state, "answer_cache", None)
    if cache is not None:
        cache.drop_company(res.company_id)   # frees memory; old answers can't match v(n+1) anyway
    return PublishResponse(company_id=res.company_id, version=res.version,
                           previous_version=res.previous_version, chunks=res.chunks,
                           quarantined=res.quarantined, effective_date=res.effective_date, changes=res.changes)
