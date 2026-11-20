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
from app.config import get_settings
from app.llm import build_llm
from app.pipeline import ask
from app.retriever import warm_up
from app.store import active_versions, chunk_counts, open_store
from app.versioning import MAX_POLICY_CHARS, PolicyRejected, PublishFailed, bootstrap, publish_policy

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
    # Seed companies with no version yet from policies/, keep existing versions (a
    # restart never rolls back an update), remove leftovers of interrupted builds.
    versions = bootstrap(app.state.store, s.policies_dir)
    # Load the embedding model and indexes now, not inside the first question.
    log.info("warm-up (ms): %s", warm_up(app.state.store))
    app.state.llm = build_llm(s)
    app.state.answer_cache = AnswerCache(s.answer_cache_size, s.answer_cache_ttl_s)
    log.info("ready: policy versions=%s chunks=%s llm=%s (%s, fallback %s)", versions,
             chunk_counts(app.state.store), s.llm_provider, s.llm_model, s.llm_fallback_model)
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
    policy_version: int | None = None   # the version of the company's policy that answered
    cached: bool = False                # served from the answer cache (0 LLM calls)


class PolicyUpload(StrictModel):
    """A new version of the admin's OWN company's policy, as markdown (same format as
    policies/*.md). The company is read from the file and must match the admin's token."""
    markdown: str = Field(min_length=1, max_length=MAX_POLICY_CHARS)


class PublishResponse(BaseModel):
    company_id: str
    version: int
    previous_version: int | None
    chunks: int
    quarantined: list[str]
    effective_date: str
    changes: dict          # {"changed": [...], "added": [...], "removed": [...], "unchanged": n}


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
        "policy_versions": active_versions(store) if store else {},
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
    # Repeated question for the SAME company and the SAME live policy version?
    # Answer from the cache: 0 LLM calls, a few ms (EVALUATION.md).
    store, cache = request.app.state.store, getattr(request.app.state, "answer_cache", None)
    if cache is not None:
        try:
            hit = cache.get(principal.company_id, store.active_version(principal.company_id),
                            body.question)
        except Exception:   # noqa: BLE001 - a cache problem must never fail the request
            hit = None
        if hit is not None:
            rid = uuid.uuid4().hex[:12]
            log.info("ask rid=%s company=%s v=%s status=%s cached=true",
                     rid, principal.company_id, hit.policy_version, hit.status)
            return AskResponse(status=hit.status, answer=hit.answer,
                               citations=[CitationOut(**c.__dict__) for c in hit.citations],
                               request_id=rid, policy_version=hit.policy_version, cached=True)

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

    if result.status == "unavailable":           # never cached: a temporary failure
        return error(503, "temporarily_unavailable", result.answer,
                     headers={"Retry-After": "5"}, request_id=result.request_id)
    if cache is not None and result.policy_version is not None:
        cache.put(principal.company_id, body.question,
                  CachedAnswer(result.status, result.answer, tuple(result.citations),
                               result.policy_version))

    return AskResponse(
        status=result.status,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        request_id=result.request_id,
        policy_version=result.policy_version,
    )


@app.post(
    "/admin/policies",
    response_model=PublishResponse,
    responses={400: {"description": "Invalid policy - nothing was changed"},
               401: {"description": "Missing or invalid token"},
               403: {"description": "Not an admin, or not the admin's company"},
               503: {"description": "Building the new version failed - the previous version is still active"}},
)
def publish_endpoint(body: PolicyUpload, request: Request,
                     principal: Principal = Depends(require_admin)):
    """Publish a NEW VERSION of the caller's company policy, on the running service.

    Built next to the live version, verified, then switched in one atomic step:
    from the next question on, only the new version answers. If anything fails,
    the previous version stays active and nothing half-built is ever served.
    The company comes ONLY from the admin's token; a file for another company -> 403.
    """
    try:
        res = publish_policy(request.app.state.store, body.markdown, source="upload",
                             expected_company=principal.company_id)
    except PolicyRejected as e:
        status_code = 403 if "you can only update" in str(e) else 400
        code = "forbidden" if status_code == 403 else "invalid_policy"
        return error(status_code, code, f"{e}. Nothing was changed.")
    except PublishFailed as e:
        return error(503, "publish_failed", str(e), headers={"Retry-After": "30"})
    log.info("admin %s published %s v%d", principal.employee_id, res.company_id, res.version)
    cache = getattr(request.app.state, "answer_cache", None)
    if cache is not None:   # free memory; correctness already comes from the version in the key
        cache.drop_company(res.company_id)
    return PublishResponse(company_id=res.company_id, version=res.version,
                           previous_version=res.previous_version, chunks=res.chunks,
                           quarantined=res.quarantined, effective_date=res.effective_date,
                           changes=res.changes)
