"""POST /ask over HTTP: token -> company -> pipeline -> JSON response, and every error shape."""
import pytest
from fastapi.testclient import TestClient

from app import messages
from app.llm import FakeLLM, LLMError
from app.main import app
from tests.conftest import ESCORT_Q, bearer

Q = {"question": ESCORT_Q}


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture(autouse=True)
def _app_state(store, llm):
    """Skip the startup hook; give the app the shared store and a fake LLM directly."""
    app.state.store, app.state.llm = store, llm
    yield
    del app.state.store, app.state.llm


@pytest.fixture
def acme(token_for):
    return bearer(token_for("E-ACME-001"))


# -------------------------------------------------------------- answered --

def test_answer_shape_and_citation(client, acme):
    r = client.post("/ask", json=Q, headers=acme)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"status", "answer", "citations", "request_id", "policy_version", "cached"}
    assert body["policy_version"] == 1
    assert body["status"] == "answered"
    assert body["citations"][0] == {
        "doc_id": "acme-transport-policy", "title": "Acme Corp Employee Transport Policy",
        "section": "4.1", "section_title": "Night-Shift Safety > Escort Requirement"}


def test_company_comes_from_token(client, token_for):
    for employee, doc, value in [("E-ACME-001", "acme-transport-policy", "8:00 PM"),
                                 ("E-BOLT-001", "bolt-commute-policy", "10:00 PM"),
                                 ("E-CRST-001", "crest-transport-guidelines", "7:00 PM")]:
        body = client.post("/ask", json=Q, headers=bearer(token_for(employee))).json()
        assert {c["doc_id"] for c in body["citations"]} == {doc}
        assert value in body["answer"]


def test_admin_of_a_company_can_ask_too(client, token_for):
    r = client.post("/ask", json=Q, headers=bearer(token_for("A-BOLT-ADM")))
    assert r.json()["citations"][0]["doc_id"] == "bolt-commute-policy"


# --------------------------------------------------- no client override --

def test_company_field_in_body_is_rejected(client, acme):
    r = client.post("/ask", json={**Q, "company_id": "bolt"}, headers=acme)
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_request",
                        "message": "Unexpected field 'company_id'. Send only the documented fields."}


def test_company_in_query_or_header_is_ignored(client, acme, llm):
    r = client.post("/ask?company_id=bolt", json=Q, headers={**acme, "X-Company-Id": "bolt"})
    assert r.json()["citations"][0]["doc_id"] == "acme-transport-policy"
    assert "10:00 PM" not in llm.calls[0]["user"]


# ------------------------------------------------------------------ auth --

def test_no_token_is_401(client):
    r = client.post("/ask", json=Q)
    assert r.status_code == 401 and r.json()["error"] == "unauthorized"


def test_bad_token_is_401_and_llm_never_called(client, llm):
    r = client.post("/ask", json=Q, headers=bearer("forged.token.value"))
    assert r.status_code == 401 and llm.calls == []


# ----------------------------------------------------------------- input --

@pytest.mark.parametrize("question", ["", "  ", "hi", "x" * 501, "abc\x00def"])
def test_bad_question_is_422_with_plain_message(client, acme, question, llm):
    r = client.post("/ask", json={"question": question}, headers=acme)
    assert r.status_code == 422
    assert r.json() == {"error": "invalid_request",
                        "message": "Please enter a question between 3 and 500 characters."}
    assert llm.calls == []


@pytest.mark.parametrize("payload", [{}, {"question": 42}, None])
def test_malformed_body_is_422(client, acme, payload):
    r = client.post("/ask", json=payload, headers=acme)
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_non_json_body_is_422(client, acme):
    r = client.post("/ask", content=b"question=hello", headers={**acme, "Content-Type": "text/plain"})
    assert r.status_code == 422


# ---------------------------------------------------------- not covered --

def test_not_covered_is_200_with_desk_message(client, acme):
    r = client.post("/ask", json={"question": "Write me a poem about cats"}, headers=acme)
    assert r.status_code == 200
    assert r.json()["status"] == "not_covered"
    assert r.json()["answer"] == messages.NOT_COVERED and "transport desk" in r.json()["answer"]
    assert r.json()["citations"] == []


# --------------------------------------------------------------- failure --

def test_llm_down_is_503_try_again(client, acme, llm):
    llm.script.extend([LLMError("timeout", "t"), LLMError("timeout", "t")])
    r = client.post("/ask", json=Q, headers=acme)
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "5"
    body = r.json()
    assert body["error"] == "temporarily_unavailable" and body["message"] == messages.TRY_AGAIN
    assert body["request_id"]


def test_unexpected_crash_is_500_without_stack_trace(acme, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("app.main.ask", boom)
    r = TestClient(app, raise_server_exceptions=False).post("/ask", json=Q, headers=acme)
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "message": messages.INTERNAL_ERROR}
    assert "secret internal detail" not in r.text and "Traceback" not in r.text


def test_health_reports_mode_and_chunks(client):
    body = client.get("/health").json()
    assert body["llm_provider"] == "fake"
    assert body["chunks_loaded"] == {"acme": 27, "bolt": 25, "crest": 18}
