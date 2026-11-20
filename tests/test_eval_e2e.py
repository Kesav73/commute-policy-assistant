"""The end-to-end grader itself: checks and failure causes (no LLM, no HTTP)."""
from types import SimpleNamespace

from eval.run_e2e import score

CASE = {"company": "acme", "expected_status": "answered", "required": ["4.1"],
        "must_include": [["8:00 PM", "8 PM"]], "must_not_include": ["10:00 PM"]}
NC_CASE = {**CASE, "expected_status": "not_covered", "required": [], "must_include": []}


def res(retrieved=("4.1", "4.2", "4.3"), calls=1, reason="answered"):
    return SimpleNamespace(retrieved=[{"section": s} for s in retrieved], llm_calls=calls, reason=reason)


def body(status="answered", answer="Escort after 8:00 PM.", sections=("4.1",), doc="acme-transport-policy"):
    return {"status": status, "answer": answer,
            "citations": [{"section": s, "doc_id": doc} for s in sections]}


def test_correct_answer_passes():
    sc = score(CASE, 200, body(), res())
    assert sc["pass"] and sc["cause"] is None


def test_missing_fact_is_incomplete():
    sc = score(CASE, 200, body(answer="There is an escort."), res())
    assert not sc["pass"] and sc["cause"] == "incomplete"


def test_wrong_citation_is_citation():
    sc = score(CASE, 200, body(sections=("4.2",)), res())
    assert sc["cause"] == "citation"


def test_other_company_value_is_unsafe():
    sc = score(CASE, 200, body(answer="Escort after 8:00 PM or 10:00 PM."), res())
    assert sc["cause"] == "unsafe"


def test_other_company_citation_is_a_leak():
    sc = score(CASE, 200, body(doc="bolt-commute-policy"), res())
    assert not sc["checks"]["no_leak"] and sc["cause"] == "unsafe"


def test_missed_section_is_retrieval_not_model():
    sc = score(CASE, 200, body(status="not_covered", answer="Not covered.", sections=()),
               res(retrieved=("4.2", "4.3", "8.3")))
    assert sc["cause"] == "retrieval"


def test_not_covered_with_full_retrieval_is_over_cautious():
    sc = score(CASE, 200, body(status="not_covered", answer="Not covered.", sections=()), res())
    assert sc["cause"] == "over_cautious"


def test_answering_a_not_covered_question_is_confident_wrong():
    sc = score(NC_CASE, 200, body(), res())
    assert sc["cause"] == "confident_wrong"


def test_503_is_unavailable():
    sc = score(CASE, 503, {"error": "unavailable", "message": "Please try again."}, None)
    assert sc["status"] == "unavailable" and sc["cause"] == "unavailable"
