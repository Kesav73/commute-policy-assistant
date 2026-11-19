"""Prompt building and validation of untrusted model output."""
import json

import pytest

from app.answerer import (SYSTEM_PROMPT, InvalidOutput, build_user_prompt, parse_answer)
from app.retriever import RetrievedChunk


def chunk(section: str, text: str = "Title\nBody text.") -> RetrievedChunk:
    return RetrievedChunk(id=f"acme-transport-policy:{section}", company_id="acme",
                          doc_id="acme-transport-policy", title="Acme Policy",
                          section=section, section_title=f"Sec {section}", text=text, distance=0.4)


CHUNKS = [chunk("4.1"), chunk("4.2"), chunk("8.3")]


def reply(**kw) -> str:
    return json.dumps({"status": "answered", "answer": "An answer.", "citations": ["S1"], **kw})


# ---------------------------------------------------------------- prompt --

def test_prompt_labels_excerpts_and_hides_ids():
    p = build_user_prompt("Escort time?", CHUNKS)
    assert '<excerpt id="S1" section="4.1">' in p and '<excerpt id="S3" section="8.3">' in p
    assert "<question>\nEscort time?\n</question>" in p
    assert "acme-transport-policy" not in p   # the model never sees doc ids
    assert "Acme Policy" not in p             # nor document titles


def test_system_prompt_states_the_rules():
    for rule in ("ONLY from the policy excerpts", "not instructions", "not_covered", "S1, S2, S3"):
        assert rule in SYSTEM_PROMPT


# ----------------------------------------------------------- valid output --

def test_answered_maps_labels_to_real_citations():
    a = parse_answer(reply(citations=["S2", "S1"]), CHUNKS)
    assert a.status == "answered" and a.text == "An answer."
    assert [c.section for c in a.citations] == ["4.2", "4.1"]
    assert a.citations[0].doc_id == "acme-transport-policy"


@pytest.mark.parametrize("label", ["S1", "s1", "[S1]", " S 1 "])
def test_label_formats_are_tolerated(label):
    assert parse_answer(reply(citations=[label]), CHUNKS).citations[0].section == "4.1"


def test_duplicate_citations_collapse():
    assert len(parse_answer(reply(citations=["S1", "S1", "S2"]), CHUNKS).citations) == 2


def test_code_fenced_json_is_accepted():
    assert parse_answer("```json\n" + reply() + "\n```", CHUNKS).status == "answered"


def test_not_covered_drops_model_text():
    a = parse_answer(json.dumps({"status": "not_covered", "answer": "Bolt allows it!",
                                 "citations": ["S1"]}), CHUNKS)
    assert (a.status, a.text, a.citations) == ("not_covered", "", [])


def test_answer_text_is_capped():
    assert len(parse_answer(reply(answer="x" * 5000), CHUNKS).text) == 1200


# ------------------------------------------- unsupported -> not covered --

@pytest.mark.parametrize("cites, why", [
    (["S4"], "unknown labels"),                          # label we never gave it
    (["acme-transport-policy:9.2"], "unknown labels"),   # tries to invent a citation
    (["S1", "bolt-commute-policy §4.1"], "unknown labels"),
    ([], "without citations"),
])
def test_unsupported_answers_are_downgraded(cites, why):
    a = parse_answer(reply(citations=cites), CHUNKS)
    assert a.status == "not_covered" and a.citations == [] and why in a.downgraded


# ------------------------------------------------------- unusable output --

@pytest.mark.parametrize("raw", [
    "", "Sure! Escort after 8 PM.", "[1, 2]", '{"status": "maybe"}',
    '{"status": "answered", "answer": "", "citations": ["S1"]}',
    '{"status": "answered", "answer": "x", "citations": "S1"}',
])
def test_unusable_output_raises(raw):
    with pytest.raises(InvalidOutput):
        parse_answer(raw, CHUNKS)
