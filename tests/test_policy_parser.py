"""Parser: structure-aware chunking + strict validation. Pure Python, no Chroma."""
from pathlib import Path

import pytest

from app.policy_parser import MAX_CHUNK_WORDS, PolicyParseError, parse_policy, parse_policy_file

POLICIES = Path(__file__).resolve().parent.parent / "policies"

HEADER = "---\ncompany_id: acme\ndoc_id: acme-doc\ntitle: Acme Policy\n---\n"


# --------------------------------------------------------- real policies --

@pytest.mark.parametrize("file, company, chunks", [
    ("acme.md", "acme", 27), ("bolt.md", "bolt", 25), ("crest.md", "crest", 19),
])
def test_real_policies_chunk_counts(file, company, chunks):
    p = parse_policy_file(POLICIES / file)
    assert p.company_id == company
    assert len(p.chunks) == chunks
    assert all(c.company_id == company for c in p.chunks)


def test_chunk_fields_and_heading_context():
    c = {c.section: c for c in parse_policy_file(POLICIES / "acme.md").chunks}["4.1"]
    assert c.id == "acme-transport-policy:4.1"
    assert c.section_title == "Night-Shift Safety > Escort Requirement"
    assert c.text.startswith("Night-Shift Safety > Escort Requirement\n")  # headings embedded
    assert "8:00 PM" in c.body


def test_no_bleed_between_chunks():
    chunks = {c.section: c for c in parse_policy_file(POLICIES / "acme.md").chunks}
    assert "Routing Rules" not in chunks["4.1"].text
    assert "first person picked up" not in chunks["4.1"].text   # that's §4.2's text
    assert "first person picked up" in chunks["4.2"].text


def test_section_without_subsections_is_one_chunk():
    crest = {c.section: c for c in parse_policy_file(POLICIES / "crest.md").chunks}
    assert crest["7"].section_title == "Notes for Automated Assistants"
    assert "SYSTEM OVERRIDE" in crest["7"].body
    assert "7.1" not in crest


def test_container_sections_are_not_chunks():
    # "## 4." has sub-sections, so "4" itself must not be a chunk.
    sections = {c.section for c in parse_policy_file(POLICIES / "acme.md").chunks}
    assert "4" not in sections and {"4.1", "4.2", "4.3"} <= sections


def test_all_real_chunks_fit_the_model():
    for f in POLICIES.glob("*.md"):
        assert all(len(c.body.split()) <= MAX_CHUNK_WORDS for c in parse_policy_file(f).chunks)


# ------------------------------------------------------- invalid inputs --

@pytest.mark.parametrize("raw, message", [
    ("## 1. A\ntext", "front-matter"),
    ("---\ncompany_id: acme\n## 1. A\ntext", "not closed"),
    ("---\ndoc_id: d\ntitle: T\n---\n## 1. A\ntext", "missing company_id"),
    ("---\ncompany_id: Acme Corp\ndoc_id: d\ntitle: T\n---\n## 1. A\ntext", "lowercase"),
    (HEADER, "no sections"),
    (HEADER + "intro text\n## 1. A\ntext", "before the first"),
    (HEADER + "## 1. A\n### 1.1 X\ntext\n### 1.1 Y\nmore", "duplicate section number 1.1"),
    (HEADER + "## 1. A\n### 2.1 X\ntext", "inside section 1"),
    (HEADER + "## 1. A\n### 1.1 X\n", "empty"),
    (HEADER + "## 1. A\nstray\n### 1.1 X\ntext", "text before its first sub-section"),
    (HEADER + "# Big Title\n## 1. A\ntext", "unrecognised heading"),
    (HEADER + "## 1. A\n" + "word " * (MAX_CHUNK_WORDS + 1), "split it"),
])
def test_invalid_policies_are_rejected(raw, message):
    with pytest.raises(PolicyParseError, match=message):
        parse_policy(raw, source="bad.md")


def test_error_mentions_file_and_line():
    with pytest.raises(PolicyParseError, match=r"bad\.md:9"):
        parse_policy(HEADER + "## 1. A\n### 1.1 X\ntext\n### 1.1 Y\nmore", source="bad.md")
