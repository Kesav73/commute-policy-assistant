"""Turn a policy markdown file into chunks: one chunk per rule.

Chunking is structure-aware:
  * one chunk per "### N.M" sub-section (a "## N." section without sub-sections is one chunk)
  * the chunk text starts with its headings, "Section > Sub-section", so the headings
    help retrieval (the text of 4.1 itself never says "night" or "safety")
  * nothing is cut in the middle of a rule, and a citation points to an exact section

Anything unexpected raises PolicyParseError with the file name and line number:
a half-understood policy must never be served.

Expected format:
    ---
    company_id: acme
    doc_id: acme-transport-policy
    title: Acme Corp Employee Transport Policy
    effective_date: 2026-01-01
    ---
    ## 4. Night-Shift Safety
    ### 4.1 Escort Requirement
    For any drop scheduled after 8:00 PM, ...
"""
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("company_id", "doc_id", "title")
MAX_CHUNK_WORDS = 180   # one rule per chunk keeps each embedding focused

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(\S.*)$")          # "## 4. Night-Shift Safety"
SUBSECTION_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(\S.*)$")    # "### 4.1 Escort Requirement"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")                  # "acme", "acme-transport-policy"


class PolicyParseError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    id: str             # "acme-transport-policy:4.1"
    company_id: str     # "acme"
    doc_id: str         # "acme-transport-policy"
    title: str          # "Acme Corp Employee Transport Policy"
    section: str        # "4.1"
    section_title: str  # "Night-Shift Safety > Escort Requirement"
    body: str           # the rule text only
    text: str           # what gets embedded and shown to the LLM: section_title + "\n" + body


@dataclass(frozen=True)
class Policy:
    company_id: str
    doc_id: str
    title: str
    source: str              # file name, for error messages
    chunks: list[Chunk]
    effective_date: str = ""


def parse_policy_file(path: Path) -> Policy:
    return parse_policy(path.read_text(encoding="utf-8"), source=path.name)


def parse_policy(raw: str, source: str = "<policy>") -> Policy:
    fields, body_lines, first_line_no = _read_front_matter(raw, source)
    blocks = _split_into_blocks(body_lines, first_line_no, source)
    chunks = _blocks_to_chunks(blocks, fields, source)
    return Policy(company_id=fields["company_id"], doc_id=fields["doc_id"], title=fields["title"],
                  source=source, chunks=chunks, effective_date=fields.get("effective_date", ""))


def _read_front_matter(raw: str, source: str) -> tuple[dict, list[str], int]:
    """The "key: value" lines between the two '---' lines. Returns (fields, the
    remaining lines, the line number of the first remaining line)."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PolicyParseError(f"{source}: must start with a '---' front-matter block")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise PolicyParseError(f"{source}: front-matter block is not closed with '---'")

    fields = {}
    for line_no in range(2, end + 1):
        line = lines[line_no - 1]
        if not line.strip():
            continue
        key, colon, value = line.partition(":")
        if not colon or not key.strip():
            raise PolicyParseError(f"{source}:{line_no}: expected 'key: value', got {line!r}")
        fields[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        raise PolicyParseError(f"{source}: front matter is missing {', '.join(missing)}")
    for f in ("company_id", "doc_id"):
        if not ID_RE.match(fields[f]):
            raise PolicyParseError(f"{source}: {f} {fields[f]!r} must be lowercase letters, digits, '-' or '_'")
    return fields, lines[end + 1:], end + 2


def _split_into_blocks(lines: list[str], first_line_no: int, source: str) -> list[dict]:
    """Walk the lines once and group them under their heading. Each block is
    {"section", "num", "sub_title", "line", "body"}; "section" is the enclosing
    "## N." heading, "sub_title" is None for that heading's own block."""
    blocks = []
    section = None    # the current "## N." heading
    current = None    # the block that receives text lines

    for line_no, line in enumerate(lines, start=first_line_no):
        section_match = SECTION_RE.match(line)
        sub_match = SUBSECTION_RE.match(line)

        if section_match:
            section = {"num": section_match.group(1), "title": section_match.group(2).strip(),
                       "has_sub": False}
            current = {"section": section, "num": section["num"], "sub_title": None,
                       "line": line_no, "body": []}
            blocks.append(current)
        elif sub_match:
            num = sub_match.group(1)
            if section is None:
                raise PolicyParseError(f"{source}:{line_no}: sub-section before any '## N.' section")
            if num.split(".")[0] != section["num"]:
                raise PolicyParseError(f"{source}:{line_no}: sub-section {num} is inside section {section['num']}")
            section["has_sub"] = True
            current = {"section": section, "num": num, "sub_title": sub_match.group(2).strip(),
                       "line": line_no, "body": []}
            blocks.append(current)
        elif line.startswith("#"):
            raise PolicyParseError(
                f"{source}:{line_no}: unrecognised heading {line!r} (use '## N. Title' or '### N.M Title')")
        elif line.strip():
            if current is None:
                raise PolicyParseError(f"{source}:{line_no}: text before the first '## N.' section")
            current["body"].append(line.strip())
    return blocks


def _blocks_to_chunks(blocks: list[dict], fields: dict, source: str) -> list[Chunk]:
    chunks = []
    seen_numbers = set()
    for block in blocks:
        section = block["section"]
        is_section_heading = block["sub_title"] is None
        body = " ".join(block["body"])
        where = f"{source}:{block['line']}"

        # A "## N." heading that has sub-sections is only a container: no chunk.
        if is_section_heading and section["has_sub"]:
            if body:
                raise PolicyParseError(f"{where}: section {section['num']} has text before its first sub-section")
            continue

        if not body:
            raise PolicyParseError(f"{where}: section {block['num']} is empty")
        if block["num"] in seen_numbers:
            raise PolicyParseError(f"{where}: duplicate section number {block['num']}")
        seen_numbers.add(block["num"])
        words = len(body.split())
        if words > MAX_CHUNK_WORDS:
            raise PolicyParseError(f"{where}: section {block['num']} has {words} words "
                                   f"(max {MAX_CHUNK_WORDS}); split it into sub-sections")

        if is_section_heading:
            section_title = section["title"]
        else:
            section_title = f"{section['title']} > {block['sub_title']}"
        chunks.append(Chunk(
            id=f"{fields['doc_id']}:{block['num']}",
            company_id=fields["company_id"],
            doc_id=fields["doc_id"],
            title=fields["title"],
            section=block["num"],
            section_title=section_title,
            body=body,
            text=f"{section_title}\n{body}",
        ))

    if not chunks:
        raise PolicyParseError(f"{source}: no sections found")
    return chunks
