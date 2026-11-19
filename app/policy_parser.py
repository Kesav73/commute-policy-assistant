"""Policy Markdown -> validated chunks. Pure Python: no database, no model.

Chunking strategy (HLD §5.2): structure-aware.
  * one chunk per `### N.M` sub-section
  * a `## N.` section with no sub-sections is one chunk on its own
  * the chunk text starts with "Section title > Sub-section title" so the headings
    help retrieval (the body of §4.1 never says "night" or "safety")
  * no overlap: boundaries are the author's own headings, nothing is cut mid-rule

Anything unexpected raises PolicyParseError with the file name and line number.
We never guess: a half-understood policy must not be served.
"""
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("company_id", "doc_id", "title")
# MiniLM reads ~256 tokens and silently drops the rest; stay well under that.
MAX_CHUNK_WORDS = 180

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(\S.*)$")         # "## 4. Night-Shift Safety"
SUBSECTION_RE = re.compile(r"^###\s+(\d+\.\d+)\s+(\S.*)$")   # "### 4.1 Escort Requirement"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class PolicyParseError(ValueError):
    pass


@dataclass(frozen=True)
class Chunk:
    id: str             # "acme-transport-policy:4.1"   (deterministic -> no duplicates)
    company_id: str     # "acme"
    doc_id: str         # "acme-transport-policy"
    title: str          # "Acme Corp Employee Transport Policy"
    section: str        # "4.1"
    section_title: str  # "Night-Shift Safety > Escort Requirement"
    body: str           # the rule text only
    text: str           # what gets embedded: section_title + "\n" + body

    def metadata(self) -> dict:
        """What Chroma stores next to the vector (used for citations and the company check)."""
        return {
            "company_id": self.company_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "section_title": self.section_title,
        }


@dataclass(frozen=True)
class Policy:
    company_id: str
    doc_id: str
    title: str
    source: str
    chunks: list[Chunk]


# ------------------------------------------------------------ front matter --

def _split_front_matter(raw: str, source: str) -> tuple[dict, list[str], int]:
    """Return (fields, body_lines, line_number_of_first_body_line)."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise PolicyParseError(f"{source}: must start with a '---' front-matter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise PolicyParseError(f"{source}: front-matter block is not closed with '---'")

    fields: dict[str, str] = {}
    for n, line in enumerate(lines[1:end], start=2):
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            raise PolicyParseError(f"{source}:{n}: expected 'key: value', got {line!r}")
        fields[key.strip()] = value.strip()

    missing = [f for f in REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        raise PolicyParseError(f"{source}: front matter is missing {', '.join(missing)}")
    for f in ("company_id", "doc_id"):
        if not ID_RE.match(fields[f]):
            raise PolicyParseError(f"{source}: {f} {fields[f]!r} must be lowercase letters, digits, '-' or '_'")

    return fields, lines[end + 1:], end + 2


# ------------------------------------------------------------------ body --

def parse_policy(raw: str, source: str = "<policy>") -> Policy:
    fields, body_lines, first_line_no = _split_front_matter(raw, source)
    company_id, doc_id, title = fields["company_id"], fields["doc_id"], fields["title"]

    # Walk the lines once, collecting (section, sub-section, body lines) blocks.
    blocks: list[dict] = []
    section: dict | None = None      # current "## N." heading
    current: dict | None = None      # block receiving body lines

    for n, line in enumerate(body_lines, start=first_line_no):
        if m := SECTION_RE.match(line):
            section = {"num": m.group(1), "title": m.group(2).strip(), "line": n, "has_sub": False}
            current = {"section": section, "num": section["num"], "sub_title": None, "line": n, "body": []}
            blocks.append(current)
        elif m := SUBSECTION_RE.match(line):
            if section is None:
                raise PolicyParseError(f"{source}:{n}: sub-section before any '## N.' section")
            num = m.group(1)
            if num.split(".")[0] != section["num"]:
                raise PolicyParseError(
                    f"{source}:{n}: sub-section {num} is inside section {section['num']}")
            section["has_sub"] = True
            current = {"section": section, "num": num, "sub_title": m.group(2).strip(), "line": n, "body": []}
            blocks.append(current)
        elif line.startswith("#"):
            raise PolicyParseError(
                f"{source}:{n}: unrecognised heading {line!r} (use '## N. Title' or '### N.M Title')")
        elif line.strip():
            if current is None:
                raise PolicyParseError(f"{source}:{n}: text before the first '## N.' section")
            current["body"].append(line.strip())

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for b in blocks:
        sec = b["section"]
        is_section_block = b["sub_title"] is None
        body = " ".join(b["body"])

        if is_section_block and sec["has_sub"]:
            # A "## N." heading that has sub-sections is only a container.
            if body:
                raise PolicyParseError(
                    f"{source}:{b['line']}: section {sec['num']} has text before its first sub-section")
            continue

        if not body:
            raise PolicyParseError(f"{source}:{b['line']}: section {b['num']} is empty")
        if b["num"] in seen:
            raise PolicyParseError(f"{source}:{b['line']}: duplicate section number {b['num']}")
        seen.add(b["num"])

        words = len(body.split())
        if words > MAX_CHUNK_WORDS:
            raise PolicyParseError(
                f"{source}:{b['line']}: section {b['num']} has {words} words "
                f"(max {MAX_CHUNK_WORDS}); split it into sub-sections")

        section_title = sec["title"] if is_section_block else f"{sec['title']} > {b['sub_title']}"
        chunks.append(Chunk(
            id=f"{doc_id}:{b['num']}",
            company_id=company_id,
            doc_id=doc_id,
            title=title,
            section=b["num"],
            section_title=section_title,
            body=body,
            text=f"{section_title}\n{body}",
        ))

    if not chunks:
        raise PolicyParseError(f"{source}: no sections found")

    return Policy(company_id=company_id, doc_id=doc_id, title=title, source=source, chunks=chunks)


def parse_policy_file(path: Path) -> Policy:
    return parse_policy(path.read_text(encoding="utf-8"), source=path.name)
