"""Ingestion: policies/*.md -> parse -> validate -> chunk store + embed -> Chroma.

Runs automatically at app startup, and by hand with:
    python -m app.ingest

Order matters: ALL files are parsed and validated first, and only then is
anything written. One bad file stops ingestion before any collection changes.
"""
import logging
import time
from pathlib import Path

from app.auth import known_companies
from app.config import get_settings
from app.policy_parser import Policy, PolicyParseError, parse_policy_file
from app.store import PolicyStore, index_company, open_store

log = logging.getLogger("policy_assistant")


class IngestError(RuntimeError):
    pass


def load_policies(policies_dir: Path) -> list[Policy]:
    """Parse + validate every policy. Raises IngestError listing every problem found."""
    files = sorted(policies_dir.glob("*.md"))
    if not files:
        raise IngestError(f"no policy files (*.md) found in {policies_dir}")

    policies, problems = [], []
    for f in files:
        try:
            policies.append(parse_policy_file(f))
        except PolicyParseError as e:
            problems.append(str(e))

    companies = known_companies()
    seen: dict[str, str] = {}
    for p in policies:
        if p.company_id not in companies:
            problems.append(f"{p.source}: company_id {p.company_id!r} is not in the employee directory")
        if p.company_id in seen:
            problems.append(f"{p.source}: company {p.company_id!r} already has a policy ({seen[p.company_id]})")
        seen.setdefault(p.company_id, p.source)

    if problems:
        raise IngestError("policy validation failed:\n  - " + "\n  - ".join(problems))
    return policies


def ingest_all(store: PolicyStore, policies_dir: Path) -> dict[str, int]:
    """Validate everything, then rebuild one collection per company. Returns {company: chunks}."""
    policies = load_policies(policies_dir)
    counts = {}
    for p in policies:
        started = time.perf_counter()
        counts[p.company_id] = index_company(store, p.company_id, p.chunks)
        log.info("ingested %s: %d chunks from %s in %.2fs",
                 p.company_id, counts[p.company_id], p.source, time.perf_counter() - started)
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    started = time.perf_counter()
    counts = ingest_all(open_store(s.data_dir), s.policies_dir)
    print(f"\nIngested {sum(counts.values())} chunks in {time.perf_counter() - started:.1f}s "
          f"into {s.data_dir}/chunks + {s.data_dir}/chroma")
    for company, n in counts.items():
        print(f"  policy__{company:<6} {n:>3} chunks")


if __name__ == "__main__":
    main()
