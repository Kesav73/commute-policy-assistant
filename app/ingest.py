"""Ingestion: policies/*.md -> parse -> validate -> publish as a new version (app/versioning.py).

Runs automatically at app startup, and by hand with:
    python -m app.ingest

Order matters: ALL files are parsed and validated first, and only then is
anything written. One bad file stops ingestion before any collection changes.
"""
import logging
import sys
from pathlib import Path

from app.auth import known_companies
from app.config import get_settings
from app.policy_parser import Policy, PolicyParseError, parse_policy_file
from app.store import PolicyStore, open_store
from app.versioning import publish_policy

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
    """Publish every policy file as a NEW version of its company (validates ALL files
    first, so one bad file changes nothing). Returns {company: chunks}.

    For tests, evals and the CLI with the app stopped. A running app is updated
    through POST /admin/policies, which goes through the same publish_policy()."""
    policies = load_policies(policies_dir)
    counts = {}
    for p in policies:
        res = publish_policy(store, (policies_dir / p.source).read_text(encoding="utf-8"),
                             source=p.source, old_version_grace_s=0)
        counts[p.company_id] = res.chunks
    return counts


def main() -> None:
    """python -m app.ingest              re-publish every file in policies/ (new versions)
       python -m app.ingest FILE.md      publish one file as the next version of its company
    Run with the app stopped; while it runs, use POST /admin/policies instead."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    store = open_store(s.data_dir)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        res = publish_policy(store, path.read_text(encoding="utf-8"), source=path.name,
                             old_version_grace_s=0)
        print(f"{res.company_id}: v{res.previous_version} -> v{res.version} ({res.chunks} chunks)")
        return
    counts = ingest_all(store, s.policies_dir)
    for company, n in counts.items():
        print(f"  {company:<6} v{store.active_version(company)}  {n:>3} chunks")


if __name__ == "__main__":
    main()
