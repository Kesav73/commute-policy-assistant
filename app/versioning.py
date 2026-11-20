"""Policy versions: publish a new version safely, and get the store ready at startup.

publish_policy(store, markdown)       used by POST /admin/policies, the CLI and startup seeding
    1. parse + validate                 any problem -> PolicyRejected, NOTHING touched
    2. quarantine instruction-like sections (guardrail on chunks)
    3. build v(n+1) NEXT TO the live v(n)   (v(n) keeps answering meanwhile)
    4. verify v(n+1)
    5. switch active.json to v(n+1)     one atomic file rename: the "flip"
    6. delete v(n) after a short grace period
    A failure in 3-4 deletes the partial v(n+1); v(n) stays fully live, so an update
    never leaves a half-updated policy.

bootstrap(store, policies_dir)        at startup
    * deletes anything not named in active.json (e.g. a build that crashed before its flip)
    * seeds companies that have no version yet from policies/<company>.md as v1;
      companies that already have one keep it, so a restart never undoes an update

Run by hand (with the app stopped):  python -m app.versioning [FILE.md]
"""
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.auth import known_companies
from app.config import get_settings
from app.guardrails import is_instruction_like
from app.policy_parser import Policy, PolicyParseError, parse_policy, parse_policy_file
from app.store import PolicyStore

log = logging.getLogger("policy_assistant")

OLD_VERSION_GRACE_S = 10.0     # longer than any request (3 s), so in-flight requests finish
MAX_POLICY_CHARS = 200_000

_company_locks = {}            # one lock per company: two uploads can't build at once
_locks_guard = threading.Lock()


class PolicyRejected(ValueError):
    """The policy is invalid. Nothing was changed."""


class PublishFailed(RuntimeError):
    """Building the new version failed; it was removed and the old version is still live."""


@dataclass
class PublishResult:
    company_id: str
    version: int
    previous_version: int | None
    chunks: int
    quarantined: list[str] = field(default_factory=list)
    effective_date: str = ""
    changes: dict = field(default_factory=dict)   # what differs from the previous version


def validate_policy(markdown: str, source: str, expected_company: str | None = None) -> Policy:
    """Parse and check a policy. Raises PolicyRejected with a clear reason."""
    if not markdown or not markdown.strip():
        raise PolicyRejected(f"{source}: empty policy")
    if len(markdown) > MAX_POLICY_CHARS:
        raise PolicyRejected(f"{source}: policy is larger than {MAX_POLICY_CHARS} characters")
    try:
        policy = parse_policy(markdown, source=source)
    except PolicyParseError as e:
        raise PolicyRejected(str(e)) from e
    if policy.company_id not in known_companies():
        raise PolicyRejected(f"{source}: company_id {policy.company_id!r} is not in the employee directory")
    if expected_company is not None and policy.company_id != expected_company:
        raise PolicyRejected(f"{source}: this policy is for {policy.company_id!r}; "
                             f"you can only update {expected_company!r}")
    return policy


def compare_versions(old_chunks: dict, new_chunks: list) -> dict:
    """Section-by-section report against the live version, e.g.
    {"changed": ["4.1 Night-Shift Safety > Escort Requirement"], "added": [], "removed": [], "unchanged": 26}"""
    before = {c.section: c for c in old_chunks.values()}
    after = {c.section: c for c in new_chunks}

    def section_order(section: str) -> list[int]:    # "10.1" sorts after "9.3"
        return [int(part) for part in section.split(".")]

    def label(chunk) -> str:
        return f"{chunk.section} {chunk.section_title}"

    changed, unchanged = [], 0
    for section in sorted(before.keys() & after.keys(), key=section_order):
        old, new = before[section], after[section]
        if (old.section_title, old.body) != (new.section_title, new.body):
            changed.append(label(new))
        else:
            unchanged += 1
    added = [label(after[s]) for s in sorted(after.keys() - before.keys(), key=section_order)]
    removed = [label(before[s]) for s in sorted(before.keys() - after.keys(), key=section_order)]
    return {"changed": changed, "added": added, "removed": removed, "unchanged": unchanged}


def publish_policy(store: PolicyStore, markdown: str, source: str = "upload",
                   expected_company: str | None = None,
                   old_version_grace_s: float = OLD_VERSION_GRACE_S) -> PublishResult:
    # 1. Validate. Nothing has been touched yet.
    policy = validate_policy(markdown, source, expected_company)
    company = policy.company_id

    # 2. Guardrail on chunks: sections that give orders to an AI are never indexed.
    kept, quarantined = [], []
    for chunk in policy.chunks:
        if is_instruction_like(chunk.text):
            quarantined.append(chunk.id)
            log.warning("quarantined %s (%s): instruction-like text, not indexed", chunk.id, chunk.section_title)
        else:
            kept.append(chunk)
    if not kept:
        raise PolicyRejected(f"{source}: no usable sections left after quarantine")

    with _lock_for(company):
        previous = store.active_version(company)
        all_numbers = store.versions_on_disk(company) | {previous or 0}
        version = max(all_numbers) + 1                       # a number is never reused
        changes = compare_versions(store.chunks(company, previous) if previous else {}, kept)

        # 3-4. Build beside the live version and verify; on any failure remove the partial build.
        try:
            count = store.build_version(company, version, kept)
            source_copy = store.root / "policies" / company / f"v{version}.md"   # kept for audit
            source_copy.parent.mkdir(parents=True, exist_ok=True)
            source_copy.write_text(markdown, encoding="utf-8")
        except Exception as e:
            log.error("publish %s v%d failed, removing the partial build: %s", company, version, e)
            _remove_quietly(store, company, version)
            raise PublishFailed(f"building {company} v{version} failed ({type(e).__name__}); "
                                f"v{previous} is still active") from e

        # 5. The flip: from the next request on, only the new version answers.
        store.set_active(company, {
            "version": version, "doc_id": policy.doc_id, "title": policy.title,
            "effective_date": policy.effective_date, "chunks": count, "quarantined": quarantined,
            "source": source, "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        log.info("published %s v%d (%d chunks); previous v%s; changed %s, added %s, removed %s",
                 company, version, count, previous, changes["changed"], changes["added"], changes["removed"])

        # 6. Retire the old version (immediately when nothing can be using it, e.g. CLI/tests).
        if previous is not None:
            if old_version_grace_s > 0:
                store.delete_later(company, previous, old_version_grace_s)
            else:
                store.delete_version(company, previous)

    return PublishResult(company, version, previous, count, quarantined, policy.effective_date, changes)


def _lock_for(company_id: str) -> threading.Lock:
    with _locks_guard:
        if company_id not in _company_locks:
            _company_locks[company_id] = threading.Lock()
        return _company_locks[company_id]


def _remove_quietly(store: PolicyStore, company_id: str, version: int) -> None:
    try:
        store.delete_version(company_id, version)
    except Exception as e:   # noqa: BLE001 - startup cleanup will retry
        log.error("could not remove partial %s v%d: %s", company_id, version, e)
    (store.root / "policies" / company_id / f"v{version}.md").unlink(missing_ok=True)


# ---------------------------------------------------------------- policy files --

def load_policies(policies_dir: Path) -> list[Policy]:
    """Parse and check every policies/*.md. Raises PolicyRejected listing every problem,
    so one bad file stops startup before anything is written."""
    files = sorted(policies_dir.glob("*.md"))
    if not files:
        raise PolicyRejected(f"no policy files (*.md) found in {policies_dir}")
    policies, problems = [], []
    for path in files:
        try:
            policies.append(parse_policy_file(path))
        except PolicyParseError as e:
            problems.append(str(e))

    companies = known_companies()
    seen = {}
    for p in policies:
        if p.company_id not in companies:
            problems.append(f"{p.source}: company_id {p.company_id!r} is not in the employee directory")
        elif p.company_id in seen:
            problems.append(f"{p.source}: company {p.company_id!r} already has a policy ({seen[p.company_id]})")
        seen.setdefault(p.company_id, p.source)
    if problems:
        raise PolicyRejected("policy validation failed:\n  - " + "\n  - ".join(problems))
    return policies


def ingest_all(store: PolicyStore, policies_dir: Path) -> dict[str, int]:
    """Publish every policy file as a NEW version of its company (tests, evals, CLI).
    Returns {company: chunks}."""
    counts = {}
    for policy in load_policies(policies_dir):
        markdown = (policies_dir / policy.source).read_text(encoding="utf-8")
        result = publish_policy(store, markdown, source=policy.source, old_version_grace_s=0)
        counts[policy.company_id] = result.chunks
    return counts


def bootstrap(store: PolicyStore, policies_dir: Path) -> dict[str, int]:
    """Make the store clean and complete at startup. Returns {company: live version}."""
    seeds = load_policies(policies_dir)          # validates every seed file first

    # Remove everything not named in active.json (a crashed build, an interrupted deletion).
    for company in sorted(store.companies_on_disk()):
        live = store.active_version(company)
        for version in sorted(store.versions_on_disk(company) - {live}):
            log.warning("removing %s v%d: not the live version (left over from an interrupted update)",
                        company, version)
            store.delete_version(company, version)
            (store.root / "policies" / company / f"v{version}.md").unlink(missing_ok=True)

    # Seed companies with no version yet; keep existing versions.
    for policy in seeds:
        if store.active_version(policy.company_id) is None:
            markdown = (policies_dir / policy.source).read_text(encoding="utf-8")
            publish_policy(store, markdown, source=policy.source)

    return {company: store.active_version(company) for company in store.active_companies()}


if __name__ == "__main__":
    # python -m app.versioning             re-publish every file in policies/ as new versions
    # python -m app.versioning FILE.md     publish one file as the next version of its company
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    store = PolicyStore(settings.data_dir)
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        res = publish_policy(store, path.read_text(encoding="utf-8"), source=path.name, old_version_grace_s=0)
        print(f"{res.company_id}: v{res.previous_version} -> v{res.version} ({res.chunks} chunks)")
    else:
        for company, n in ingest_all(store, settings.policies_dir).items():
            print(f"{company}: v{store.active_version(company)}, {n} chunks")
