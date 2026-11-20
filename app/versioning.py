"""Policy versions: publish a new version safely, and restore state at startup.

publish_policy(store, markdown)          (POST /admin/policies, ingest CLI, startup seeding)
    1. parse + validate (company known, matches the admin's company, sections ok)
       -> any problem: PolicyRejected, NOTHING has been touched
    2. under the company's lock: build v(n+1) NEXT TO the live v(n)
       (own chunk file + own Chroma collection; v(n) keeps answering)
    3. verify v(n+1): same ids in both stores, expected count, a search works
    4. keep the uploaded markdown as data/policies/<company>/v(n+1).md (audit)
    5. FLIP: active.json -> v(n+1)   (one atomic file rename)
    6. delete v(n) after a short grace period (never served again from step 5)
    Any failure in 2-4 -> the partial v(n+1) is deleted and v(n) is still active:
    a failed ingestion never leaves the policy half-updated (PS.md R3).

bootstrap(store, policies_dir)           (app startup)
    * a company with no active version is seeded from policies/<company>.md as v1;
      a company that already has one keeps it (a restart never rolls back an update)
    * anything not named in active.json - a build that crashed before its flip, an old
      version whose deletion was interrupted, pre-versioning collections - is deleted
    * an active version whose files are missing or were built with another embedding
      model is rebuilt from its kept markdown as a new version
"""
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.auth import known_companies
from app.guardrails import is_instruction_like
from app.policy_parser import Policy, PolicyParseError, parse_policy
from app.store import (PolicyStore, StoreError, build_version, delete_later, delete_version,
                       legacy_collections, vector_versions, verify_version)

log = logging.getLogger("policy_assistant")

OLD_VERSION_GRACE_S = 10.0   # > request time limit (3 s): in-flight requests on the old version finish
MAX_POLICY_CHARS = 200_000

_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


class PolicyRejected(ValueError):
    """The uploaded policy is invalid. Nothing was changed."""


class PublishFailed(RuntimeError):
    """Building the new version failed. It was rolled back; the old version is still active."""


@dataclass
class PublishResult:
    company_id: str
    version: int
    previous_version: int | None
    chunks: int
    quarantined: list[str] = field(default_factory=list)
    doc_id: str = ""
    effective_date: str = ""
    changes: dict = field(default_factory=dict)   # what differs from the previous version


def compare_versions(old: dict, new: list) -> dict:
    """Section-by-section report of a new version against the live one.

    old: {chunk id: Chunk} of the previous version ({} for a first version)
    new: the new version's indexed chunks
    -> {"changed": [...], "added": [...], "removed": [...], "unchanged": n}
    Entries read "4.1 Night-Shift Safety > Escort Requirement". The admin can check
    the update did what they meant (one rule edited vs. a wrong or truncated file).
    """
    before = {c.section: c for c in old.values()}
    after = {c.section: c for c in new}
    label = lambda c: f"{c.section} {c.section_title}"  # noqa: E731
    order = lambda s: [int(x) for x in s.split(".")]    # noqa: E731
    changed = [label(after[s]) for s in sorted(before.keys() & after.keys(), key=order)
               if (before[s].section_title, before[s].body) != (after[s].section_title, after[s].body)]
    return {
        "changed": changed,
        "added": [label(after[s]) for s in sorted(after.keys() - before.keys(), key=order)],
        "removed": [label(before[s]) for s in sorted(before.keys() - after.keys(), key=order)],
        "unchanged": len(before.keys() & after.keys()) - len(changed),
    }


def _lock(company_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks[company_id]


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


def publish_policy(store: PolicyStore, markdown: str, source: str = "upload",
                   expected_company: str | None = None,
                   old_version_grace_s: float = OLD_VERSION_GRACE_S) -> PublishResult:
    policy = validate_policy(markdown, source, expected_company)   # 1. nothing touched yet
    co = policy.company_id

    # Guardrail (chunks): instruction-like sections are never indexed.
    kept, quarantined = [], []
    for c in policy.chunks:
        if is_instruction_like(c.text):
            quarantined.append(c.id)
            log.warning("quarantined %s (%s): instruction-like text, not indexed", c.id, c.section_title)
        else:
            kept.append(c)
    if not kept:
        raise PolicyRejected(f"{source}: no usable sections left after quarantine")

    with _lock(co):
        previous = store.active_version(co)
        on_disk = set(store.chunks.versions_on_disk(co)) | set(vector_versions(store).get(co, []))
        version = max([previous or 0, *on_disk]) + 1   # never reuse a number
        changes = compare_versions(store.chunks.all(co, previous) if previous else {}, kept)
        started = time.perf_counter()
        try:
            n = build_version(store, co, version, kept)                       # 2-3
            src_dir = store.root / "policies" / co
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / f"v{version}.md").write_text(markdown, encoding="utf-8")   # 4
        except Exception as e:
            log.error("publish %s v%d failed, rolling back: %s", co, version, e)
            _discard(store, co, version)
            raise PublishFailed(f"building {co} v{version} failed ({type(e).__name__}); "
                                f"v{previous} is still active") from e

        store.set_active(co, {                                                # 5. FLIP
            "version": version, "doc_id": policy.doc_id, "title": policy.title,
            "effective_date": policy.effective_date,
            "chunks": n, "quarantined": quarantined, "source": source,
            "changes": {k: (len(x) if isinstance(x, list) else x) for k, x in changes.items()},
            "activated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        log.info("published %s v%d (%d chunks, %d quarantined) in %.2fs; previous v%s; "
                 "changed %s, added %s, removed %s", co, version, n, len(quarantined),
                 time.perf_counter() - started, previous, changes["changed"], changes["added"],
                 changes["removed"])
        if previous is not None:                                              # 6
            if old_version_grace_s > 0:
                delete_later(store, co, previous, old_version_grace_s)
            else:
                delete_version(store, co, previous)   # offline (CLI/tests): nothing in flight
    return PublishResult(co, version, previous, n, quarantined, policy.doc_id,
                         policy.effective_date, changes)


def _discard(store: PolicyStore, company_id: str, version: int) -> None:
    try:
        delete_version(store, company_id, version)
    except Exception as e:   # noqa: BLE001 - startup cleanup will retry
        log.error("could not remove partial %s v%d: %s", company_id, version, e)
    (store.root / "policies" / company_id / f"v{version}.md").unlink(missing_ok=True)


# ------------------------------------------------------------------ startup --

def bootstrap(store: PolicyStore, policies_dir: Path) -> dict[str, int]:
    """Make the store consistent and complete at startup. Returns {company: active version}."""
    from app.ingest import load_policies   # validates ALL seed files first (one bad file stops startup)
    seeds = {p.company_id: p for p in load_policies(policies_dir)}

    # 1. Remove everything that isn't the active version (leftovers of crashes / old layouts).
    for name in legacy_collections(store):
        log.warning("removing pre-versioning collection %s", name)
        store.vectors.delete_collection(name)
    for f in store.chunks.root.glob("*.jsonl"):          # pre-versioning chunk files
        log.warning("removing pre-versioning chunk file %s", f.name)
        f.unlink(missing_ok=True)
    companies = set(store.chunks.companies_on_disk()) | set(vector_versions(store))
    for co in sorted(companies):
        active = store.active_version(co)
        leftovers = (set(store.chunks.versions_on_disk(co)) | set(vector_versions(store).get(co, []))) - {active}
        for v in sorted(leftovers):
            log.warning("removing %s v%d: not the active version (crashed build or old version)", co, v)
            delete_version(store, co, v)
            (store.root / "policies" / co / f"v{v}.md").unlink(missing_ok=True)

    # 2. Every active version must be intact; otherwise rebuild it as a new version.
    for co in store.active_companies():
        v = store.active_version(co)
        try:
            verify_version(store, co, v, expected=store.active_info(co)["chunks"])
        except StoreError as e:
            kept = store.root / "policies" / co / f"v{v}.md"
            seed = seeds.get(co)
            source_text = kept.read_text(encoding="utf-8") if kept.exists() else (
                (policies_dir / seed.source).read_text(encoding="utf-8") if seed else None)
            if source_text is None:
                raise StoreError(f"{co} v{v} is broken ({e}) and there is no source to rebuild it") from e
            log.warning("active %s v%d is not usable (%s); rebuilding from its source", co, v, e)
            publish_policy(store, source_text, source=f"rebuild of v{v}", old_version_grace_s=0)

    # 3. Seed companies that have no version yet (first start, or a new seed file).
    for co, p in seeds.items():
        if store.active_version(co) is None:
            publish_policy(store, (policies_dir / p.source).read_text(encoding="utf-8"), source=p.source)

    return {co: store.active_version(co) for co in store.active_companies()}
