# 16. Policy versioning and atomic updates (R2, R3, A3)

**Date:** 2026-11-20 · **Code:** `app/versioning.py`, `app/store.py`, `app/chunk_store.py`, `app/main.py` (`POST /admin/policies`), `app/pipeline.py` · **Tests:** `tests/test_versioning.py` (18 tests) · **Demo files:** `sample_data/policy_updates/acme_v2.md` (escort 8 PM → 9 PM), `acme_broken.md`

## Question
The brief: *"Ingestion of a policy, including a new version of an existing one. Once a new version is active, only it answers. A failed ingestion must not leave the policy half-updated."* How do we do that on a **running** container, with no restart and no rebuild, so that a half-finished update never serves anything?

## Design: never edit the live version; build beside it, then flip one pointer
```
data/active.json                      {"acme": {"version": 1, ...}}   ← THE switch
data/chunks/acme/v1.jsonl + policy__acme__v1        ← live
admin uploads v2 → build data/chunks/acme/v2.jsonl + policy__acme__v2   (v1 keeps answering)
                 → verify v2 → flip active.json → v2 (os.replace: atomic) → delete v1 after 10 s
```

| Situation | Behaviour |
|---|---|
| Upload while questions arrive | v2 built beside v1; v1 serves until the flip |
| Question in flight at the flip | Finishes on the version it pinned at its start (never a mix) |
| Next question after the flip | v2 only; v1 is deleted 10 s later (longer than the 3 s request limit) |
| Invalid file | 400 before anything is touched |
| Failure while building (embedding, Chroma, verification) | Partial v2 deleted; pointer never moved; v1 fully active |
| Process killed mid-build | Restart: pointer still v1; the leftover v2 is deleted by the startup cleanup |
| Restart after a successful update | v2 kept; **not** reseeded from `policies/` |
| Two uploads at once | Per-company lock: they become v2 and v3 |
| Admin uploads another company's file | 403, nothing changed (company comes from the admin's token) |

- **The update arrives through** `POST /admin/policies` (admin token + markdown) on the running service. `python -m app.ingest FILE.md` does the same with the app stopped.
- **Versions** auto-increment; the uploaded markdown is kept (`data/policies/<co>/v<n>.md`) for audit and rebuilds.
- **Isolation unchanged:** collections stay per company (now per company and version), and the chunk-store check still applies.
- **No Docker rebuild or restart is needed:** code lives in the image, policies in the `app-data` volume.

## Bugs found while verifying
1. **Version pinning outside error handling.** `ask()` read the active version before the guarded retrieval block, so a crashing store would have given a 500 instead of "try again". Caught by an existing test; moved inside the guard.
2. **First question after a restart failed** (found by restarting a real container). Before versioning, startup re-ingested everything, which loaded the embedding model as a side effect. Now startup keeps the existing versions and embeds nothing, so the first question loaded the model (~1.1 s) *inside* the 1 s retrieval limit and got "try again". Fixed with a startup **warm-up** retrieval per company (log: `warm-up (ms): {'acme': 1084.6, 'bolt': 19.3, 'crest': 18.5}`), plus a regression test.

## Verification
**Tests (18):** seeding; new version answers immediately and old never again (also through the full pipeline: `policy_version` 2, "9:00 PM"); other companies untouched; grace period then deletion; invalid file → nothing changes; failure mid-build (vectors fail after the chunk file is written) → rollback, v2 still answers, no v3 leftovers; failed verification → rollback; **crash mid-build then restart** → old version active, leftovers removed; restart after update keeps v2; own-company only; unknown company; quarantine on uploads; concurrent updates → v2 and v3; the API flow (before/after, 403/400/401); startup warm-up.

**Live container** (offline mode, so no LLM quota used):

| Step | Result |
|---|---|
| Ask Acme escort question | v1: "…after **8:00 PM**…" |
| `POST /admin/policies` acme_v2.md | `{"version": 2, "previous_version": 1}`, no restart |
| Ask again | v2: "…after **9:00 PM**…" |
| Upload `acme_broken.md` | 400 `invalid_policy … Nothing was changed`; still v2 |
| `docker restart` | health `{"acme": 2, "bolt": 1, "crest": 1}`; only `v2` files in the volume |
| First question after restart | v2, HTTP 200 in 38 ms (after the warm-up fix) |

223 tests pass; retrieval scorecard unchanged (recall@1 0.63, recall@3 0.92).

## Decision
R2, R3 and A3 are met. The live version is never modified, a new one becomes visible in one atomic step, and every failure path leaves the previous version fully active.
