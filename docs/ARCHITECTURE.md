# Architecture

How the Commute Policy Assistant is built: the components, the two flows (asking and publishing), storage, time limits and failure handling. *Why* each choice was made is in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

## At a glance

```
                ┌──────────────────────────── FastAPI service (one container) ─────────────────────────────┐
 Employee ─JWT─►│ POST /ask ─► auth ─► cache ─► LLM call 1 ─► retrieval ─► gate ─► LLM call 2 ─► checks ─► │─► answer + citations
                │                                (rewrite +    (hybrid,              (answer)      (code)    │
                │                                 input check)  company only)                              │
 Admin ───JWT──►│ POST /admin/policies ─► validate ─► build new version beside live one ─► verify ─► switch │
                │                                                                                          │
                │  storage (app-data volume):  active.json  ·  chunks/<company>/v<n>.jsonl  ·  chroma/      │
                └──────────────────────────────────────────────────────────────────────────────────────────┘
                                         ▲ LLM: Groq (qwen3.8-27b, fallback gpt-oss-120b) or a fake LLM offline
```

## Components (`app/`)

| Module | Job | Uses the LLM? |
|---|---|---|
| `main.py` | The 4 endpoints, request/response models, error shape, startup, the 3 s request limit | — |
| `auth.py` | Issue (mock) and verify JWTs; company and role come only from the token | — |
| `pipeline.py` | The `/ask` flow, the 2-call budget, deadlines, the fallback model | — |
| `rewriter.py` | **LLM call 1:** search queries + input verdict | ✅ |
| `retriever.py` + `bm25.py` | Hybrid search in the caller's company only; round-robin merge | — |
| `embeddings.py` | nomic-embed-text-v1.5 with query/document prefixes | — |
| `answerer.py` | **LLM call 2:** answer prompt, output parsing, label → citation | ✅ |
| `guardrails.py` | Instruction-like text detection; other-company and injected-text checks | — |
| `store.py` | Chunk files, Chroma collections, the live-version switch | — |
| `versioning.py` | Publish a version safely; startup bootstrap; CLI | — |
| `policy_parser.py` | Markdown policy → one chunk per sub-section | — |
| `answer_cache.py` | Repeated questions answered with 0 LLM calls | — |
| `llm.py` | Groq client with a total-time limit; fake LLM | — |
| `timeouts.py` | Hard time limit for any blocking step | — |
| `config.py`, `messages.py` | Settings; every message an employee can see | — |

## Answering a question: `POST /ask`

| # | Step | What happens | If it goes wrong |
|---|---|---|---|
| 1 | **Auth** | Verify the JWT (HS256, pinned) → `company_id` + role, from the token only | 401 |
| 2 | **Input check** | Body is exactly `{"question"}`, 3–500 characters | 422 |
| 3 | **Answer cache** | Key = (company, live version, normalised question) → hit: answer, 0 LLM calls | a cache error is ignored |
| 4 | **Pin version** | Read the company's live version once; every later step uses it | no policy → "not covered" |
| 5 | **LLM call 1** | Rewrite into 1–3 queries using the company's **section titles**, and judge the question: `ok` / `injection` / `other_company` / `off_topic` | pure attack → refusal (1 call); call failed → search the original question; model down → call 2 uses the fallback model |
| 6 | **Retrieval** | Per query: vector rank + BM25 rank, fused (RRF 2:1); queries merged round-robin → **top 3** | > 1 s → "please try again" |
| 7 | **Evidence gate** | Closest match farther than 0.50 → "not covered" | — |
| 8 | **LLM call 2** | Answer from excerpts S1–S3, or `not_covered` / `blocked` | failure → "please try again" (no 3rd call) |
| 9 | **Checks (code)** | Labels ⊆ S1–S3, 1–3 citations, no other company named, no injected text repeated | → "not covered" |
| 10 | **Citations** | Code maps labels → `{doc_id, title, section, section_title}` from the chunk files | — |
| 11 | **Respond** | `{status, answer, citations, request_id, policy_version, cached}`; stored in the cache | whole request > 3 s → "please try again" |

**LLM calls per question:** 0 (cache hit) · 1 (attack, off-topic, or nothing relevant) · 2 (normal) · **never 3**.

**Example, Acme:** *"If I cancel my drop one hour before, will I be suspended?"*
```
call 1 → {"verdict": "ok", "queries": ["cancellation window", "no-show penalty suspension"]}
search → [§3.1 Cancellation Window, §9.2 No-Show Penalty, §3.3]        (round-robin: each query's best first)
call 2 → "Cancelling one hour before the trip is treated as a no-show. You will only be suspended if this is
          your second no-show in the calendar month..."   citations: §3.1, §9.2
```

## Publishing a policy: `POST /admin/policies`

```
 admin token + full markdown
   │
   ├─ 1. validate: format, sections, company known, = the admin's company     bad → 400 / 403, nothing touched
   ├─ 2. quarantine sections that give orders to an AI (never indexed)
   ├─ 3. build v(n+1) BESIDE the live v(n): chunk file + Chroma collection    (v(n) keeps answering)
   ├─ 4. verify: same ids in both, expected count, embedding model, a search works
   ├─ 5. switch active.json → v(n+1)        ONE atomic file rename: every request from now on gets v(n+1)
   └─ 6. delete v(n) after 10 s             (a request that started on it can finish)
   any failure in 3–4 → v(n+1) removed, v(n) still live          response includes changed/added/removed sections
```

- **No restart and no rebuild:** code is in the image, policies are in the data volume.
- **One request = one version:** the version is pinned at step 4 of `/ask`, so a switch mid-request can't mix two.
- **Startup (`bootstrap`):** removes anything not named in `active.json` (a crashed build), seeds v1 only for companies without a version, keeps existing versions, then warms up the embedding model.

## Storage (`app-data` volume)

```
data/
  active.json                         {"acme": {"version": 2, "chunks": 27, ...}, ...}   ← the live-version switch
  chunks/acme/v2.jsonl                one line per chunk: id, company, doc, title, section, section_title, text
  chroma/  policy__acme__v2           chunk ids + 768-number vectors only (cosine distance)
  policies/acme/v2.md                 the uploaded file, kept for audit
  jwt_secret                          generated on first start if JWT_SECRET isn't set
```

**A chunk:** `acme-transport-policy:4.1` · section `4.1` · title `Night-Shift Safety > Escort Requirement` · text = title + rule body.

## Isolation: four layers

| Layer | How |
|---|---|
| Identity | `company_id` comes only from a verified token; the request body can't carry one |
| Scope | Only `policy__<company>__v<n>` is opened; the rewriter's context is that company's titles only |
| Re-check | Every id Chroma returns must exist in that company's chunk file, or it is dropped and logged |
| Citations | Built by code from the retrieved chunks; the model only picks S1–S3 |

## Time limits

| Stage | Limit | When hit |
|---|---|---|
| Retrieval | 1.0 s (and 0.6 s kept for the answer) | "please try again" |
| LLM call 1 (rewrite) | 1.2 s, total time | search the original question |
| LLM call 2 (answer) | what's left of 2.8 s (≤ 2.0 s) | "please try again" |
| Whole request | 3.0 s | "please try again" |

A blocking step runs in a worker thread and the caller stops *waiting* at the limit (`timeouts.py`): a thread can't be cancelled, and waiting for it would itself be a hang.

## What an employee can see

| Case | Message | HTTP |
|---|---|---|
| Answer | the answer + citations | 200 |
| Policy is silent / nothing relevant | "Your company's transport policy doesn't cover this. Please contact your transport desk for help." | 200 |
| Attack or another company's policy | "I can only answer questions about your own company's transport policy. For anything else, please contact your transport desk." | 200 |
| LLM or search failed / too slow | "We couldn't get an answer right now. Please try again in a moment…" | 503 + `Retry-After` |
| Bad token / bad question | "Please sign in…" / "Please enter a question between 3 and 500 characters." | 401 / 422 |
| Anything unexpected | "Something went wrong on our side…" (details only in the server log) | 500 |
