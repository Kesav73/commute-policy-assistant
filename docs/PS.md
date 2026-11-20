# Problem Statement — Case Study B: Commute Policy Assistant

> Source: `casestudies/Case Study B – Commute Policy Assistant.docx` (MoveInSync, AI Developer campus case study B).
> This file restates the brief in full, then breaks it into a numbered checklist (R-, C-, D- IDs) that the code, tests and README will refer back to.

---

## 1. Background

MoveInSync runs employee transport for large companies in India. Every working day the platform moves people to and from office on fixed rosters: shared cabs, shuttles, transport desks, cab vendors, drivers — and **each client company's own policy sits on top of all that**. MoveInSync is building AI assistants into the platform; this case study is a small, real slice of that work.

### The domain

Every client company writes its own **transport policy**, usually a 10–30 page document owned by Admin or HR. It covers:

| Topic | Example |
|---|---|
| Eligibility | By grade, shift, or distance from office |
| Night-shift rules | Security escort for women on drops after **8 PM at one company, 10 PM at another** |
| Cut-off timings | Booking / cancellation deadlines |
| Cost caps | Limits on trip cost |
| Ad-hoc trip approvals | Who approves an unscheduled or airport trip |
| Weekend & holiday pickups | Allowed or not, and under what conditions |
| Guest travel | Whether non-employees can ride |
| No-show penalties | What happens after repeated no-shows |

Policies change **a few times a year**. **Once a new version is live, the old one is dead.**

Employees ask the transport desk the same questions every day:
- "Am I eligible for a 9:30 PM drop?"
- "Can I book a cab on Saturday?"
- "Who approves an airport trip?"
- "What happens if I no-show twice?"

---

## 2. The challenge (what goes wrong today)

1. **Wrong company's rule.** Desk executives serve many companies at once and, under pressure, quote another company's rule.
2. **No trust without a source.** Employees don't trust an answer unless they can see *which part of the policy* it came from.
3. **Silent policies.** Some questions have no answer in the policy. The honest reply is *"the policy does not say"* — not a confident guess.
4. **Adversarial text.** A policy document, or a user, may contain text that tries to steer the assistant off its job (prompt injection).

---

## 3. What we build

An assistant that:
- answers **from the employee's own company's current policy** only,
- **cites the document and section** for every answer,
- **says plainly when the policy is silent**,
- **never answers from another company's document**, whatever the request or the retrieved text tries to make it do.

**Correctness is measured, not felt:** one fixed set of questions, scored the same way every time something changes.

---

## 4. Hard constraints ("two numbers to design around")

| ID | Constraint | Value |
|---|---|---|
| **C1** | Latency | Answer within **3 seconds at p95**, measured on our machine, **LLM call included** |
| **C2** | LLM calls | **At most 2 LLM calls per question** |
| **C3** | Citations | **No more than 3 cited chunks per answer** |
| **C4** | Documentation | README must explain **how we stay under** C2 and C3 |

---

## 5. Functional requirements

| ID | Requirement | What "done" means |
|---|---|---|
| **R1** | **Three policy documents** | Three short policies for three fictional companies, with **overlapping terms** and **deliberately contradictory rules**, so any cross-company leak is visible. |
| **R2** | **Ingestion + versioning** | Ingest a policy, including **a new version of an existing one**. Once a new version is active, **only it answers**. |
| **R3** | **Atomic ingestion** | A **failed ingestion must not leave the policy half-updated** — the previous version stays fully active. |
| **R4** | **Authenticated question API** | A question API behind a **JWT that carries the company**. |
| **R5** | **Server-side tenant isolation** | Retrieval is restricted to the token's company **on the server**, with **no client-controlled override** (no request field, header, or prompt text can change it). |
| **R6** | **Grounded, cited answers** | Every answer cites **document and section**. |
| **R7** | **"Not covered"** | An explicit "not covered" reply when evidence is missing. |
| **R8** | **Injection resistance** | **No obedience to instructions found inside retrieved documents** (or in the user's question). |
| **R9** | **Graceful "can't answer"** | When the assistant can't answer, a **plain message saying so** that **points to the transport desk**. |
| **R10** | **Graceful failure** | When retrieval is slow or the LLM fails: a clear **"try again"** — never a hang, never a half answer, never a stack trace. |
| **R11** | **Evaluation set** | **≥ 10 questions with verified answers**, including **paraphrases**, **cross-company traps**, and **"not covered"** cases, in the required JSONL format (§7). |
| **R12** | **Scoring** | Report **retrieval quality** and **answer quality** using scoring **we define**; rerunnable. |
| **R13** | **README** | Design and trade-offs. |

### Bonus (optional, scored extra)

| ID | Bonus |
|---|---|
| **B1** | Implement and **measure one retrieval improvement**: hybrid search, reranking, query rewriting, *or* a chunking change. |
| **B2** | Compare **before vs after on the same evaluation set**. |
| **B3** | State what it **cost in latency and complexity**. |

---

## 6. Required stack

| Layer | Requirement |
|---|---|
| Language | **Python** |
| API | **FastAPI** |
| Vector store | **Chroma** |
| Models | **Any** embedding model and **any** LLM |
| Auth | **JWT** |

---

## 7. General rules (apply to all case studies)

| ID | Rule |
|---|---|
| **G1** | Any LLM provider. Free tiers (Gemini, Groq) or a local model are enough. **Do not spend money.** |
| **G2** | Tests **may use a fake LLM**. Only the demo needs a real one — **a quota error must never cost a passing test**. |
| **G3** | **Nothing is provided.** Build everything: sample data, every mock service, and the **JWT issuer**. A JSON file or SQLite behind a function is fine. Don't build a realistic backend — spend time on the requested behaviour. |
| **G4** | AI coding tools allowed, but **you must be able to explain and change every line** in the viva. |
| **G5** | **Never commit API keys.** |
| **G6** | Submission stays ours; used only for evaluation. |
| **G7** | Scored on **correctness, reasoning, testing, and explanation**. Framework count and line count earn nothing. |

### What they look for
- Not a chatbot that only works on the happy path — **how we think when things go wrong**.
- **Use the LLM where it earns its place**; say where **plain code is the better tool**.
- Handle **bad input, failing tools, strange model output** without falling over.
- **Prove correctness with rerunnable tests**; report what failed and why.
- **Treat model output as untrusted** — authorisation, data isolation and business rules live **outside the model**.
- **Explain trade-offs.** Simple and maintainable beats clever, every time.

---

## 8. Deliverables

| ID | Deliverable | Detail |
|---|---|---|
| **D1** | GitHub repository | Public, or shared with the email in the mail. |
| **D2** | **One-command start** | `docker compose up` is ideal; README must say which command. |
| **D3** | README | How to run, design decisions, trade-offs (incl. C4). |
| **D4** | Evaluation report | Test scenarios, method, results, **known failures**, limitations, and **at least one trade-off and why**. |
| **D5** | Evaluation set (JSONL) | One line per case, **exactly these fields**: `id`, `input`, `expected`, `actual`, `pass`, `notes`. Judging method is ours; **format is fixed** (they compare submissions side by side). |
| **D6** | Screen recording | ~2 minutes, showing it working. |

---

## 9. How we are evaluated

1. **Ten-minute gate check** — Does it start with one command? Do the tests pass? Does the evaluation set run? **Fail this and the submission is not reviewed further.**
2. **Full review** — requirements and evaluation report, **bonus** for the optional extension.
3. **Viva (shortlisted, 15–20 min)** — explain **one design decision**, then make **one small live change**. **The viva weighs more than the repository.**

---

## 10. "What good looks like" (acceptance criteria)

| ID | Acceptance criterion | How we will prove it |
|---|---|---|
| **A1** | A cross-company question returns **nothing from the other company, every time**. | Automated test + eval cases asserting no chunk/citation from another company. |
| **A2** | **Every answer carries a citation** you can check against the document. | Citation validated in code against retrieved chunks; eval checks section IDs. |
| **A3** | After a policy update, **only the new version answers**. | Test: ingest v2 → old-version facts never returned; failed v3 ingest → v2 still active, no v3 leftovers. |
| **A4** | You can **rerun the score**, and show a **before and after**. | One command runs the eval and prints a scorecard; baseline vs bonus numbers in the report. |

---

## 11. Checklist (traceability)

Status is updated as we build.

| ID | Item | Status |
|---|---|---|
| R1 | 3 policies with contradictions | ✅ `policies/`, `docs/POLICY_MATRIX.md` |
| R2 | Versioned ingestion | ✅ `POST /admin/policies`, versions + atomic pointer (`app/versioning.py`) |
| R3 | Atomic ingest / rollback | ✅ build beside → verify → flip; failure/crash leaves the old version (tests) |
| R4 | JWT question API | ✅ `POST /ask` behind `current_principal` |
| R5 | Server-side company filter, no override | ✅ collection from token; body field → 422; query/header ignored (tests) |
| R6 | Document + section citations | ✅ `app/answerer.py` (labels → real sections, in code) |
| R7 | "Not covered" | ✅ gate (0 calls) + model status + downgrade of unsupported answers |
| R8 | Injection resistance | 🟡 isolation + prompt rules + citation checks; to be measured in eval |
| R9 | "Contact transport desk" message | ✅ `app/messages.py` |
| R10 | "Try again" on slow retrieval / LLM failure | ✅ 503 + Retry-After; 500 handler without stack trace |
| R11 | ≥10-question JSONL eval set | ☐ |
| R12 | Retrieval + answer scoring | ☐ |
| R13 | README design + trade-offs | ☐ |
| C1 | p95 < 3 s measured | ☐ |
| C2 | ≤ 2 LLM calls / question | ✅ enforced + tested |
| C3 | ≤ 3 cited chunks | ✅ top_k=3 + cap in validator |
| C4 | README explains C2/C3 | ☐ |
| B1–B3 | Retrieval improvement, before/after, cost | ☐ |
| G2 | Tests run on fake LLM | ✅ conftest forces `LLM_PROVIDER=fake` |
| G5 | No keys committed | ☐ |
| D1 | GitHub repo | ✅ github.com/Kesav73/commute-policy-assistant (private for now) |
| D2 | One-command start | ✅ `docker compose up` (works with or without a key) |
| D3–D6 | README, report, JSONL, video | 🟡 JSONL ✅ (`eval/results/retrieval.jsonl`); README / report / video pending |
