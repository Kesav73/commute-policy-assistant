# Requirements and checklist

The brief (MoveInSync AI Developer case study B, *Commute Policy Assistant*), condensed, and where each requirement is met.

## The brief in short

- Every client company has its own transport policy: eligibility, night-shift escorts, cut-offs, cost caps, approvals, weekend trips, guests, no-show penalties. Policies change a few times a year; **once a new version is live, the old one is dead**.
- What goes wrong today: desk staff quote **the wrong company's rule**; employees don't trust an answer **without a source**; some questions have **no answer** in the policy; a document or a user may contain text that tries to **steer the assistant** off its job.
- Build an assistant that answers **only from the employee's own company's current policy**, **cites document and section**, **says plainly when the policy is silent**, and **never answers from another company's document**.
- **Stack:** Python, FastAPI, Chroma, any embedding model and LLM, JWT. Free tiers only. Tests may use a fake LLM.
- **Two numbers:** an answer within **3 s at p95** (LLM included); **at most 2 LLM calls** per question and **at most 3 cited chunks** per answer, and the README must say how.

## Checklist

| # | Requirement | Status | Where |
|---|---|---|---|
| R1 | Three short policies, overlapping topics, deliberately contradictory rules | ✅ | `policies/`, [POLICY_MATRIX.md](POLICY_MATRIX.md) |
| R2 | Ingest a policy, including a new version; once active, only it answers | ✅ | `POST /admin/policies`, `app/versioning.py` |
| R3 | A failed ingestion never leaves the policy half-updated | ✅ | build beside → verify → atomic switch; tests incl. a crash mid-build |
| R4 | Question API behind a JWT that carries the company | ✅ | `POST /ask`, `app/auth.py` |
| R5 | Retrieval restricted to that company on the server, no client override | ✅ | collection per company, id re-check, body rejects extra fields |
| R6 | Grounded answers citing document and section | ✅ | `app/answerer.py`: labels S1–S3 mapped to sections by code |
| R7 | Explicit "not covered" when evidence is missing | ✅ | evidence gate, model status, output checks |
| R8 | No obedience to instructions in retrieved documents (or questions) | ✅ | quarantine at publish, input verdict, prompt rules, output checks |
| R9 | Can't answer → a plain message pointing to the transport desk | ✅ | `app/messages.py` |
| R10 | Slow retrieval or LLM failure → a clear "try again", never a hang or half answer | ✅ | time limit on every stage + 3 s request limit (`app/timeouts.py`) |
| R11 | Evaluation set ≥ 10 verified questions incl. paraphrases, cross-company traps, not-covered; JSONL | ✅ | `eval/questions.jsonl` (90), `eval/results/e2e.jsonl` |
| R12 | Report retrieval and answer quality with our own scoring | ✅ | `eval/run_retrieval.py`, `eval/run_e2e.py`, [EVALUATION.md](../EVALUATION.md) |
| R13 | README with design and trade-offs | ✅ | [README](../README.md), [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) |
| C1 | p95 < 3 s, LLM included | ✅ | measured **1.25 s** |
| C2 | ≤ 2 LLM calls per question | ✅ | `app/pipeline.py`, tested |
| C3 | ≤ 3 cited chunks per answer | ✅ | 3 chunks retrieved, ≤ 3 citations kept |
| C4 | README says how we stay under C2 and C3 | ✅ | README: "How we stay within 2 LLM calls and 3 cited chunks" |
| Bonus | One retrieval improvement, before vs after, with its cost | ✅ | embeddings, hybrid, re-ranking (rejected), query rewriting: [EVALUATION.md](../EVALUATION.md) |

**"What good looks like"**

| Criterion | Status |
|---|---|
| A cross-company question returns nothing from the other company, every time | ✅ 0 leaks in every run; isolation tests |
| Every answer carries a citation you can check | ✅ no citation → the answer isn't shown |
| After a policy update, only the new version answers | ✅ tests + live demo |
| The score can be rerun, with a before and after | ✅ one command per scorer |

**Deliverables:** GitHub repository · one-command start (`docker compose up`) · README · evaluation report · evaluation set as JSONL · 2-minute screen recording.
