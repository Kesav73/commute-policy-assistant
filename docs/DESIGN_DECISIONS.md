# Design decisions

Each decision: what we chose, why, what else we considered, and what it costs. Measurements are in [EVALUATION.md](../EVALUATION.md).

## Retrieval

| Decision | Why | Considered instead | Cost |
|---|---|---|---|
| **One chunk per `### N.M` sub-section**, headings prepended | The policies are structured: one rule per chunk, so a citation points to an exact section and nothing is cut mid-rule | Fixed-size chunks with overlap | Needs structured documents; an unstructured PDF would need a different splitter |
| **nomic-embed-text-v1.5**, with its `search_query:` / `search_document:` prefixes, run locally | Best of four models on our evaluation set (recall@3 0.82 → 0.89); free, offline, ~30 ms | MiniLM (faster, less accurate), bge-base, arctic-embed, bge-large (tied, 2× slower, 1.2 GB) | +0.5 GB image, ~30 ms per query |
| **Hybrid search:** vectors + stemmed BM25, merged by rank (RRF, vectors weighted 2:1) | Catches exact terms ("HR Head", "no-show") that embeddings blur; recall@3 0.89 → 0.92 for < 0.5 ms | Cross-encoder re-ranking (best of four only tied, at ~1 s per question) | ~85 lines of code |
| **Query rewriting before retrieval**, given the company's section titles | Everyday wording and multi-part questions become the policy's own topics: recall@3 0.92 → 0.98 | Searching the question as asked | One LLM call (~0.5 s) |
| **Round-robin merge** of the rewritten queries | Each sub-question keeps its own best section; RRF favoured sections the queries shared and lost multi-part answers | RRF merge (multi-part recall@3 0.61) | — |
| **3 chunks read = 3 chunks citable** | Every fact in an answer traces to a citation the employee can see; smaller prompt | Read 5, cite 3 | A question needing a 4th rule stays partly unanswered |
| **Evidence gate** at cosine distance 0.50 | Plain nonsense ("poem about cats") is "not covered" without an answer call | No gate | Rarely fires; the LLM makes most "not covered" calls |

## Using the two LLM calls

| Decision | Why | Considered instead | Cost |
|---|---|---|---|
| **Call 1 = rewrite + input verdict; call 2 = answer (+ may refuse)** | A typical design spends separate calls on rewriting, input screening, answering and a grounding check. With a limit of 2, the input check rides on the rewrite call and the refusal on the answer call; everything else is code | Call 1 answers, call 2 is only a backup | ~0.5 s per question; the free tier allows ~80–100 questions a day |
| **Fallback model (gpt-oss-120b)** | If call 1 fails because the model is down, call 2 goes to a different model, so the service keeps answering | No fallback | When call 1 succeeded, a failed answer gets "please try again" (no 3rd call) |
| **Temperature 0, JSON mode, no hidden reasoning** | Same input → same output; valid JSON guaranteed; reasoning only added latency in our benchmark | Free-text output | — |
| **The model never writes a citation**: it picks labels S1–S3, code maps them to document and section | Citations can't be invented, misspelled or point at another company | Ask the model for section numbers | — |

## Safety

| Decision | Why | Considered instead | Cost |
|---|---|---|---|
| **Company only from the JWT;** the request body rejects any extra field | No request field, header or prompt text can change the company | Company as a parameter | — |
| **One Chroma collection per company (and version)** | Isolation by construction: a missing filter can't leak another company | One shared collection + a filter | More collections |
| **Content kept apart from vectors** (chunk files vs Chroma) | Citations and excerpts never depend on the vector index; ids are re-checked against the company's own file | Text inside Chroma | Two stores kept in step (verified on publish) |
| **Guardrails in three places, code has the final word** | Chunks: instruction-like text quarantined at publish. Input: verdict in call 1. Output: refusal in call 2 + code checks (citations, other company names, injected phrases) | A separate guard-model call (a 3rd call) | — |
| **No number-matching check on answers** | With complete retrieval the facts were right in 63 of 66 answers and none were invented; strict matching would mostly block correct answers ("8 PM" vs "8:00 PM") | Match every number/time against the excerpts | A wrong number would not be caught automatically |
| **Fixed messages for "not covered", refusals and failures** | Consistent, always pointing to the transport desk, never the model's own guess | Model-written messages | — |

## Versions, speed, reliability

| Decision | Why | Considered instead | Cost |
|---|---|---|---|
| **Build the new version beside the live one, then switch one pointer** (`active.json`, atomic rename) | A failed or interrupted update never leaves a half-updated policy; no restart | Update the live collection in place | Old version kept for 10 s after the switch |
| **Each request pinned to one version** | A switch mid-request can't mix two versions | Read "live" at every step | — |
| **Whole document uploaded, change report returned** | Matches "a new version of the policy"; the whole version is validated; the admin still sees which sections changed | Section-level patches | Re-sending unchanged sections (cheap: their vectors are cached) |
| **Answer cache keyed by (company, version, question)** | Employees repeat questions; 0 LLM calls, ~15 ms; an update can never serve an old answer | No cache; "similar question" matching (could answer a different question) | Exact matches only |
| **Hard time limit on every blocking step + a 3 s request limit** | "Please try again" instead of a hang | Rely on library timeouts (they limit each network step, not the total) | Worker threads |
| **Embedded Chroma, one container** | One command to start, nothing else to deploy | A separate vector database | Not built for horizontal scaling |
| **No conversation memory** | Each question stands alone: no stored history to leak, fits the 2-call budget | Chat history | Follow-ups like "what about Saturday?" aren't understood |
| **Deterministic evaluation** (string and section checks) | Same score every run, free to rerun, every failure names what was missing | An LLM judge | A correct answer worded differently can be marked wrong |
