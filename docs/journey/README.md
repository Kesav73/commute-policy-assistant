# Engineering journey

A running log of every experiment and improvement: what we tried, how we measured it, what the numbers said, and what we decided. Newest entries at the bottom of the index. Nothing is deleted; when a later experiment overturns an earlier decision, the new entry says so and links back.

## How every entry is written

1. **Question.** What we want to find out, and why it matters for the brief.
2. **Setup.** What is held fixed, what varies, which data, which command reproduces it.
3. **Results.** A table. Retrieval experiments always report the same three metrics:
   - **recall@1**: share of the gold `required` sections that come back in first place
   - **recall@3**: share of the gold `required` sections in the top 3 (what the LLM sees)
   - **avg latency**: mean time of embedding the question plus searching Chroma, in ms
   - Cross-company leaks are checked on every question as a hard rule (must be 0), not as a metric.
4. **Decision.** What we kept, what we dropped, and the trade-off we accepted.
5. **Commit.** The commit that holds the change, so the result can be reproduced.

**Evaluation data:** `eval/questions.jsonl`, 90 hand-labelled questions (30 per company, 75 answerable). **Scorer:** `python -m eval.run_retrieval`. Same data, same scoring, every time.

## Current best (retrieval)

| Configuration | recall@1 | recall@3 | avg latency | Entry |
|---|---|---|---|---|
| Sub-section chunks + nomic-embed-text-v1.5 (with prefixes), top 3, cosine; separate chunk store | 0.64 | 0.89 | ~33 ms | [08](08-switch-to-nomic.md) |

## Index

| # | Date | Entry | Outcome |
|---|---|---|---|
| 01 | 2026-11-19 | [Embedding speed: Chroma's fixed padding](01-embedding-padding.md) | 440 ms → 5.9 ms per question, identical vectors |
| 02 | 2026-11-19 | [Choosing the LLM](02-llm-selection.md) | Qwen 3.8 27B primary, gpt-oss-120b fallback |
| 03 | 2026-11-19 | [Evidence gate threshold](03-evidence-gate.md) | Kept conservative at 0.75; the LLM decides borderline cases |
| 04 | 2026-11-19 | [First end-to-end evaluation](04-first-e2e-eval.md) | 72/90 pass, 0 leaks, p95 0.83 s; most failures come from retrieval |
| 05 | 2026-11-19 | [Gold set audit and new retrieval baseline](05-gold-set-audit-and-baseline.md) | 4 labels fixed; baseline recall@1 0.55, recall@3 0.82, 5.1 ms |
| 06 | 2026-11-19 | [Separate chunk store](06-separate-chunk-store.md) | Chroma holds ids + vectors only; same recall, ~10% faster |
| 07 | 2026-11-20 | [Embedding models](07-embedding-models.md) | nomic-embed-text-v1.5 best value: recall@1 0.64, recall@3 0.89 (+8 questions), ~35 ms; bge-large ties at 2–3× the cost; switched in 08 |
| 08 | 2026-11-20 | [Switch the app to nomic](08-switch-to-nomic.md) | App now 0.64 / 0.89; gate re-tuned 0.75 → 0.50; image 1.9 GB, RAM ~1.28 GB |
