# 07. Embedding models

**Date:** 2026-11-20 · **Command:** `pip install -r requirements-bench.txt && python -m eval.bench_embeddings` · **Results:** `eval/results/embeddings_summary.json`

## Question
With chunking fixed (one chunk per sub-section, headings prepended), which embedding model retrieves best, and what does it cost?

## Setup
- **Held fixed:** the 71 chunks the app ingests; Chroma cosine HNSW, one collection per company; top 3; the 90 gold questions; the scorer from `eval/run_retrieval.py`.
- **Varied:** the model, and whether its query/passage prefixes are used.
- **Four models, picked to differ as much as possible:**

| Model | Size / dims | Family | Input style |
|---|---|---|---|
| all-MiniLM-L6-v2 (baseline) | 22M / 384 | sentence-transformers, general similarity | none |
| snowflake-arctic-embed-s | 33M / 384 | Snowflake, trained for search | query instruction |
| bge-base-en-v1.5 | 110M / 768 | BAAI, larger | query instruction |
| nomic-embed-text-v1.5 | 137M / 768 | Nomic, long-context | `search_query:` / `search_document:` |

e5-small-v2 was the first pick for the search-trained small slot, but fastembed only ships the 2.2 GB multilingual E5, so arctic-embed-s took its place. All models run locally on CPU through ONNX (fastembed); the baseline uses the app's own MiniLM code, so its row matches the app exactly.

Also measured for the winner: the **int8-quantised** nomic model (same model, 0.13 GB instead of 0.52 GB).

## Results

| Variant | recall@1 | recall@3 | avg latency | pass (75) |
|---|---|---|---|---|
| all-MiniLM-L6-v2 (baseline) | 0.55 | 0.82 | 3–5 ms | 56 |
| arctic-embed-s | 0.56 | 0.82 | 6–10 ms | 57 |
| arctic-embed-s + prefix | 0.57 | 0.84 | 7–10 ms | 58 |
| bge-base-en-v1.5 | 0.55 | 0.84 | 18–24 ms | 58 |
| bge-base-en-v1.5 + prefix | 0.57 | 0.87 | 25–35 ms | 61 |
| nomic-embed-v1.5 | 0.63 | **0.89** | 25–32 ms | **64** |
| **nomic-embed-v1.5 + prefix** | **0.64** | **0.89** | 31–40 ms | **64** |
| nomic-embed-v1.5 int8 | 0.60 | 0.88 | 11–19 ms | 62 |
| nomic-embed-v1.5 int8 + prefix | 0.59 | 0.87 | 16 ms | 61 |

Recall is deterministic (identical over 4 runs). Latency is the range over the runs: embedding the question + Chroma search, after a warm-up.

**recall@3 by category**

| Variant | direct | paraphrase | same_section | multi_hop | edge | cross_company | injection |
|---|---|---|---|---|---|---|---|
| MiniLM (baseline) | 0.92 | 0.81 | 0.90 | 0.72 | 0.60 | 1.00 | 0.50 |
| arctic-s + prefix | 0.97 | 0.81 | 0.90 | 0.64 | 0.80 | 1.00 | 1.00 |
| bge-base + prefix | 1.00 | 0.92 | 0.85 | 0.74 | 0.80 | 0.67 | 0.50 |
| nomic + prefix | 1.00 | 0.92 | 0.90 | 0.76 | 0.80 | 1.00 | 0.50 |

**Which questions changed against the baseline**

| Variant | Fixed | Broke | Net |
|---|---|---|---|
| arctic-s | 9 | 8 | +1 |
| arctic-s + prefix | 5 | 3 | +2 |
| bge-base | 9 | 7 | +2 |
| bge-base + prefix | 8 | 3 | +5 |
| nomic | 9 | 1 (acme-21) | **+8** |
| nomic + prefix | 9 | 1 (acme-22) | **+8** |

nomic fixes acme-06, acme-08, acme-09, bolt-21, bolt-30, crest-01, crest-09 and crest-18/19: mostly paraphrases ("skip my cab" → no-show, "call off" → cancellation) and the typo question "wat hapens if i cancle late".

**Secondary measurements**

| Variant | model size | load | index 71 chunks | gate gap* |
|---|---|---|---|---|
| MiniLM | 0.09 GB | <0.1 s | 0.8 s | 0.21 |
| nomic + prefix | 0.52 GB | ~1 s | 7–10 s | 0.11 |
| nomic int8 + prefix | 0.13 GB | ~0.3 s | 3–4 s | 0.11 |

\* gate gap = mean best distance of not-covered questions minus that of answerable ones. Larger means a distance cut-off separates them more easily.

## Reading the results
- **nomic is the clear winner:** +8 questions (56 → 64 of 75), recall@1 +0.09, recall@3 +0.07. That's far beyond the ~0.03 noise level of a 75-question set, and it broke only one question.
- **Prefixes help bge and arctic, and are neutral for nomic** on recall@3 (+0.01 recall@1). The nomic model card says to use them, so we keep them.
- **int8 costs 2–3 questions,** all on multi_hop, for ~2× speed and a 4× smaller model. The speed doesn't matter here: 35 ms is about 1% of the 3 s budget.
- **Bigger isn't automatically better:** bge-base (110M) gains less than nomic, and without its prefix gains almost nothing.
- **Cost of nomic:** query embedding goes from ~5 ms to ~35 ms. The Docker image grows by about 0.5 GB. The app gains a dependency (fastembed). The gate gap halves (0.21 → 0.11), so the evidence gate must be re-tuned after switching.
- **Caveat:** the winner was picked on the same 75 questions we report on. That's acceptable for choosing among 4 models, but the number may be slightly optimistic. A synthetic dev set (TODO) would fix this.

## Scope: why only these four, and not bigger or smaller models
Four models was a deliberate limit, chosen for variety (family, size 22M–137M, input style) rather than size. The limits that shaped it:
1. **Local, CPU, free.** The question is embedded on every request, inside the 3 s budget, on the reviewer's machine. A hosted embedding API would add a network call and a second quota to every question.
2. **The model ships in the Docker image.** MiniLM is 0.09 GB; large models are 1–2.4 GB, which means a slower build and a heavier start for the reviewer.
3. **Available in fastembed** (ONNX, no PyTorch). This is why e5-small-v2 dropped out.

Not tested:

| Group | Examples | Expected trade-off |
|---|---|---|
| Smaller | potion-base-8M (static, ~0.1 ms), arctic-embed-xs | Faster than MiniLM, usually less accurate; latency is already not a problem |
| Larger | bge-large, mxbai-embed-large, embeddinggemma-300m, Qwen3-Embedding-0.6B | Often better on public benchmarks; ~100–300 ms per question on CPU, 0.6–2.4 GB |

**Why bigger models are unlikely to change the picture:** of the 11 questions nomic still misses, 6 need two or three *different* sections in the top 3 (acme-18/19/20, bolt-19, bolt-20, crest-16). One question vector can't sit close to several unrelated chunks at once, whatever the model. These need a retrieval technique (hybrid search, re-ranking, parent-section context), which are the next experiments. Expected gain from a large model: 1–3 questions at 5–10× the latency and image size. That's an estimate, not a measurement; an extended sweep can be run with the same harness if needed.

## Decision
**Recommended:** nomic-embed-text-v1.5, full precision, with its prefixes. It's the best on both recall metrics, and the extra ~30 ms is negligible against the 3 s budget.

**Status:** benchmark recorded. The app still runs MiniLM until the switch is made and verified, which needs a new embedding module, the Docker image updated, the gate re-tuned and the eval re-run.
