# 10. Re-ranking with cross-encoders

**Date:** 2026-11-20 · **Command:** `python -m eval.bench_rerank` (~25 min on CPU, including model downloads) · **Results:** `eval/results/rerank_summary.json` · **Code:** `app/retriever.py` gained `rank_candidates()` (the ranking step without the top-3 cut), used by the benchmark

## Question
A cross-encoder reads the question and one chunk *together* and scores how well the chunk answers it, instead of comparing two separately made vectors. Re-ranking the first stage's top candidates with one is the usual biggest single retrieval gain. Does it help here, on top of nomic + hybrid (entry 09), and is its score a better "not covered" signal than distance?

## Setup
18 configurations = 2 baselines + 4 re-rankers × 2 first stages × 2 pool sizes:
- **Re-rankers** (local ONNX via fastembed, chosen for variety):

| Re-ranker | Size | Note |
|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 0.08 GB | classic small cross-encoder (web search QA) |
| jina-reranker-v1-turbo-en | 0.15 GB | different family, built for speed |
| bge-reranker-base | 1.04 GB | larger, English |
| bge-reranker-v2-m3 (int8) | 0.57 GB | newest generation, quantised |

- **First stage:** vector (nomic) or hybrid (nomic + stemmed BM25, entry 09).
- **Pool N:** the first stage's top 10, or all of the company's chunks (19–27). The re-ranker orders them; the top 3 go to the LLM.
- Same chunks, gold set and scorer. Latency = full retrieval (embed + search [+ BM25] + re-rank) after a warm-up.
- **sep** (secondary) = AUC of the top score for answerable vs not-covered questions: 0.5 = no signal, 1.0 = perfect separation. Reference: nomic best distance gives **0.935**.

## Results

| Configuration | recall@1 | recall@3 | avg latency | pass (75) | sep |
|---|---|---|---|---|---|
| no re-rank, vector | 0.64 | 0.89 | 34 ms | 64 | 0.935 |
| **no re-rank, hybrid (current)** | 0.63 | **0.92** | **34 ms** | **66** | 0.935 |
| minilm-L6 (any stage, any N) | 0.59 | 0.88 | 155–372 ms | 62 | 0.874 |
| jina-turbo, vector, N=10 | 0.48 | 0.82 | 196 ms | 56 | 0.924 |
| jina-turbo, N=all | 0.47 | 0.77 | 478 ms | 52 | 0.923 |
| bge-reranker-base (any stage, any N) | 0.62 | 0.87 | 0.9–2.6 s | 60 | 0.876 |
| **bge-v2-m3 int8, vector, N=10** | **0.66** | **0.93** | 983 ms | **66** | 0.953 |
| bge-v2-m3 int8, hybrid, N=10 | 0.63 | 0.91 | 1018 ms | 64 | **0.956** |
| bge-v2-m3 int8, vector or hybrid, N=all | **0.67** | 0.90 | 2.86 s | 64 | 0.951 |

Leaks: 0 everywhere. Several rows are identical across first stage and N: the pool already held the required sections, so the re-ranker alone decided the order.

**Against the current setup (hybrid, no re-rank: 66/75)**

| Re-ranker | Fixed | Broke |
|---|---|---|
| minilm-L6 | acme-22 | acme-06, acme-14, bolt-21, crest-07, crest-18 |
| jina-turbo | 0–1 | 10–14 (mostly paraphrases) |
| bge-reranker-base | acme-22, crest-26 (injection chunk no longer first) | 8 |
| bge-v2-m3 int8, vector, N=10 | acme-30 (typo), bolt-11 (paraphrase) | acme-06, acme-14 |

## Reading the results
- **Re-ranking doesn't beat hybrid here.** The best configuration (bge-v2-m3, vector, N=10) **ties** hybrid at 66/75 (+0.03 recall@1, +0.01 recall@3, i.e. one or two questions: noise on this set) and costs **~1 s per question, ~30× slower**. Three of the four re-rankers make retrieval *worse*.
- **Why:** these re-rankers were trained mostly on web search pairs (MS MARCO). Our chunks are short policy rules with a heading, and nomic + BM25 already rank them well. The re-rankers tend to reward word overlap with the question, which overrules good paraphrase matches: "I moved to a new flat" (acme-14) and "bring a guest" (acme-06) break in almost every re-ranked variant. A cross-encoder also scores each chunk on its own, so it doesn't help questions that need two different sections (multi-hop recall@3 stays at 0.76–0.86).
- **A bigger pool doesn't help:** N=all costs ~3× the time of N=10 and is never better on recall@3.
- **As a "not covered" signal:** bge-v2-m3's score separates slightly better than distance (AUC 0.95 vs 0.935). Not enough to justify ~1 s per question.
- **Budget:** ~1 s of retrieval plus the LLM (p95 ~0.8 s in entry 04) plus a fallback would put the 3 s p95 at real risk. The model would also add ~0.6 GB to the image and a similar amount of RAM.

## Decision
**Not adopted.** Hybrid without re-ranking stays the default. It matches the best re-ranker on accuracy at about 1/30 of the latency, with no extra model. The benchmark stays in the repo (`eval/bench_rerank.py`), so the comparison can be rerun if the policies, questions or available models change.

Also recorded: `rank_candidates()` was split out of `retrieve()` so a later stage can see the full candidate list. Behaviour is unchanged (vector 0.64/0.89, hybrid 0.63/0.92, 155 tests pass).
