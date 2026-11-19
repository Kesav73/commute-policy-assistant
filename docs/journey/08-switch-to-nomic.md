# 08. Switch the app to nomic-embed-text-v1.5 and re-tune the gate

**Date:** 2026-11-20 · **Code:** `app/embeddings.py`, `app/store.py`, `app/retriever.py`, `app/config.py`, `Dockerfile`

## Question
Entry 07 chose nomic-embed-text-v1.5 with its prefixes. Does the app reproduce the benchmark numbers once switched, and what does the switch cost in practice (image, RAM, startup, test time)? The "not covered" gate threshold was tuned on MiniLM distances, so it has to be re-tuned too.

## Change
- **`app/embeddings.py`:** nomic via fastembed, the same library and files as the benchmark. Two entry points: `embed_documents` adds `search_document: ` and `embed_query` adds `search_query: `. Both prefixes and the model live in this one file.
- **Mismatch guard:** each collection records `embedding = "nomic-ai/nomic-embed-text-v1.5|search_document:|search_query:"`. The retriever refuses a collection built differently (`StoreError`, which the pipeline turns into "try again"). A stale index can't silently return wrong results. Tested.
- **Docker:** the model is downloaded at build time and `HF_HUB_OFFLINE=1` is set, so the running container needs no network for embeddings. Verified with `docker run --network none`.
- **Chunk-vector cache:** vectors already computed in the process are reused on re-ingest. Without it the test suite went from ~12 s to ~120 s, since every test that ingests re-embedded all 71 chunks. With it: 9.7 s.
- **Gate threshold:** 0.75 → 0.50 (below).
- **Tests:** the shared test question became "Does a trained security escort travel with women on night drops?". With nomic, Acme §4.2 (routing rules for women "after 8:00 PM") now edges out §4.1 for "Do women get an escort on late drops?". That's a defensible ranking, but unit tests need an unambiguous one. Gate tests use "Write me a poem about cats" (0.555) instead of "capital of France" (0.506, too close to the line).
- `requirements-bench.txt` removed: fastembed is now an app dependency.

## Results: retrieval (same scorer, same 90 questions)

| | recall@1 | recall@3 | avg latency | pass (75) |
|---|---|---|---|---|
| Before: all-MiniLM-L6-v2 | 0.55 | 0.82 | ~4 ms | 56 |
| **After: nomic-embed-text-v1.5 + prefixes (app)** | **0.64** | **0.89** | ~31–35 ms | **64** |

The app matches the benchmark exactly (entry 07), locally and inside Docker. Leaks: 0.

## Gate re-tune
Best-chunk cosine distance with nomic:

| Group | Range |
|---|---|
| 75 answerable gold questions | 0.13–0.49 (median 0.24; worst: crest-30 "esc0rt tme for ladies??" at 0.489) |
| 15 not-covered gold questions | 0.27–0.42 |
| Clearly off-topic probes | weather 0.455, laptop password 0.484, capital of France 0.506, 2+2 0.520, poem 0.555, cricket 0.558 |

| Threshold | Answerable wrongly blocked | Not-covered caught |
|---|---|---|
| 0.30 | 14 | 14 |
| 0.34 | 4 | 12 |
| 0.38 | 1 | 4 |
| 0.42 | 1 | 0 |
| **0.50** | **0** | **0** |

The groups overlap even more than with MiniLM: nomic puts every question fairly close to *some* policy text. Same rule as entry 03: **the gate must never block a real question**, so it sits at **0.50**, the tightest value that blocks none of the 75. It catches nonsense like "poem about cats", "2+2" and "capital of France" with 0 LLM calls; everything else goes to the LLM, which made every not-covered call correctly in entry 04.

**Risk:** the margin over the worst real question is thin (0.489 vs 0.50), and a worse typo could be wrongly blocked. Alternatives for later: drop the gate entirely, or use a better not-covered signal (re-ranker score). Revisit after the re-ranking experiment.

## Cost of the switch

| | MiniLM | nomic |
|---|---|---|
| Docker image | 1.16 GB | **1.9 GB** |
| Container RAM (idle, and after 20 questions) | ~0.6–0.7 GB (est.) | **~1.28 GB**, flat under load |
| Model in RAM (measured in-process) | ~130 MB | ~770 MB |
| Query embedding | ~4 ms | ~31–35 ms (~1% of the 3 s budget) |
| Startup ingest (71 chunks) | ~1 s | ~8 s |
| Test suite | ~12 s | ~10 s (with the vector cache) |

## Decision
Kept. +8 questions retrieved correctly (56 → 64 of 75) for ~30 ms per question, ~0.75 GB more image and ~0.65 GB more RAM. These are acceptable for a single-container service on a laptop. The mismatch guard makes any future model change fail loudly instead of quietly.
