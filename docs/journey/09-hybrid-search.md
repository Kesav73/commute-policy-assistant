# 09. Hybrid search: BM25 keywords + vectors (bonus B1–B3)

**Date:** 2026-11-20 · **Code:** `app/bm25.py`, `app/retriever.py` (`RETRIEVAL_MODE=vector|hybrid`) · **Command:** `RETRIEVAL_MODE=vector python -m eval.run_retrieval` vs `RETRIEVAL_MODE=hybrid …`

## Question
Embeddings capture meaning but can blur exact terms ("no-show", "HR Head", "INR 900"). Does adding keyword search, the brief's own bonus example, retrieve the required sections more often, and at what cost?

## How it works
```
question ─┬─► vector search (nomic)        → ranking of ALL the company's chunks
          └─► BM25 keyword search (stemmed) → ranking of ALL the company's chunks
                       │
        weighted Reciprocal Rank Fusion: score = 2/(60 + vector rank) + 1/(60 + keyword rank)
                       │
                  top 3 → LLM
```
- **BM25** (`app/bm25.py`, ~60 lines, our own): lowercase words minus stopwords, Snowball stemming ("booking"/"booked" → "book"), standard Okapi BM25 (k1 = 1.5, b = 0.75). Built per company from the chunk store, in memory, so it only ever contains that company's text.
- **RRF** fuses by rank only, so a cosine distance never has to be compared with a BM25 score; they live on unrelated scales. k = 60 is the standard constant.
- **The "not covered" gate is unchanged:** it still uses the best *vector* distance, whatever the fused order.
- **Isolation, tightened:** every vector candidate is checked against the company's chunk store *before* fusion. The first version checked only the final top 3. A test that plants a Bolt vector in Acme's index caught that in hybrid mode the foreign id was ranked below the cut-off, so it was never returned but the ISOLATION ALERT never fired either. Now it's dropped and logged in both modes (tested in both).

## Results (75 answerable gold questions)

| Configuration | recall@1 | recall@3 | pass | multi_hop recall@3 |
|---|---|---|---|---|
| vector only (entry 08) | **0.64** | 0.89 | 64 | 0.76 |
| BM25 only | 0.46 | 0.74 | 51 | 0.55 |
| BM25 only, stemmed | 0.60 | 0.84 | 59 | 0.78 |
| hybrid 1:1 | 0.61 | 0.87 | 61 | 0.72 |
| hybrid 1:1, stemmed | 0.61 | **0.92** | 65 | **0.82** |
| **hybrid 2:1, stemmed (chosen)** | 0.63 | **0.92** | **66** | **0.82** |
| hybrid 3:1, stemmed | **0.64** | **0.92** | **66** | 0.80 |

Leaks: 0 in every configuration.

**What the numbers say**
- **Plain hybrid made things worse** (64 → 61). Without stemming, BM25 misses word forms ("book" vs "booking"), and an equal vote let that weaker ranking push correct vector results out of the top 3 (acme-02, acme-15, crest-02, crest-09, crest-18 broke).
- **Stemming is what makes keywords useful:** BM25 alone goes from 51 to 59.
- **Vectors get twice the vote** because alone they are the stronger ranker (recall@3 0.89 vs 0.84). Keywords then mainly add what vectors miss instead of overruling them. 2:1 and 3:1 tie on pass; 2:1 keeps multi-hop at its best and departs less from standard RRF.
- **Fixed versus vector only:** acme-14 ("I moved to a new flat…" → §1.1) and crest-19 ("far away, over INR 900…" → now also §1.3). Nothing broke.

## Cost (B3)

| | vector | hybrid |
|---|---|---|
| Chroma search | 1.48 ms (top 3) | 1.69 ms (all ~27 chunks) |
| BM25 + fusion | — | 0.21 ms |
| Total retrieval (incl. ~30–45 ms embedding) | 42–50 ms | 38–49 ms (no measurable difference) |
| Code | — | ~60 lines BM25 + ~25 lines fusion; one small pinned dependency (`py-rust-stemmers`, already pulled in by fastembed) |

## Caveats
- **+2 questions is a modest gain,** close to what noise could produce on 75 questions. What makes it credible: multi-hop recall@3 improves in every stemmed variant (0.76 → 0.78–0.82), and nothing broke.
- **The fusion weight was picked on the same 75 questions we report on.** Only 3 weights were tried, and the choice follows from the measured strength of each ranker. Still, a held-out set would confirm it.
- recall@1 drops by 0.01 (one question no longer ranks first), so the top-1 ordering is marginally noisier.

## Decision
**Hybrid (stemmed, 2:1) is the default** (`RETRIEVAL_MODE=hybrid`); `RETRIEVAL_MODE=vector` switches it off. Kept because it costs under 0.5 ms and ~85 lines, improves recall@3 and multi-hop, and broke nothing.

Still missing (9): acme-18, 19, 20, 22, bolt-19 (multi-hop; several need a section that the retrieved one explicitly points to), acme-30 (typo), bolt-11 (paraphrase), bolt-20, crest-26 (injection chunk ranks first). Next: re-ranking (entry 10).
