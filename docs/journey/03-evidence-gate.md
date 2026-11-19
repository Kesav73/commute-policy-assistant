# 03. Evidence gate threshold

**Date:** 2026-11-19 · **Commit:** `8ecc7c2` (feat(retrieval)) · **Code:** `app/retriever.py`, setting `RETRIEVAL_MAX_DISTANCE`

## Question
If even the best retrieved chunk is far from the question, we can say "not covered" without calling the LLM: zero calls, instant, and no chance of a confident guess. Where should the cut-off be?

## Setup
Cosine distance of the best chunk, for the 75 answerable and the 15 not-covered gold questions. A sweep over thresholds counts real questions wrongly blocked and not-covered questions correctly caught.

## Results

| Threshold | Answerable wrongly blocked (of 75) | Not-covered caught (of 15) |
|---|---|---|
| 0.55 | 12 | 14 |
| 0.60 | 6 | 11 |
| 0.65 | 3 | 8 |
| 0.70 | 1 | 4 |
| **0.75** | **0** | **0** |
| 0.80 | 0 | 0 |

Real matches sit around 0.38–0.52 and off-topic questions around 0.77–0.87, but the middle overlaps. Examples: a valid Bolt guest question scores 0.68; "can I bring my dog" (not covered) scores 0.61; a not-covered Crest cancellation question scores 0.50, because the booking cut-off sections look similar.

## Decision
Keep the gate **conservative at 0.75**. It never blocks a real question; the LLM makes the final "not covered" call from the text. Blocking a real question is worse than spending one LLM call on an unanswerable one.

**Open:** retune this after every retrieval change, since a better embedding model shifts the distances. Distance alone can't separate the two groups well with the current model.
