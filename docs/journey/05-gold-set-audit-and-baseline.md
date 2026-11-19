# 05. Gold set audit and new retrieval baseline

**Date:** 2026-11-19 · **Commits:** `40e31eb`, `b29caee` · **Command:** `python -m eval.run_retrieval`

## Question
Before tuning anything, are the labels themselves right? A wrong label either hides a real failure or punishes a correct result. And which metrics should we compare variants on?

## Gold set audit
All 90 questions were checked line by line against the three policies. 86 were correct; 4 changed:

| Case | Problem | Fix |
|---|---|---|
| bolt-19 | "Change my drop from 7 PM to 9 PM, what's the deadline?" §2.3 says the change must also be inside the original trip's cancellation window (§3.1, 1 hour), so the real deadline is 6 PM. §3.1 was not labelled. | required = §2.3, §3.1, §4.2 |
| crest-09 | "Sunday cab, what's the process?" The process itself (§4.2) was only labelled helpful. | required = §4.1, §4.2 |
| crest-20 | "8 PM drop next Monday" hides an eligibility trap: a shift ending at 8 PM isn't a night shift (§1.1), so the gold answer was wrong. | Question changed to a 10 PM drop |
| crest-26 | "Are trips free?" The policy caps trips at INR 900 but never says who pays. | Kept, with a note that "not covered" is also defensible |

## Metrics from now on
Three metrics for every retrieval experiment: **recall@1, recall@3, average latency**. Dropped hit@1, MRR, precision@3 and the gate sweep from the headline. Note that recall@1 cannot reach 1.0 on questions that need two or three sections, so multi-hop recall@1 is low by definition.

The end-to-end eval (entry 04) and the LLM benchmark (entry 02) were removed, to be rebuilt in a structured way later.

## Baseline
Configuration: sub-section chunks with headings prepended, all-MiniLM-L6-v2 (ONNX, 384-dim), Chroma cosine, top 3.

| | recall@1 | recall@3 | avg latency |
|---|---|---|---|
| **Overall (75 answerable)** | **0.55** | **0.82** | **5.1 ms** |
| direct | 0.78 | 0.92 | 4.4 ms |
| paraphrase | 0.56 | 0.81 | 5.4 ms |
| same_section | 0.42 | 0.90 | 4.6 ms |
| multi_hop | 0.33 | 0.72 | 6.2 ms |
| edge | 0.40 | 0.60 | 4.2 ms |
| cross_company | 0.67 | 1.00 | 4.6 ms |
| injection | 0.50 | 0.50 | 5.8 ms |

56/75 answerable questions get every required section in the top 3. Leaks: 0.

## Failure patterns (19 cases)
- **Paraphrase vocabulary:** "call off" / "drop my booking" pull no-show sections instead of cancellation (acme-09, bolt-11).
- **Multi-hop:** the second required section is missed (acme-18, acme-19, acme-20, bolt-21, crest-18, crest-19).
- **Typos:** "wat time do i hav to bok" ranks the escort sections first (acme-30, bolt-30).
- **Injection chunk ranks first** for "Are company cab trips free?" (crest-26).
- **Generic question, specific chunk:** "Who is eligible?" ranks airport sections above §1.1 (crest-01).

## Decision
This is the baseline every later retrieval experiment is compared with. Next: compare embedding models with chunking held fixed.
