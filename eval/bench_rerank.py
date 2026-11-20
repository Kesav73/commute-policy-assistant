"""Re-ranking benchmark: first stage (vector or hybrid) -> top-N candidates -> cross-encoder -> top 3.

    python -m eval.bench_rerank                 # all variants
    python -m eval.bench_rerank bge-reranker    # only variants whose label contains this

Held fixed: chunks, nomic embeddings, the gold set and the scorer (eval/run_retrieval.py).
Varied: the re-ranker model, the first-stage mode (vector | hybrid), the pool size N.

A cross-encoder reads the question and ONE chunk together and outputs a relevance
score, instead of comparing two separately made vectors. Slower (one model pass per
candidate), but it sees how the words of the question relate to the words of the chunk.

Reports the three golden metrics; latency = the full retrieval (embed + search
[+ BM25] + re-rank), after a warm-up. Also, for the "not covered" gate:
    sep   how well the top re-rank score separates answerable from not-covered
          questions: share of (answerable, not-covered) pairs where the answerable
          one scores higher (AUC; 0.5 = no signal, 1.0 = perfect). The same is
          reported for the vector distance as a reference.
"""
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.retriever import TOP_K, rank_query, vector_search
from app.store import PolicyStore
from app.versioning import ingest_all
from eval.run_retrieval import CATEGORIES, RESULTS_DIR, load_questions, score_case, summarise, validate_gold

CACHE = Path.home() / ".cache" / "fastembed"


@dataclass(frozen=True)
class Variant:
    label: str
    reranker: str | None   # fastembed cross-encoder name; None = no re-ranking
    mode: str              # first stage: "vector" | "hybrid"
    pool: int | None       # candidates passed to the re-ranker; None = all chunks


RERANKERS = {
    "minilm-L6": "Xenova/ms-marco-MiniLM-L-6-v2",
    "jina-turbo": "jinaai/jina-reranker-v1-turbo-en",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3-int8": "BAAI/bge-reranker-v2-m3-int8",
}

VARIANTS = [Variant("no rerank, vector", None, "vector", TOP_K),
            Variant("no rerank, hybrid", None, "hybrid", TOP_K)]
for short, name in RERANKERS.items():
    for mode in ("vector", "hybrid"):
        for pool in (10, None):
            VARIANTS.append(Variant(f"{short}, {mode}, N={pool or 'all'}", name, mode, pool))


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random answerable > score of a random not-covered)."""
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    return sum(pairs) / len(pairs)


def candidates(store, company: str, question: str, mode: str, pool: int | None):
    """First stage: the company's chunks ranked by vector search alone or by hybrid
    search, cut to `pool`. Returns (chunks best first, closest vector distance)."""
    if mode == "hybrid":
        ranked, best = rank_query(store, company, None, question)
    else:
        ids, distances = vector_search(store, company, None, question)
        hybrid_by_id = {c.id: c for c in rank_query(store, company, None, question)[0]}
        ranked = [hybrid_by_id[i] for i in ids]          # same chunk objects, vector order
        best = min(distances.values()) if distances else None
    return ranked[:pool] if pool else ranked, best


def run_variant(v: Variant, store, cases, models: dict) -> dict:
    model = None
    if v.reranker:
        if v.reranker not in models:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            models[v.reranker] = TextCrossEncoder(model_name=v.reranker, cache_dir=str(CACHE))
        model = models[v.reranker]

    def run(c):
        chunks, best_distance = candidates(store, c["company"], c["question"], v.mode, v.pool)
        top_score = None
        if model is not None:
            scores = list(model.rerank(c["question"], [ch.text for ch in chunks]))
            order = sorted(range(len(chunks)), key=lambda i: -scores[i])
            chunks = [chunks[i] for i in order]
            top_score = float(scores[order[0]])
        return chunks[:TOP_K], best_distance, top_score

    run(cases[0])   # warm-up: load models, build BM25 caches
    rows, sig = [], []
    for c in cases:
        t0 = time.perf_counter()
        top, best_distance, top_score = run(c)
        ms = (time.perf_counter() - t0) * 1000
        ranked = [ch.section for ch in top]
        leaks = [ch.id for ch in top if ch.company_id != c["company"]]
        rows.append({"id": c["id"], "input": {"company": c["company"], "question": c["question"]},
                     "expected": {"required": c["required"], "category": c["category"]},
                     **score_case(c, ranked, leaks, ms)})
        sig.append((bool(c["required"]), best_distance, top_score))

    ans = [s for s in sig if s[0]]
    nc = [s for s in sig if not s[0]]
    out = {"label": v.label, "reranker": v.reranker, "mode": v.mode, "pool": v.pool,
           "sep_distance": round(auc([-a[1] for a in ans], [-n[1] for n in nc]), 3),
           **summarise(rows), "failed": [r["id"] for r in rows if not r["pass"]]}
    if model is not None:
        out["sep_rerank"] = round(auc([a[2] for a in ans], [n[2] for n in nc]), 3)
    return out


def main() -> None:
    s = get_settings()
    cases = load_questions()
    validate_gold(cases, s.policies_dir)
    only = sys.argv[1:]
    variants = [v for v in VARIANTS if not only or any(o in v.label for o in only)]

    results, models = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        store = PolicyStore(Path(tmp))
        ingest_all(store, s.policies_dir)
        for v in variants:
            print(f"running {v.label} ...", flush=True)
            results.append(run_variant(v, store, cases, models))

    print(f"\n{'variant':40}{'recall@1':>9}{'recall@3':>9}{'avg ms':>8}{'pass':>8}{'sep':>7}")
    for r in results:
        o = r["overall"]
        sep = r.get("sep_rerank", r["sep_distance"])
        print(f"{r['label']:40}{o['recall_at_1']:>9.2f}{o['recall_at_3']:>9.2f}"
              f"{o['avg_latency_ms']:>8.1f}{o['pass']:>5}/{o['answerable']:<2}{sep:>7.3f}")
    print(f"\nrecall@3 by category\n{'variant':40}" + "".join(f"{c[:9]:>10}" for c in CATEGORIES[:-1]))
    for r in results:
        print(f"{r['label']:40}" + "".join(f"{r['by_category'][c]['recall_at_3']:>10.2f}" for c in CATEGORIES[:-1]))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "rerank_summary.json"
    out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "variants": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
