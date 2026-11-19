"""Embedding model benchmark: same chunks, same questions, same scoring - only the embedder changes.

    python -m eval.bench_embeddings

Held fixed: sub-section chunks (exactly what the app ingests), Chroma cosine HNSW,
one collection per company, top 3, the 90-question gold set, the scorer in
eval/run_retrieval.py.

Varied: the embedding model, and whether its query/passage prefixes are used.

Reports the three golden metrics (recall@1, recall@3, avg latency = embed the
question + Chroma search, after a warm-up), plus, for the record:
    load s        time to load the model
    index s       time to embed all 71 chunks
    gap           mean best distance of not-covered questions minus that of
                  answerable ones; larger = easier "not covered" gate
"""
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.ingest import load_policies
from eval.run_retrieval import (CATEGORIES, RESULTS_DIR, load_questions, score_case,
                                summarise, validate_gold)

FASTEMBED_CACHE = Path.home() / ".cache" / "fastembed"
BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class Variant:
    label: str
    model: str            # fastembed model name
    query_prefix: str = ""
    passage_prefix: str = ""


VARIANTS = [
    Variant("all-MiniLM-L6-v2 (baseline)", "sentence-transformers/all-MiniLM-L6-v2"),
    Variant("arctic-embed-s", "snowflake/snowflake-arctic-embed-s"),
    Variant("arctic-embed-s + prefix", "snowflake/snowflake-arctic-embed-s", BGE_QUERY),
    Variant("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5"),
    Variant("bge-base-en-v1.5 + prefix", "BAAI/bge-base-en-v1.5", BGE_QUERY),
    # Same family, 3x larger (335M, 1.2 GB): does size alone help?
    Variant("bge-large-en-v1.5", "BAAI/bge-large-en-v1.5"),
    Variant("bge-large-en-v1.5 + prefix", "BAAI/bge-large-en-v1.5", BGE_QUERY),
    Variant("nomic-embed-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
    Variant("nomic-embed-v1.5 + prefix", "nomic-ai/nomic-embed-text-v1.5",
            "search_query: ", "search_document: "),
    # Same nomic model, int8-quantised (0.13 GB instead of 0.52 GB): the speed/size check.
    Variant("nomic-embed-v1.5 int8", "nomic-ai/nomic-embed-text-v1.5-Q"),
    Variant("nomic-embed-v1.5 int8 + prefix", "nomic-ai/nomic-embed-text-v1.5-Q",
            "search_query: ", "search_document: "),
]


class Embedder:
    """A fastembed model that returns unit vectors."""

    def __init__(self, model: str) -> None:
        from fastembed import TextEmbedding
        m = TextEmbedding(model_name=model, cache_dir=str(FASTEMBED_CACHE))
        self._embed = lambda texts: np.asarray(list(m.embed(texts)), dtype=np.float32)

    def __call__(self, texts: list[str]) -> np.ndarray:
        v = self._embed(texts)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def run_variant(v: Variant, policies, cases, embedders: dict) -> dict:
    if v.model not in embedders:   # load each model once; later variants reuse it
        t0 = time.perf_counter()
        embedders[v.model] = (Embedder(v.model), time.perf_counter() - t0)
    emb, load_s = embedders[v.model]

    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp, settings=ChromaSettings(anonymized_telemetry=False))
        cols, section_of, t_index = {}, {}, 0.0
        for p in policies:
            t1 = time.perf_counter()
            vecs = emb([v.passage_prefix + c.text for c in p.chunks])
            t_index += time.perf_counter() - t1
            col = client.create_collection(f"bench__{p.company_id}", embedding_function=None,
                                           configuration={"hnsw": {"space": "cosine"}})
            col.add(ids=[c.id for c in p.chunks], embeddings=vecs)
            cols[p.company_id] = col
            section_of.update({c.id: (c.company_id, c.section) for c in p.chunks})

        def search(company, question):
            q = emb([v.query_prefix + question])
            return cols[company].query(query_embeddings=q, n_results=3, include=["distances"])

        search(cases[0]["company"], "warm up")
        rows, best = [], []
        for c in cases:
            t1 = time.perf_counter()
            raw = search(c["company"], c["question"])
            latency_ms = (time.perf_counter() - t1) * 1000
            ids = raw["ids"][0]
            ranked = [section_of[i][1] for i in ids]
            leaks = [i for i in ids if section_of[i][0] != c["company"]]
            best.append((bool(c["required"]), raw["distances"][0][0]))
            rows.append({"id": c["id"], "input": {"company": c["company"], "question": c["question"]},
                         "expected": {"required": c["required"], "category": c["category"]},
                         **score_case(c, ranked, leaks, latency_ms)})

    summary = summarise(rows)
    ans = [d for a, d in best if a]
    nc = [d for a, d in best if not a]
    return {"label": v.label, "model": v.model, "query_prefix": v.query_prefix,
            "passage_prefix": v.passage_prefix, "load_s": round(load_s, 2),
            "index_s": round(t_index, 2), "gap": round(float(np.mean(nc) - np.mean(ans)), 3),
            "best_distance_answerable": round(float(np.mean(ans)), 3),
            "best_distance_not_covered": round(float(np.mean(nc)), 3),
            **summary, "failed": [r["id"] for r in rows if not r["pass"]]}


def main() -> None:
    s = get_settings()
    cases = load_questions()
    validate_gold(cases, s.policies_dir)
    policies = load_policies(s.policies_dir)
    only = sys.argv[1:]  # optional: substrings of labels to run
    variants = [v for v in VARIANTS if not only or any(o in v.label for o in only)]

    embedders, results = {}, []
    for v in variants:
        print(f"running {v.label} ...", flush=True)
        results.append(run_variant(v, policies, cases, embedders))

    print(f"\n{'variant':30}{'recall@1':>9}{'recall@3':>9}{'avg ms':>8}{'pass':>8}"
          f"{'load s':>8}{'index s':>8}{'gap':>7}")
    for r in results:
        o = r["overall"]
        print(f"{r['label']:30}{o['recall_at_1']:>9.2f}{o['recall_at_3']:>9.2f}"
              f"{o['avg_latency_ms']:>8.1f}{o['pass']:>5}/{o['answerable']:<2}"
              f"{r['load_s']:>8.1f}{r['index_s']:>8.2f}{r['gap']:>7.3f}")

    print(f"\nrecall@3 by category\n{'variant':30}" + "".join(f"{c[:9]:>10}" for c in CATEGORIES[:-1]))
    for r in results:
        print(f"{r['label']:30}" + "".join(
            f"{r['by_category'][c]['recall_at_3']:>10.2f}" for c in CATEGORIES[:-1]))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "embeddings_summary.json"
    out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "variants": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
