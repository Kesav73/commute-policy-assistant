"""Retrieval: question -> embedding -> top-k chunks from the CALLER'S company only.

    company_id (from the verified token, never from the request)
        -> open ONLY collection policy__<company_id>   (ids + vectors)
        -> embed the question with the same model used at ingest
        -> rank the chunks by cosine distance (vector) and, in hybrid mode, also by
           BM25 keywords, fused with weighted RRF; keep the best k
        -> look each id up in THIS company's chunk store (text + metadata);
           an id it doesn't hold is dropped and logged (defence in depth)
        -> evidence gate: is the best match close enough to bother the LLM?

Try it:  python -m app.retriever acme "Can I book a cab on Saturday?"
"""
import logging
import sys
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.bm25 import BM25Index
from app.embeddings import EMBEDDING_ID, embed_query
from app.store import PolicyStore, StoreError, get_company_collection, open_store

log = logging.getLogger("policy_assistant")

TOP_K = 3  # PS.md C3: never more than 3 chunks reach the LLM / the answer


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    company_id: str
    doc_id: str
    title: str
    section: str
    section_title: str
    text: str
    distance: float  # cosine distance: 0 = same meaning, larger = less related


@dataclass
class RetrievalResult:
    company_id: str
    question: str
    chunks: list[RetrievedChunk]          # best first, at most TOP_K
    has_evidence: bool                     # best distance <= max_distance
    max_distance: float
    timings_ms: dict = field(default_factory=dict)
    best_distance: float | None = None     # closest vector match (what the gate uses)


def retrieve(
    store: PolicyStore,
    company_id: str,
    question: str,
    k: int = TOP_K,
    max_distance: float | None = None,
    mode: str | None = None,          # "vector" | "hybrid"; default from settings
) -> RetrievalResult:
    k = max(1, min(k, TOP_K))
    if max_distance is None:
        max_distance = get_settings().retrieval_max_distance

    collection = get_company_collection(store, company_id)
    if collection is None or collection.count() == 0:
        # No policy for this company: nothing to search, nothing to leak.
        return RetrievalResult(company_id, question, [], False, max_distance)
    if (collection.metadata or {}).get("embedding") != EMBEDDING_ID:
        # Vectors made by a different model/prefix would be compared with this
        # question's vector: results would be quietly wrong. Fail loudly instead
        # (the pipeline turns this into "try again"; a re-ingest fixes it).
        raise StoreError(f"{collection.name} was built with {(collection.metadata or {}).get('embedding')!r}, "
                         f"expected {EMBEDDING_ID!r}; re-ingest needed")

    mode = mode or get_settings().retrieval_mode
    n = collection.count()
    # Hybrid ranks EVERY chunk of the company (<= ~30) by both methods, then fuses.
    pool = n if mode == "hybrid" else min(k, n)

    t0 = time.perf_counter()
    query_vector = embed_query(question)   # same model + matching prefix as ingest
    t1 = time.perf_counter()
    raw = collection.query(
        query_embeddings=query_vector,
        n_results=pool,
        include=["distances"],   # ids come back always; content comes from the chunk store
    )
    t2 = time.perf_counter()
    # Defence in depth (HLD §7.1 layer 3): EVERY candidate id must be one of this
    # company's chunks, checked before any ranking or fusion, so a foreign id is
    # caught (and logged) even when it would not make the final top k.
    # With one collection per company this should never fire.
    own = store.chunks.all(company_id)
    distance, vector_order = {}, []
    for cid, dist in zip(raw["ids"][0], raw["distances"][0]):   # best first
        c = own.get(cid)
        if c is None or c.company_id != company_id:
            log.error("ISOLATION ALERT: id %s in %s's collection is not one of %s's chunks",
                      cid, company_id, company_id)
            continue
        distance[cid] = dist
        vector_order.append(cid)
    timings = {"embed": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 1)}

    if mode == "hybrid":
        keyword = _bm25(store, company_id).scores(question)
        keyword_order = sorted(keyword, key=lambda i: -keyword[i])
        order = rrf([vector_order, keyword_order], weights=[VECTOR_WEIGHT, KEYWORD_WEIGHT])
        timings["keyword"] = round((time.perf_counter() - t2) * 1000, 1)
    else:
        order = vector_order

    chunks = []
    for cid in order:
        c = own[cid]          # every id in `order` was validated above (BM25 is built from `own`)
        chunks.append(RetrievedChunk(
            id=c.id, company_id=c.company_id, doc_id=c.doc_id, title=c.title,
            section=c.section, section_title=c.section_title,
            text=c.text, distance=float(distance.get(cid, 1.0)),
        ))
        if len(chunks) == k:
            break

    # The gate always uses the closest VECTOR match, whatever the final order:
    # "is anything in this policy about this question at all?"
    best = min(distance.values(), default=None)
    has_evidence = best is not None and best <= max_distance
    return RetrievalResult(
        company_id=company_id, question=question, chunks=chunks,
        has_evidence=has_evidence, max_distance=max_distance,
        timings_ms=timings, best_distance=best,
    )


# --------------------------------------------------------------- hybrid --

RRF_K = 60  # standard constant from the RRF paper (Cormack et al., 2009)
# Vectors get twice the vote of keywords: alone, vectors are the stronger ranker
# (recall@3 0.89 vs 0.84 for stemmed BM25), so keywords should mainly add what
# vectors miss rather than overrule them. Measured in docs/journey/09.
VECTOR_WEIGHT = 2.0
KEYWORD_WEIGHT = 1.0
STEM = True  # "booking"/"booked"/"book" match each other (BM25 alone: 51 -> 59 of 75)


def rrf(rankings: list[list[str]], weights: list[float] | None = None, k: int = RRF_K) -> list[str]:
    """Weighted Reciprocal Rank Fusion: score(id) = sum of weight / (k + rank).

    Uses only ranks, not raw scores, so cosine distances and BM25 scores (which
    live on unrelated scales) never have to be compared or normalised.
    """
    weights = weights or [1.0] * len(rankings)
    score: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + w / (k + rank)
    return sorted(score, key=lambda i: -score[i])


_BM25_CACHE: dict[str, tuple[int, BM25Index]] = {}


def _bm25(store: PolicyStore, company_id: str) -> BM25Index:
    """Per-company BM25 index built from the chunk store; rebuilt when its chunks change."""
    chunks = store.chunks.all(company_id)
    cached = _BM25_CACHE.get(company_id)
    if cached is None or cached[0] != id(chunks):
        cached = (id(chunks), BM25Index(list(chunks), [c.text for c in chunks.values()], stem=STEM))
        _BM25_CACHE[company_id] = cached
    return cached[1]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: python -m app.retriever <company_id> "<question>"')
    company_id, question = sys.argv[1], sys.argv[2]
    embed_query("warm up")  # load the model first, so the timing below is per-question
    r = retrieve(open_store(get_settings().data_dir), company_id, question)
    best = f"{r.best_distance:.3f}" if r.chunks else "-"
    print(f"\n{company_id}: {question!r}")
    print(f"evidence: {'YES' if r.has_evidence else 'NO -> not covered'} "
          f"(best distance {best}, gate <= {r.max_distance})   timings {r.timings_ms}\n")
    for i, c in enumerate(r.chunks, 1):
        body = c.text.split("\n", 1)[1]
        print(f"  {i}. §{c.section:<4} {c.distance:.3f}  {c.section_title}")
        print(f"     {body[:110]}{'...' if len(body) > 110 else ''}")


if __name__ == "__main__":
    main()
