"""Retrieval: search queries -> the 3 best chunks from the CALLER'S company only.

For each search query (the rewritten queries, or the original question):
  1. vector search:  embed "search_query: <query>", rank every chunk of the company's
                     collection by cosine distance (0 = same meaning)
  2. keyword search: rank the same chunks with BM25 (stemmed words, rare words count more)
  3. fuse the two rankings with weighted Reciprocal Rank Fusion (vectors count double)
Then merge the queries' rankings ROUND-ROBIN (each query's best chunk first) and keep 3.

Isolation: only that company's collection is opened, and every id it returns must exist
in that company's chunk file; anything else is dropped and logged as an alert.
"""
import logging
import time
from dataclasses import dataclass

from app.bm25 import BM25Index
from app.config import get_settings
from app.embeddings import EMBEDDING_ID, embed_query
from app.store import PolicyStore, StoreError

log = logging.getLogger("policy_assistant")

TOP_K = 3              # chunks given to the LLM; also the most it can cite
RRF_K = 60             # standard constant from the RRF paper (Cormack et al., 2009)
VECTOR_WEIGHT = 2.0    # vectors are the stronger search alone (recall@3 0.89 vs 0.84),
KEYWORD_WEIGHT = 1.0   # so keywords add what vectors miss rather than overrule them


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    company_id: str
    doc_id: str
    title: str
    section: str
    section_title: str
    text: str
    distance: float   # cosine distance of the vector match: 0 = same meaning


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]      # best first, at most TOP_K
    best_distance: float | None       # closest vector match of any query (used by the gate)
    has_evidence: bool                # best_distance <= the "not covered" threshold
    timings_ms: dict


def retrieve(store: PolicyStore, company_id: str, queries: list[str],
             version: int | None = None, max_distance: float | None = None) -> RetrievalResult:
    """The 3 best chunks for these queries, from one company's policy version
    (default: the live one). With several queries, each one's best chunk comes first."""
    if max_distance is None:
        max_distance = get_settings().retrieval_max_distance
    if version is None:
        version = store.active_version(company_id)
    started = time.perf_counter()

    rankings, best = [], None
    for query in queries:
        ranked, query_best = rank_query(store, company_id, version, query)
        rankings.append(ranked)
        if query_best is not None and (best is None or query_best < best):
            best = query_best

    top = round_robin(rankings)[:TOP_K]
    return RetrievalResult(
        chunks=top,
        best_distance=best,
        has_evidence=best is not None and best <= max_distance,
        timings_ms={"retrieve": round((time.perf_counter() - started) * 1000, 1), "queries": len(queries)},
    )


def rank_query(store: PolicyStore, company_id: str, version: int | None, query: str):
    """Hybrid ranking of ALL the company's chunks for one query.
    Returns (chunks best first, closest vector distance)."""
    vector_ids, distances = vector_search(store, company_id, version, query)
    if not vector_ids:
        return [], None
    keyword_ids = keyword_ranking(store, company_id, version, query)
    order = rrf([vector_ids, keyword_ids], weights=[VECTOR_WEIGHT, KEYWORD_WEIGHT])

    chunks = store.chunks(company_id, version)
    ranked = []
    for chunk_id in order:
        c = chunks[chunk_id]
        ranked.append(RetrievedChunk(id=c.id, company_id=c.company_id, doc_id=c.doc_id, title=c.title,
                                     section=c.section, section_title=c.section_title, text=c.text,
                                     distance=float(distances.get(chunk_id, 1.0))))
    return ranked, min(distances.values())


def vector_search(store: PolicyStore, company_id: str, version: int | None, query: str):
    """Every chunk of the company, ranked by cosine distance to the query.
    Returns (ids best first, {id: distance}). Empty if the company has no policy."""
    collection = store.collection(company_id, version)
    if collection is None or collection.count() == 0:
        return [], {}
    if (collection.metadata or {}).get("embedding") != EMBEDDING_ID:
        # Vectors from another model would be compared with this question's vector and
        # give quietly wrong results. Fail loudly instead ("try again"; a republish fixes it).
        raise StoreError(f"{collection.name} was built with a different embedding model")

    raw = collection.query(query_embeddings=embed_query(query), n_results=collection.count(),
                           include=["distances"])
    own_chunks = store.chunks(company_id, version)
    ids, distances = [], {}
    for chunk_id, distance in zip(raw["ids"][0], raw["distances"][0]):
        chunk = own_chunks.get(chunk_id)
        if chunk is None or chunk.company_id != company_id:
            # Should never happen with one collection per company: drop it, and say so loudly.
            log.error("ISOLATION ALERT: id %s in %s's collection is not one of its chunks", chunk_id, company_id)
            continue
        ids.append(chunk_id)
        distances[chunk_id] = distance
    return ids, distances


_bm25_indexes = {}   # (company, version) -> BM25Index, built on first use


def keyword_ranking(store: PolicyStore, company_id: str, version: int | None, query: str) -> list[str]:
    """Every chunk id of the company, ranked by BM25 keyword score (best first)."""
    chunks = store.chunks(company_id, version)
    key = (company_id, version)
    if key not in _bm25_indexes or _bm25_indexes[key].ids != list(chunks):
        _bm25_indexes[key] = BM25Index(list(chunks), [c.text for c in chunks.values()], stem=True)
    scores = _bm25_indexes[key].scores(query)
    return sorted(scores, key=scores.get, reverse=True)


def rrf(rankings: list[list[str]], weights: list[float] | None = None) -> list[str]:
    """Weighted Reciprocal Rank Fusion: score(id) = sum of weight / (60 + rank).
    Uses only ranks, so a cosine distance and a BM25 score never have to be compared."""
    if weights is None:
        weights = [1.0] * len(rankings)
    score = {}
    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + weight / (RRF_K + rank)
    return sorted(score, key=score.get, reverse=True)


def round_robin(rankings: list[list]) -> list:
    """Take the best not-yet-chosen item of ranking 1, then of ranking 2, ... and repeat.
    For a split-up question each sub-query looks for a different rule, so each one's
    top hit gets a place (RRF would favour items the sub-queries happen to share)."""
    merged, seen = [], set()
    longest = max((len(r) for r in rankings), default=0)
    for depth in range(longest):
        for ranking in rankings:
            if depth < len(ranking):
                item = ranking[depth]
                key = getattr(item, "id", item)   # works for chunks and for plain ids
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
    return merged


def warm_up(store: PolicyStore) -> dict[str, float]:
    """One retrieval per company at startup, before requests are accepted: loads the
    embedding model (~1 s) and builds the indexes, so the first question isn't slow."""
    took = {}
    for company_id in store.active_companies():
        started = time.perf_counter()
        retrieve(store, company_id, ["warm up"])
        took[company_id] = round((time.perf_counter() - started) * 1000, 1)
    return took
