"""Retrieval: question -> embedding -> top-k chunks from the CALLER'S company only.

    company_id (from the verified token, never from the request)
        -> open ONLY collection policy__<company_id>   (ids + vectors)
        -> embed the question with the same model used at ingest
        -> nearest k chunk ids by cosine distance
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

    @property
    def best_distance(self) -> float | None:
        return self.chunks[0].distance if self.chunks else None


def retrieve(
    store: PolicyStore,
    company_id: str,
    question: str,
    k: int = TOP_K,
    max_distance: float | None = None,
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

    t0 = time.perf_counter()
    query_vector = embed_query(question)   # same model + matching prefix as ingest
    t1 = time.perf_counter()
    raw = collection.query(
        query_embeddings=query_vector,
        n_results=min(k, collection.count()),
        include=["distances"],   # ids come back always; content comes from the chunk store
    )
    t2 = time.perf_counter()

    chunks = []
    for cid, dist in zip(raw["ids"][0], raw["distances"][0]):
        # Defence in depth (HLD §7.1 layer 3). With one collection per company this
        # should never fire; if it does, the id is dropped and loudly logged.
        c = store.chunks.get(company_id, cid)
        if c is None or c.company_id != company_id:
            log.error("ISOLATION ALERT: id %s in %s's collection is not one of %s's chunks",
                      cid, company_id, company_id)
            continue
        chunks.append(RetrievedChunk(
            id=c.id, company_id=c.company_id, doc_id=c.doc_id, title=c.title,
            section=c.section, section_title=c.section_title,
            text=c.text, distance=float(dist),
        ))

    has_evidence = bool(chunks) and chunks[0].distance <= max_distance
    return RetrievalResult(
        company_id=company_id, question=question, chunks=chunks,
        has_evidence=has_evidence, max_distance=max_distance,
        timings_ms={"embed": round((t1 - t0) * 1000, 1), "search": round((t2 - t1) * 1000, 1)},
    )


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
