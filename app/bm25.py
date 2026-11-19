"""BM25 keyword search over one company's chunks (Okapi BM25, in memory).

Why keywords next to vectors: an embedding captures meaning but can blur exact
terms - "no-show", "HR Head", "INR 900", "Transport Manager". BM25 scores a chunk
by how many of the question's words it contains, weighting rare words more
(IDF) and not over-rewarding repeats or long chunks (k1, b).

    score(q, d) = sum over words w in q of
        idf(w) * tf(w,d) * (k1 + 1) / (tf(w,d) + k1 * (1 - b + b * len(d) / avg_len))
    idf(w) = ln(1 + (N - df(w) + 0.5) / (df(w) + 0.5))

The index is built from the chunk store, per company, so it can only ever
contain that company's text (same isolation as the vector collection).
"""
import math
import re
from collections import Counter
from functools import lru_cache

K1 = 1.5
B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# Very common words carry no signal for matching policy rules.
STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i if in is it its
my of on or so than that the their them then there these they this to was we what
when where which who will with would you your me am our any all
""".split())


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Lowercase words minus stopwords; with stem=True, "booking"/"booked" -> "book"."""
    words = [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [_stemmer().stem_word(w) for w in words] if stem else words


@lru_cache
def _stemmer():
    import py_rust_stemmers   # Snowball (Porter2) English stemmer
    return py_rust_stemmers.SnowballStemmer("english")


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], stem: bool = False) -> None:
        self.ids = ids
        self.stem = stem
        self.docs = [Counter(tokenize(t, stem)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        """{chunk id: score} for every chunk (0.0 when no word matches)."""
        q = tokenize(query, self.stem)
        out = {}
        for cid, d, length in zip(self.ids, self.docs, self.lengths):
            s = 0.0
            for w in q:
                tf = d.get(w, 0)
                if tf:
                    s += self.idf[w] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self.avg_len))
            out[cid] = s
        return out
