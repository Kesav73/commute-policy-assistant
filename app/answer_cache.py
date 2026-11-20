"""Answer cache: employees ask the desk the same things every day (the brief), so a
repeated question is answered instantly with 0 LLM calls.

Key = (company, ACTIVE policy version, normalised question).
  * company in the key -> isolation holds: Acme's answer can't reach a Bolt employee.
  * version in the key -> correctness under updates: once v2 is live, every lookup
    uses v2, so no v1 answer can ever be returned again. No manual invalidation.
  * exact match after normalising case, spaces and punctuation - deliberately NOT
    "similar questions": "drop at 9 PM" and "drop after 9 PM" must not share an answer.

Only stable outcomes are cached: answered, not covered, guardrail refusal. A
"try again" (temporary failure) is never cached. Bounded size (least recently used
entries leave first) and a time-to-live as a safety net. In memory: a restart
starts empty, which is fine.
"""
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^a-z0-9 ]+")
CACHEABLE = {"answered", "not_covered"}


def normalise(question: str) -> str:
    """'  Can I book a cab on SATURDAY?? ' -> 'can i book a cab on saturday'"""
    q = _NON_WORD.sub(" ", question.lower())
    return " ".join(q.split())


@dataclass(frozen=True)
class CachedAnswer:
    status: str
    answer: str
    citations: tuple          # tuple of Citation (immutable)
    policy_version: int


class AnswerCache:
    def __init__(self, max_entries: int = 1000, ttl_s: float = 6 * 3600) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._data: OrderedDict[tuple, tuple[float, CachedAnswer]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(company_id: str, version: int, question: str) -> tuple:
        return (company_id, version, normalise(question))

    def get(self, company_id: str, version: int | None, question: str) -> CachedAnswer | None:
        if self.max_entries <= 0 or version is None:
            return None
        k = self.key(company_id, version, question)
        with self._lock:
            item = self._data.get(k)
            if item is None or time.monotonic() - item[0] > self.ttl_s:
                if item is not None:
                    del self._data[k]          # expired
                self.misses += 1
                return None
            self._data.move_to_end(k)          # recently used
            self.hits += 1
            return item[1]

    def put(self, company_id: str, question: str, answer: CachedAnswer) -> None:
        if self.max_entries <= 0 or answer.status not in CACHEABLE:
            return
        k = self.key(company_id, answer.policy_version, question)
        with self._lock:
            self._data[k] = (time.monotonic(), answer)
            self._data.move_to_end(k)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # least recently used leaves first

    def drop_company(self, company_id: str) -> int:
        """Free the memory of a company's entries (e.g. after a new version is
        published). Not needed for correctness - the version key already does that."""
        with self._lock:
            old = [k for k in self._data if k[0] == company_id]
            for k in old:
                del self._data[k]
            return len(old)

    def __len__(self) -> int:
        return len(self._data)
