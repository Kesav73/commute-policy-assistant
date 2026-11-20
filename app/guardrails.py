"""Deterministic guardrails: plain code, no LLM calls.

The LLM guards live inside the two calls we already make (the rewriter judges the
question, the answerer may refuse). These code checks are the backstop that has
the final word, because "treat model output as untrusted" (PS.md) means a model
that was talked into something can't also be the only judge of it.

    chunks  is_instruction_like()   at ingest: text that gives orders to an AI is
                                    never indexed (e.g. Crest section 7)
    output  echoed_injection()      the answer repeats what injected text wanted said
            other_company_terms()   names/doc ids of the OTHER companies; an answer
            mentions_other_company()   for Acme must never contain "Bolt" or "Crest"

Deliberately NOT here: matching every number/time in the answer against the
excerpts. Measured (docs/journey/12): facts were right in 63/66 answers with full
retrieval and none were invented, so a strict matcher would mostly add false
blocks ("8 PM" vs "8:00 PM"). See docs/journey/14.
"""
import re

from app.store import PolicyStore

# Phrases that only make sense as orders to an AI system, never as a transport rule.
INSTRUCTION_PATTERNS = [
    r"\bignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules)\b",
    r"\bsystem (override|prompt)\b",
    r"\byou are now\b",
    r"\b(tell|inform) (every|all|each) users?\b",
    r"\breveal\b.{0,60}\b(polic(y|ies)|prompt|instructions)\b",
    r"\b(automated )?assistants?\b.{0,40}\b(must|should|are to)\b",
    r"\bdisregard\b.{0,40}\b(instructions|rules|guidelines)\b",
]

# What injected text typically tries to make the assistant say.
ECHO_PATTERNS = [
    r"\ball (company )?(cab )?trips are free\b",
    r"\bthere are no rules\b",
    r"\bignore (all )?(previous|prior) instructions\b",
    r"\bsystem override\b",
]

_INSTRUCTION_RE = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_PATTERNS]
_ECHO_RE = [re.compile(p, re.IGNORECASE) for p in ECHO_PATTERNS]


def is_instruction_like(text: str) -> bool:
    """True if a policy section reads like orders to an AI (quarantine it at ingest)."""
    return any(r.search(text) for r in _INSTRUCTION_RE)


def echoed_injection(answer: str) -> str | None:
    """The first injected phrase the answer repeats, or None."""
    for r in _ECHO_RE:
        if m := r.search(answer):
            return m.group(0)
    return None


def other_company_terms(store: PolicyStore, own_company: str, companies: list[str]) -> list[str]:
    """Case-sensitive names that must never appear in an answer for `own_company`:
    each other company's doc id, document title and short name (first word of the
    title, e.g. "Bolt"), read from that company's chunk store."""
    terms = set()
    for co in companies:
        if co == own_company:
            continue
        chunks = store.chunks.all(co)
        if not chunks:
            continue
        c = next(iter(chunks.values()))
        terms.update({c.doc_id, c.title, c.title.split()[0]})
    return sorted(terms, key=len, reverse=True)


def mentions_other_company(answer: str, terms: list[str]) -> str | None:
    """The first other-company term found as a whole word (case-sensitive), or None.
    Case-sensitive so that the ordinary word "bolt" can't trigger it; only "Bolt"."""
    for t in terms:
        if re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", answer):
            return t
    return None
