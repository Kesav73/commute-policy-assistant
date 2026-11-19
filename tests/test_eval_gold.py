"""The gold evaluation set itself must stay valid (sections exist, labels consistent)."""
from collections import Counter

from app.config import get_settings
from eval.run_retrieval import load_questions, validate_gold


def test_gold_set_is_valid():
    validate_gold(load_questions(), get_settings().policies_dir)  # raises SystemExit if not


def test_gold_set_size_and_balance():
    cases = load_questions()
    assert Counter(c["company"] for c in cases) == {"acme": 30, "bolt": 30, "crest": 30}
    cats = Counter(c["category"] for c in cases)
    for needed in ("paraphrase", "multi_hop", "not_covered", "cross_company", "injection"):
        assert cats[needed] >= 3, needed
