import math

import pytest

from plexus.eval.bootstrap import bootstrap_ci, paired_bootstrap
from plexus.eval.metrics import (
    Span,
    is_hit,
    locate_quote,
    mrr,
    ndcg_at_k,
    recall_at_k,
    retrieval_metrics,
)

DOC = (
    "Item 1A.\nWe rely on  Taiwan Semiconductor\nManufacturing Company (“TSMC”) — a single foundry."
)


def test_locate_quote_ignores_case_whitespace_and_typography():
    span = locate_quote(
        DOC, 'we rely on Taiwan Semiconductor Manufacturing Company ("TSMC") - a single'
    )
    assert span is not None
    s, e = span
    assert DOC[s:e].startswith("We rely on") and DOC[s:e].endswith("a single")
    assert locate_quote(DOC, "Intel foundry") is None
    assert locate_quote(DOC, "   ") is None


def test_locate_quote_matches_across_table_delimiters():
    doc = "Revenue | $ | 130,497 | 60,922"
    assert locate_quote(doc, "Revenue $ | 130,497") is None  # different cell structure
    assert locate_quote(doc, "revenue |  $ | 130,497") == (0, 21)


G = Span("D", 100, 200)


@pytest.mark.parametrize(
    ("r", "hit"),
    [
        (Span("D", 0, 1000), True),  # big chunk covering all of the evidence
        (Span("D", 150, 400), True),  # covers exactly 50% of the evidence
        (Span("D", 160, 400), False),  # 40% of evidence, 17% of chunk
        (Span("D", 120, 140), True),  # small chunk fully inside the evidence
        (Span("E", 100, 200), False),  # other document
        (Span("D", 200, 300), False),  # touching, no overlap
    ],
)
def test_hit_rule(r, hit):
    assert is_hit(r, G) is hit


def test_metrics_on_hand_computed_example():
    gold = [Span("D", 0, 100), Span("E", 0, 100)]
    retrieved = [
        Span("X", 0, 100),  # rank 1 miss
        Span("D", 0, 100),  # rank 2 hits gold 0
        Span("D", 10, 90),  # rank 3 duplicate of gold 0: earns nothing
        Span("E", 0, 100),  # rank 4 hits gold 1
    ]
    assert recall_at_k(retrieved, gold, 1) == 0
    assert recall_at_k(retrieved, gold, 3) == 0.5
    assert recall_at_k(retrieved, gold, 4) == 1
    assert mrr(retrieved, gold) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(retrieved, gold, 10) == pytest.approx(dcg / idcg)


def test_perfect_and_empty_rankings():
    gold = [Span("D", 0, 10)]
    assert retrieval_metrics([Span("D", 0, 10)], gold)["ndcg@10"] == 1.0
    m = retrieval_metrics([], gold)
    assert m["recall@10"] == 0 and m["mrr"] == 0 and m["ndcg@10"] == 0
    with pytest.raises(ValueError):
        recall_at_k([], [], 5)


def test_duplicates_cannot_push_scores_above_one():
    gold = [Span("D", 0, 100)]
    retrieved = [Span("D", 0, 100)] * 10
    assert ndcg_at_k(retrieved, gold) == 1.0 and recall_at_k(retrieved, gold, 10) == 1.0


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    vals = [0, 1] * 50
    a, b = bootstrap_ci(vals), bootstrap_ci(vals)
    assert a == b
    assert a.low < 0.5 < a.high and a.mean == 0.5 and a.n == 100
    assert bootstrap_ci([0.7] * 10).low == pytest.approx(0.7)


def test_paired_bootstrap_detects_a_real_improvement():
    a = [0.2, 0.4, 0.3, 0.5, 0.1] * 10
    better = [x + 0.1 for x in a]
    res = paired_bootstrap(a, better)
    assert res.diff.mean == pytest.approx(0.1) and res.diff.low > 0 and res.p_not_better == 0
    same = paired_bootstrap(a, a)
    assert same.diff.mean == 0 and same.p_not_better == 1
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
