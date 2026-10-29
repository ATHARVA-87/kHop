from plexus.eval.drift import carry_over, recall_pairs, status_table
from plexus.eval.golden import Evidence, GoldenQuery


def q(qid, *evidence, qtype="local"):
    return GoldenQuery(
        qid=qid,
        qtype=qtype,
        question="What does the filing say?",
        answer="Something.",
        evidence=[Evidence(doc_id=d, quote=t) for d, t in evidence],
    )


NEW = {
    "NVDA_10K_2026-02-25": "Our  products are used in data centers. Demand may fall.",
    "NVDA_10Q_2026-05-27": "Demand may fall.",
    "AMD_10Q_2026-05-05": "We rely on TSMC for wafers.",
}


def test_carry_over_points_at_same_form_and_keeps_quotes():
    kept, status = carry_over(
        [q("a", ("NVDA_10K_2025-02-26", "our products are used in data centers"))], NEW
    )
    assert status == {"a": "carried"}
    assert kept[0].evidence[0].doc_id == "NVDA_10K_2026-02-25"  # whitespace/case ignored
    assert kept[0].evidence[0].quote == "our products are used in data centers"


def test_carry_over_prefers_same_form_then_any_form():
    kept, _ = carry_over([q("a", ("NVDA_10Q_2025-05-28", "Demand may fall."))], NEW)
    assert kept[0].evidence[0].doc_id == "NVDA_10Q_2026-05-27"
    kept, _ = carry_over([q("b", ("AMD_10K_2025-02-04", "We rely on TSMC for wafers."))], NEW)
    assert kept[0].evidence[0].doc_id == "AMD_10Q_2026-05-05"


def test_statuses():
    queries = [
        q("gone", ("NVDA_10K_2025-02-26", "Revenue was $130.5 billion.")),
        q(
            "partly",
            ("NVDA_10K_2025-02-26", "Demand may fall."),
            ("AMD_10K_2025-02-04", "A sentence AMD no longer writes."),
        ),
        q("nofile", ("INTC_10K_2025-01-31", "Anything.")),
        q("none"),
    ]
    kept, status = carry_over(queries, NEW)
    assert kept == []
    assert status == {
        "gone": "gone",
        "partly": "partly",
        "nofile": "no_filings",
        "none": "no_evidence",
    }
    table = status_table(queries, status)
    assert table["all"]["gone"] == 1 and table["local"]["no_evidence"] == 1


def test_never_matches_another_company():
    _, status = carry_over([q("a", ("AMD_10K_2025-02-04", "Demand may fall."))], NEW)
    assert status["a"] == "gone"


def test_recall_pairs_only_questions_on_both():
    qids, a, b = recall_pairs(
        [{"qid": "x", "recall@10": 1.0}, {"qid": "y", "recall@10": 0.0}],
        [{"qid": "y", "recall@10": 1.0}],
    )
    assert qids == ["y"] and a == [0.0] and b == [1.0]
