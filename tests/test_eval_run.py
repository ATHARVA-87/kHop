import pytest
from pydantic import ValidationError

from plexus.config import ExperimentConfig
from plexus.eval.golden import Evidence, GoldenQuery
from plexus.eval.run import aggregate, gold_spans, score_retrieval
from plexus.index.dense import Hit

DOCS = {
    "NVDA_10K": "Intro. We depend on TSMC to manufacture our GPUs. Export controls limit China sales.",
    "AMD_10K": "We also depend on TSMC for leading-edge wafers.",
}


def _q(qid, qtype, evidence):
    return GoldenQuery(
        qid=qid,
        qtype=qtype,
        question=f"{qid}?",
        answer="a",
        evidence=[Evidence(doc_id=d, quote=q) for d, q in evidence],
        verdict="accept",
    )


def _hit(doc_id, phrase):
    s = DOCS[doc_id].index(phrase)
    return Hit(f"{doc_id}#x", doc_id, s, s + len(phrase), "risk_factors", 1.0)


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, question, k=None):
        return self.ranking[question][:k]


def test_score_retrieval_end_to_end_with_label_errors():
    queries = [
        _q("local1", "local", [("NVDA_10K", "depend on TSMC to manufacture")]),
        _q(
            "hop1", "multi_hop", [("NVDA_10K", "depend on TSMC"), ("AMD_10K", "depend on TSMC for")]
        ),
        _q("bad1", "local", [("NVDA_10K", "this quote was paraphrased")]),
        _q("bad2", "local", [("XYZ_10K", "unknown doc")]),
    ]
    retriever = FakeRetriever(
        {
            "local1?": [_hit("NVDA_10K", "We depend on TSMC to manufacture our GPUs.")],
            "hop1?": [
                _hit("NVDA_10K", "Export controls"),
                _hit("AMD_10K", "We also depend on TSMC for"),
            ],
        }
    )
    rows, errors = score_retrieval(queries, retriever, DOCS, k=10)
    assert [r["qid"] for r in rows] == ["local1", "hop1"]  # bad queries are not scored
    assert {e["reason"] for e in errors} == {"quote not found", "unknown doc_id"}
    local, hop = rows
    assert local["recall@1"] == 1 and local["mrr"] == 1
    assert hop["recall@10"] == 0.5 and hop["mrr"] == 0.5

    agg = aggregate(rows)
    assert set(agg) == {"all", "local", "multi_hop"}
    assert agg["all"]["recall@10"]["mean"] == 0.75
    assert agg["local"]["recall@10"]["n"] == 1
    assert "p95" in agg["all"]["retrieval_ms"]


def test_gold_spans_locate_quotes():
    spans, missing = gold_spans(_q("x", "local", [("AMD_10K", "LEADING-EDGE  wafers")]), DOCS)
    assert not missing and DOCS["AMD_10K"][spans[0].start : spans[0].end] == "leading-edge wafers"


def test_configs_load_and_typos_fail(tmp_path):
    cfg = ExperimentConfig.load("configs/a0_naive.yaml")
    assert cfg.name == "a0_naive" and cfg.retrieval.top_k == 20
    assert cfg.index_key == "semis-2025__fixed-300-50__BAAI_bge-small-en-v1.5"
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nretreival:\n  top_k: 5\n")
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad)
