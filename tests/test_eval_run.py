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
    [(g,)] = spans  # one evidence item, one occurrence
    assert not missing and DOCS["AMD_10K"][g.start : g.end] == "leading-edge wafers"


def test_configs_load_and_typos_fail(tmp_path):
    cfg = ExperimentConfig.load("configs/a0_naive.yaml")
    assert cfg.name == "a0_naive" and cfg.retrieval.top_k == 20
    from plexus.ingest.parse import VERSION

    assert cfg.index_key == f"semis-2025__p{VERSION}__fixed-300-50__BAAI_bge-small-en-v1.5"
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nretreival:\n  top_k: 5\n")
    with pytest.raises(ValidationError):
        ExperimentConfig.load(bad)


def test_score_generation_citation_precision_and_refusals():
    from plexus.baseline.naive import NOT_ENOUGH, Answer
    from plexus.eval.run import aggregate_generation, score_generation

    good = _hit("NVDA_10K", "We depend on TSMC to manufacture our GPUs.")
    off = _hit("NVDA_10K", "Export controls limit China sales.")
    off = Hit("NVDA_10K#y", off.doc_id, off.start, off.end, off.section, off.score)

    class Pipe:
        def answer(self, question):
            if question == "local1?":
                return Answer(question, "TSMC [1]. China [2].", [good, off], ["NVDA_10K#x", "NVDA_10K#y"],
                              generation_ms=5.0)  # fmt: skip
            return Answer(question, NOT_ENOUGH, [good], [], cached=True)

    queries = [
        _q("local1", "local", [("NVDA_10K", "depend on TSMC to manufacture")]),
        _q("local2", "local", [("AMD_10K", "leading-edge wafers")]),
    ]
    rows = [{"qid": "local1", "qtype": "local"}, {"qid": "local2", "qtype": "local"}]
    score_generation(queries, rows, Pipe(), DOCS)
    assert rows[0]["citation_gold_precision"] == 0.5 and rows[0]["n_citations"] == 2
    assert rows[1]["refused"] and rows[1]["citation_gold_precision"] is None
    agg = aggregate_generation(rows)["all"]
    assert agg["refusal_rate"] == 0.5
    assert agg["citation_gold_precision"]["n"] == 1  # refusal has no citations to score
    assert agg["faithfulness"] is None  # no judge was run
    assert agg["generation_ms_uncached"]["p50"] == 5.0  # cached answers excluded


def test_query_with_partly_missing_evidence_is_not_scored():
    # review finding #1: dropping the missing quote inflated recall from 0.5 to 1.0
    q = _q(
        "hop",
        "multi_hop",
        [("NVDA_10K", "depend on TSMC"), ("AMD_10K", "a paraphrase not in the doc")],
    )
    retr = FakeRetriever({"hop?": [_hit("NVDA_10K", "We depend on TSMC to manufacture our GPUs.")]})
    rows, errors = score_retrieval([q], retr, DOCS, k=10)
    assert rows == [] and errors[0]["qid"] == "hop"


def test_recall_beyond_top_k_is_not_reported():
    q = _q("l", "local", [("NVDA_10K", "depend on TSMC")])
    retr = FakeRetriever({"l?": [_hit("NVDA_10K", "We depend on TSMC to manufacture our GPUs.")]})
    rows, _ = score_retrieval([q], retr, DOCS, k=5)
    assert "recall@5" in rows[0] and "recall@10" not in rows[0] and "recall@20" not in rows[0]
    assert "recall@10" not in aggregate(rows)["all"]
