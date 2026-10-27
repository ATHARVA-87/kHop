import json
import math

import pytest

from plexus.config import ExperimentConfig
from plexus.index.dense import Hit
from plexus.retrieve.bm25 import BM25Index, s_stem, tokenize
from plexus.retrieve.fusion import rrf
from plexus.retrieve.pipeline import BM25Retriever, HybridRetriever


def test_s_stem_folds_plurals_only():
    assert s_stem("foundries") == "foundry"
    assert s_stem("wafers") == "wafer"
    assert s_stem("rambus") == "rambus"  # -us kept
    assert s_stem("business") == "business"  # -ss kept
    assert s_stem("shoes") == "shoe"  # -oes keeps the e


def test_tokenize_drops_stopwords_keeps_numbers_and_decimals():
    toks = tokenize("What did the Company's foundries ship in 2025? Margin was 4.5%.")
    assert toks == ["company", "foundry", "ship", "2025", "margin", "4.5"]


def test_bm25_matches_hand_computed_score():
    idx = BM25Index.build(
        ["tsmc wafer", "wafer wafer supply", "unrelated text here"], k1=1.2, b=0.75
    )
    # doc 0: "tsmc" tf=1, len 2, avgdl = (2+3+3)/3
    avgdl = 8 / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / avgdl))
    assert idx.scores("tsmc")[0] == pytest.approx(expected, rel=1e-5)


def test_bm25_ranks_rare_term_match_first_and_skips_zero_scores():
    idx = BM25Index.build(["risk risk risk", "mohawk valley fab ramp", "risk of the fab"])
    res = idx.search("Mohawk Valley risk", 10)
    assert res[0][0] == 1
    assert {i for i, _ in res} == {0, 1, 2}
    assert idx.search("nonexistent", 10) == []


def test_bm25_save_load_roundtrip(tmp_path):
    idx = BM25Index.build(["alpha beta", "beta gamma"])
    idx.save(tmp_path / "b.pkl")
    again = BM25Index.load(tmp_path / "b.pkl")
    assert again is not None
    assert list(again.scores("beta")) == pytest.approx(list(idx.scores("beta")))


def _hit(cid, score=0.0):
    return Hit(cid, "D", 0, 1, "s", score, cid)


def test_rrf_rewards_agreement_and_dedupes():
    a = [_hit("x"), _hit("y"), _hit("z")]
    b = [_hit("y"), _hit("w")]
    fused = rrf([a, b], k=60)
    assert [h.chunk_id for h in fused] == ["y", "x", "w", "z"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_is_deterministic_on_ties():
    a = [_hit("b"), _hit("a")]
    b = [_hit("a"), _hit("b")]
    assert [h.chunk_id for h in rrf([a, b])] == ["b", "a"]  # tie -> first list's rank wins


def test_bm25_retriever_builds_and_caches(tmp_path):
    chunks = [
        {"chunk_id": f"D#c{i}", "doc_id": "D", "start": i, "end": i + 1, "section": "s", "text": t}
        for i, t in enumerate(["TSMC is our sole foundry", "We sell to Apple", "Inventory risk"])
    ]
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks) + "\n")
    r = BM25Retriever.from_index_dir(tmp_path)
    assert (tmp_path / "bm25.pkl").exists()
    assert r.retrieve("Which foundry?", 2)[0].chunk_id == "D#c0"
    assert BM25Retriever.from_index_dir(tmp_path).retrieve("Apple", 1)[0].chunk_id == "D#c1"


def test_hybrid_fuses_candidate_pools():
    class Fixed:
        def __init__(self, ids):
            self.ids = ids

        def retrieve(self, q, k):
            return [_hit(i) for i in self.ids[:k]]

    h = HybridRetriever([Fixed(["a", "b", "c"]), Fixed(["c", "d"])], candidates=3)
    assert [x.chunk_id for x in h.retrieve("q", 2)] == ["c", "a"]


def test_retrieval_config_rejects_unknown_type():
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"name": "x", "retrieval": {"type": "magic"}})


def test_compare_paired_diff(tmp_path):
    from plexus.eval.compare import compare

    def write(name, vals):
        rows = [{"qid": f"q{i}", "qtype": "local", "recall@10": v} for i, v in enumerate(vals)]
        (tmp_path / f"{name}.json").write_text(json.dumps({"per_query": rows}))
        return tmp_path / f"{name}.json"

    res = compare(write("a", [0, 0, 1, 0]), write("b", [1, 1, 1, 0]), metrics=("recall@10",))
    assert res["all"]["recall@10"]["diff"]["mean"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compare(write("a", [0, 1]), write("c", [0]), metrics=("recall@10",))


def test_reranker_reorders_only_the_candidate_head():
    from plexus.retrieve.rerank import RerankingRetriever

    class Base:
        def retrieve(self, q, k):
            return [_hit(c) for c in "abcdef"[:k]]

    # scores favour later letters; only the first 3 candidates are reranked
    rr = RerankingRetriever(Base(), lambda q, ps: [ord(p) for p in ps], candidates=3)
    assert [h.chunk_id for h in rr.retrieve("q", 6)] == ["c", "b", "a", "d", "e", "f"]
    # the reranked prefix does not depend on k
    assert [h.chunk_id for h in rr.retrieve("q", 2)] == ["c", "b"]


def test_reranker_ties_keep_base_order():
    from plexus.retrieve.rerank import RerankingRetriever

    class Base:
        def retrieve(self, q, k):
            return [_hit(c) for c in "xyz"]

    rr = RerankingRetriever(Base(), lambda q, ps: [1.0] * len(ps), candidates=50)
    assert [h.chunk_id for h in rr.retrieve("q", 3)] == ["x", "y", "z"]
