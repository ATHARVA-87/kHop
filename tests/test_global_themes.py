import csv
import json

import pytest

from plexus.eval.bootstrap import bootstrap_ci
from plexus.eval.global_themes import COLUMNS, build, load_labels, spot_check_stats
from plexus.eval.golden import Evidence, GoldenQuery
from plexus.eval.run import aggregate, company_coverage, score_retrieval
from plexus.index.dense import Hit
from plexus.llm import FakeProvider, LLMCache, LLMClient
from tests.test_candidates import _corpus


def test_build_labels_companies_and_drops_unverifiable_quotes(tmp_path, monkeypatch):
    data = _corpus(tmp_path)  # NVDA/AMD/QCOM filings mentioning TSMC and export control
    monkeypatch.setattr(
        "plexus.eval.global_themes.load_themes",
        lambda path=None: {"export control": "U.S. export controls", "climate": "climate rules"},
    )

    def responder(req):
        prompt = req.messages[-1].content
        quote = (
            "export control rules"
            if "Qualcomm" not in prompt
            else "made-up words not in the filing at all"
        )
        return json.dumps({"themes": [
            {"theme": "U.S. export controls", "discussed": True, "quote": quote},
            {"theme": "climate rules", "discussed": True, "quote": ""},  # no keyword hits -> forced no
        ]})  # fmt: skip

    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"gemini": FakeProvider(responder)})
    out = tmp_path / "themes.csv"
    stats = build(llm, data, "s", "gemini", ["m"], out)
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert stats["rows"] == 6 and list(rows[0]) == COLUMNS
    ec = {r["ticker"]: r for r in rows if r["keyword"] == "export control"}
    assert all(r["llm_discussed"] == "yes" for r in ec.values())
    assert ec["QCOM"]["quote"] == "" and stats["quote_not_found"] == 1  # unverifiable quote dropped
    assert all(r["llm_discussed"] == "no" for r in rows if r["keyword"] == "climate")
    assert sum(r["spot_check"] == "yes" for r in rows) == 1  # ~20% sample
    assert load_labels(out) == {"export control": {"NVDA", "AMD", "QCOM"}}

    # a human "no" overrides the LLM; agreement stats
    for r in rows:
        if r["ticker"] == "AMD" and r["keyword"] == "export control":
            r["human_discussed"] = "no"
        if r["ticker"] == "NVDA" and r["keyword"] == "climate":
            r["human_discussed"] = "no"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert load_labels(out) == {"export control": {"NVDA", "QCOM"}}
    st = spot_check_stats(out)
    assert st["checked"] == 2 and st["agreement"] == 0.5 and st["llm_yes_human_no"] == 1


def _hit(doc_id, text):
    return Hit(f"{doc_id}#c", doc_id, 0, len(text), "risk_factors", 1.0, text)


def test_company_coverage_counts_on_theme_chunks_of_relevant_companies():
    hits = [
        _hit("NVDA_10K_x", "Export controls restrict our China sales."),
        _hit("NVDA_10Q_y", "More export control text."),  # same company twice -> once
        _hit("AMD_10K_x", "Gaming revenue declined."),  # relevant company, off-theme
        _hit("XYZ_10K_x", "Export control everywhere."),  # not a relevant company
    ]
    assert company_coverage(hits, {"NVDA", "AMD", "MU", "INTC"}, "export control", 10) == 0.25
    assert company_coverage(hits[2:], {"NVDA"}, "export control", 10) == 0.0


def test_global_rows_get_coverage_and_cluster_bootstrap_by_theme():
    docs = {
        "NVDA_10K_x": "Export controls restrict our China sales.",
        "AMD_10K_x": "export control",
    }
    qs = [
        GoldenQuery(qid=f"G{i}", qtype="global", question=f"g{i}?", answer="a",
                    evidence=[Evidence(doc_id="NVDA_10K_x", quote="Export controls restrict our China sales")],
                    source=f"llm:x theme={'U.S. export controls' if i < 3 else 'exposure to China'}",
                    verdict="accept")
        for i in range(5)
    ]  # fmt: skip

    class R:
        def retrieve(self, question, k=None):
            return [_hit("NVDA_10K_x", docs["NVDA_10K_x"])]

    labels = {"export control": {"NVDA", "AMD"}, "China": {"NVDA"}}
    l2k = {"U.S. export controls": "export control", "exposure to China": "China"}
    rows, _ = score_retrieval(qs, R(), docs, 20, labels, l2k)
    assert [r["company_coverage@10"] for r in rows] == [0.5, 0.5, 0.5, 1.0, 1.0]
    assert rows[0]["theme"] == "export control" and rows[0]["n_relevant_companies"] == 2
    cov = aggregate(rows)["global"]["company_coverage@10"]
    assert cov["mean"] == pytest.approx(0.7)
    # only 2 themes -> resampling themes gives a wide interval (0.5 .. 1.0)
    assert cov["low"] == 0.5 and cov["high"] == 1.0


def test_cluster_bootstrap_is_wider_than_naive_for_correlated_clusters():
    vals = [0.0] * 10 + [1.0] * 10
    clusters = ["a"] * 10 + ["b"] * 10
    naive, clus = bootstrap_ci(vals), bootstrap_ci(vals, clusters=clusters)
    assert (clus.high - clus.low) > (naive.high - naive.low)


def test_theme_synonyms_catch_usual_wording_and_skip_false_friends():
    from plexus.eval.global_themes import theme_regex

    assert theme_regex("customer concentration").search("a small number of customers account for")
    assert theme_regex("single source").search("certain components are sole-source")
    assert theme_regex("artificial intelligence").search("demand for AI accelerators")
    assert not theme_regex("artificial intelligence").search("the said fee was paid")
    assert not theme_regex("climate").search("the current business climate")
    hits = [_hit("MU_10K_x", "We rely on a limited number of customers for most revenue.")]
    assert company_coverage(hits, {"MU"}, "customer concentration", 10) == 1.0
