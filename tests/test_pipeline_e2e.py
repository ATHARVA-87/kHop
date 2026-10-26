"""End-to-end A0 on a synthetic 3-filing corpus: manifest -> parse -> chunk -> embed ->
index -> golden -> score -> results file. Uses the real bge-small model (ml marker)."""

import json

import pytest

from plexus.config import ExperimentConfig
from plexus.eval.golden import Evidence, GoldenQuery

pytestmark = pytest.mark.ml

PAD = " ".join(f"Filler sentence number {i} about general corporate matters." for i in range(60))


def _filing(name, business, risks, mdna):
    return f"""<html><body>
<p>Table of Contents</p><p>Item 1. Business</p><p>Item 1A. Risk Factors</p>
<p>Item 1. Business</p><p>{name} overview. {business} {PAD}</p>
<p>Item 1A. Risk Factors</p><p>{risks} {PAD}</p>
<p>Item 7. Management's Discussion and Analysis</p><p>{mdna} {PAD}</p>
</body></html>"""


FILINGS = {
    "NVDA_10K_2025-02-26": _filing(
        "NVIDIA",
        "We design accelerated computing platforms used in data centers.",
        "We depend on Taiwan Semiconductor Manufacturing Company to manufacture our GPUs.",
        "Data center revenue grew strongly because of demand for AI training systems.",
    ),
    "AMD_10K_2025-02-05": _filing(
        "AMD",
        "We design CPUs and GPUs for PCs, gaming consoles and servers.",
        "U.S. export controls restrict shipments of certain accelerators to China.",
        "Gaming segment revenue declined due to lower semi-custom console sales.",
    ),
    "MU_10K_2025-10-03": _filing(
        "Micron",
        "We manufacture DRAM and NAND memory in our own fabs.",
        "Memory pricing is cyclical and oversupply can sharply reduce our margins.",
        "High-bandwidth memory shipments increased for AI servers.",
    ),
}


def test_a0_end_to_end(tmp_path):
    from plexus.eval.golden import build_golden, write_review_csv
    from plexus.eval.run import run
    from plexus.index.build import build
    from plexus.ingest.process import process_snapshot

    data = tmp_path / "data"
    (data / "snapshots").mkdir(parents=True)
    rows = []
    for doc_id, html in FILINGS.items():
        p = data / "raw" / "e2e" / f"{doc_id}.htm"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html)
        rows.append(
            {"doc_id": doc_id, "ticker": doc_id.split("_")[0], "company": doc_id, "form": "10-K",
             "filing_date": doc_id[-10:], "report_date": "", "sha256": "x", "path": f"e2e/{doc_id}.htm"}
        )  # fmt: skip
    (data / "snapshots" / "e2e.manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    (data / "snapshots" / "e2e.summary.json").write_text(json.dumps({"corpus_checksum": "abc"}))

    report = process_snapshot(data, "e2e")
    assert report["coverage"]["10-K"]["risk_factors"] == 1.0

    cfg = ExperimentConfig(name="a0_e2e", snapshot="e2e")
    cfg = cfg.model_copy(update={"embedder": cfg.embedder.model_copy(update={"device": "cpu"})})
    info = build(cfg, data)
    assert info["documents"] == 3 and info["chunks"] == info["indexed_points"] > 3

    golden = [
        GoldenQuery(qid="q1", qtype="local", question="Which foundry manufactures NVIDIA's GPUs?",
                    answer="TSMC", evidence=[Evidence(doc_id="NVDA_10K_2025-02-26",
                    quote="depend on Taiwan Semiconductor Manufacturing Company")]),
        GoldenQuery(qid="q2", qtype="local", question="What happened to AMD gaming revenue?",
                    answer="It declined.", evidence=[Evidence(doc_id="AMD_10K_2025-02-05",
                    quote="Gaming segment revenue declined")]),
        GoldenQuery(qid="q3", qtype="local", question="Why are Micron's margins volatile?",
                    answer="Cyclical memory pricing.", evidence=[Evidence(doc_id="MU_10K_2025-10-03",
                    quote="this text is not in the filing")]),
    ]  # fmt: skip
    gdir = data / "golden"
    gdir.mkdir()
    write_review_csv(golden, gdir / "review.csv")
    # mark all three accepted, as a human reviewer would
    from tests.test_golden import _set_verdicts

    _set_verdicts(gdir / "review.csv", {q.qid: {"verdict": "accept"} for q in golden})
    build_golden(gdir / "review.csv", gdir / "golden.jsonl", gdir / "review_stats.json")

    res = run(cfg, data_dir=data, results_dir=tmp_path / "results")
    assert (tmp_path / "results" / "a0_e2e.json").exists()
    assert res["golden"]["scored_queries"] == 2  # q3's quote cannot be located
    assert res["golden"]["label_errors"][0]["qid"] == "q3"
    assert res["corpus_checksum"] == "abc"
    per_q = {r["qid"]: r for r in res["per_query"]}
    assert per_q["q1"]["recall@5"] == 1.0 and per_q["q2"]["recall@5"] == 1.0
    assert res["retrieval"]["all"]["recall@20"]["mean"] == 1.0
