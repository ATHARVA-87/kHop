import json

from plexus.ingest.process import iter_docs, process_snapshot
from tests.test_parse import TENK, TENQ


def test_process_snapshot_round_trip(tmp_path):
    raw = tmp_path / "raw" / "snap" / "NVDA"
    raw.mkdir(parents=True)
    (raw / "k.htm").write_text(TENK)
    (raw / "q.htm").write_text(TENQ)
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    rows = [
        {"doc_id": "NVDA_10K", "form": "10-K", "path": "snap/NVDA/k.htm"},
        {"doc_id": "NVDA_10Q", "form": "10-Q", "path": "snap/NVDA/q.htm"},
    ]
    for r in rows:
        r.update(
            ticker="NVDA", company="NVIDIA", filing_date="2025-02-26", report_date="", sha256="x"
        )
    (snaps / "snap.manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    report = process_snapshot(tmp_path, "snap")
    assert report["documents"] == 2
    assert report["coverage"]["10-K"] == {"business": 1.0, "risk_factors": 1.0, "mdna": 1.0}
    assert report["coverage"]["10-Q"] == {"mdna": 1.0, "financial_statements": 1.0}
    assert (snaps / "snap.parse_report.json").exists()

    docs = list(iter_docs(tmp_path / "processed", "snap"))
    assert [m["doc_id"] for m, _ in docs] == ["NVDA_10K", "NVDA_10Q"]
    meta, doc = docs[0]
    assert (
        meta["ticker"] == "NVDA"
        and doc.section_at(doc.text.index("single foundry")).name == "risk_factors"
    )
