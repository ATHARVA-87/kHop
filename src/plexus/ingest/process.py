"""Parse every document in a snapshot and report section-detection quality.

    python -m plexus.ingest.process --snapshot semis-2025

Writes data/processed/<snapshot>/docs.jsonl and data/snapshots/<snapshot>.parse_report.json
(the report is committed: it is the evidence that the parser works on real filings).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from plexus.ingest.parse import ParsedDoc, Section, parse_filing
from plexus.settings import Settings

# Sections each form should have. Missing ones are flagged in the report.
EXPECTED = {"10-K": ["business", "risk_factors", "mdna"], "10-Q": ["mdna", "financial_statements"]}


def load_manifest(snapshots_dir: Path, snapshot: str) -> list[dict]:
    path = snapshots_dir / f"{snapshot}.manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make ingest` first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iter_docs(processed_dir: Path, snapshot: str) -> Iterator[tuple[dict, ParsedDoc]]:
    """Yield (metadata, ParsedDoc) from docs.jsonl."""
    with open(processed_dir / snapshot / "docs.jsonl") as fh:
        for line in fh:
            row = json.loads(line)
            sections = [Section(**s) for s in row.pop("sections")]
            doc = ParsedDoc(row["doc_id"], row.pop("text"), sections)
            yield row, doc


def process_snapshot(data_dir: Path, snapshot: str) -> dict:
    manifest = load_manifest(data_dir / "snapshots", snapshot)
    out_dir = data_dir / "processed" / snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc = []
    with open(out_dir / "docs.jsonl", "w") as fh:
        for m in manifest:
            html = (data_dir / "raw" / m["path"]).read_bytes()
            doc = parse_filing(m["doc_id"], html, m["form"])
            meta = {
                k: m[k]
                for k in (
                    "doc_id",
                    "ticker",
                    "company",
                    "form",
                    "filing_date",
                    "report_date",
                    "sha256",
                )
            }
            fh.write(
                json.dumps(
                    {**meta, "text": doc.text, "sections": [asdict(s) for s in doc.sections]}
                )
                + "\n"
            )
            found = {s.name for s in doc.sections}
            missing = [e for e in EXPECTED.get(m["form"], []) if e not in found]
            per_doc.append(
                {
                    "doc_id": m["doc_id"],
                    "chars": len(doc.text),
                    "sections": {s.name: s.end - s.start for s in doc.sections},
                    "missing_expected": missing,
                }
            )
    report = {
        "snapshot": snapshot,
        "documents": len(per_doc),
        "docs_missing_expected_sections": sum(bool(d["missing_expected"]) for d in per_doc),
        "coverage": {
            form: {
                name: sum(
                    name in d["sections"]
                    for d, m in zip(per_doc, manifest, strict=True)
                    if m["form"] == form
                )
                / max(1, sum(m["form"] == form for m in manifest))
                for name in names
            }
            for form, names in EXPECTED.items()
        },
        "per_doc": per_doc,
    }
    (data_dir / "snapshots" / f"{snapshot}.parse_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="semis-2025")
    args = ap.parse_args(argv)
    report = process_snapshot(Settings.load().data_dir, args.snapshot)
    print(json.dumps({k: v for k, v in report.items() if k != "per_doc"}, indent=2))
    bad = [d for d in report["per_doc"] if d["missing_expected"]]
    for d in bad[:20]:
        print(f"  {d['doc_id']}: missing {d['missing_expected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
