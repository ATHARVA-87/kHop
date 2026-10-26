"""Golden set: candidates -> human review (review.csv) -> verified golden.jsonl.

    python -m plexus.eval.golden build     # review.csv -> golden.jsonl + noise report

Rules enforced here:
- Only rows the human marked `accept` or `edit` enter golden.jsonl. `reject` and blank
  verdicts never do, so unverified queries are never scored.
- Label-noise rate = (edit + reject) / reviewed, reported in data/golden/review_stats.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from plexus.settings import REPO_ROOT

QueryType = Literal["local", "multi_hop", "global"]
Verdict = Literal["accept", "edit", "reject", ""]

GOLDEN_DIR = REPO_ROOT / "data" / "golden"
EVIDENCE_SEP = " ||| "

REVIEW_COLUMNS = [
    "qid", "qtype", "question", "answer", "evidence_doc_ids", "evidence_quotes",
    "sub_questions", "source",
    # filled in by the human reviewer:
    "verdict", "corrected_question", "corrected_answer", "corrected_evidence_doc_ids",
    "corrected_evidence_quotes", "corrected_qtype", "reviewer_notes",
]  # fmt: skip


class Evidence(BaseModel):
    doc_id: str
    quote: str  # verbatim (whitespace-insensitive) text from the document


class GoldenQuery(BaseModel):
    qid: str
    qtype: QueryType
    question: str
    answer: str
    evidence: list[Evidence] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)  # multi-hop decomposition
    source: str = ""  # e.g. "llm:gemini-2.0-flash" or "human"
    verdict: Verdict = ""
    reviewer_notes: str = ""

    @field_validator("question", "answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @property
    def verified(self) -> bool:
        return self.verdict in ("accept", "edit")


def _split(cell: str) -> list[str]:
    return [p.strip() for p in (cell or "").split(EVIDENCE_SEP) if p.strip()]


def to_review_row(q: GoldenQuery) -> dict[str, str]:
    row = dict.fromkeys(REVIEW_COLUMNS, "")
    row.update(
        qid=q.qid,
        qtype=q.qtype,
        question=q.question,
        answer=q.answer,
        evidence_doc_ids=EVIDENCE_SEP.join(e.doc_id for e in q.evidence),
        evidence_quotes=EVIDENCE_SEP.join(e.quote for e in q.evidence),
        sub_questions=EVIDENCE_SEP.join(q.sub_questions),
        source=q.source,
    )
    return row


def write_review_csv(queries: list[GoldenQuery], path: Path) -> None:
    """Write candidates for review. Refuses to overwrite rows that already have a verdict."""
    existing = read_review_csv(path) if path.exists() else []
    reviewed = {r["qid"] for r in existing if r.get("verdict", "").strip()}
    clash = reviewed & {q.qid for q in queries}
    if clash:
        raise ValueError(f"refusing to overwrite reviewed rows: {sorted(clash)[:5]}...")
    keep = [r for r in existing if r["qid"] in reviewed]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in REVIEW_COLUMNS})
        for q in queries:
            w.writerow(to_review_row(q))


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def row_to_query(row: dict[str, str]) -> GoldenQuery:
    """Apply the reviewer's corrections (if any) on top of the candidate."""
    verdict = row.get("verdict", "").strip().lower()
    if verdict not in ("accept", "edit", "reject", ""):
        raise ValueError(f"{row['qid']}: verdict must be accept/edit/reject, got {verdict!r}")

    def pick(field: str) -> str:
        return (row.get(f"corrected_{field}") or "").strip() or row.get(field, "")

    doc_ids, quotes = _split(pick("evidence_doc_ids")), _split(pick("evidence_quotes"))
    if len(doc_ids) != len(quotes):
        raise ValueError(f"{row['qid']}: {len(doc_ids)} evidence doc_ids but {len(quotes)} quotes")
    return GoldenQuery(
        qid=row["qid"],
        qtype=pick("qtype").strip(),
        question=pick("question"),
        answer=pick("answer"),
        evidence=[Evidence(doc_id=d, quote=q) for d, q in zip(doc_ids, quotes, strict=True)],
        sub_questions=_split(row.get("sub_questions", "")),
        source=row.get("source", ""),
        verdict=verdict,
        reviewer_notes=row.get("reviewer_notes", ""),
    )


def build_golden(review_csv: Path, out_jsonl: Path, stats_json: Path) -> dict:
    rows = read_review_csv(review_csv)
    queries = [row_to_query(r) for r in rows]
    counts = Counter(q.verdict or "unreviewed" for q in queries)
    reviewed = counts["accept"] + counts["edit"] + counts["reject"]
    verified = [q for q in queries if q.verified]
    out_jsonl.write_text("".join(q.model_dump_json() + "\n" for q in verified))
    stats = {
        "candidates": len(queries),
        "reviewed": reviewed,
        "reviewed_fraction": round(reviewed / len(queries), 4) if queries else 0.0,
        "verdicts": dict(counts),
        # noise = share of reviewed candidates that were wrong enough to fix or drop
        "label_noise_rate": round((counts["edit"] + counts["reject"]) / reviewed, 4)
        if reviewed
        else None,
        "verified_by_type": dict(Counter(q.qtype for q in verified)),
    }
    stats_json.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_golden(path: Path | None = None, qtypes: set[str] | None = None) -> list[GoldenQuery]:
    """Load verified queries only. Raises if there are none (never score on nothing)."""
    path = path or GOLDEN_DIR / "golden.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: review data/golden/review.csv, then build it")
    qs = [GoldenQuery.model_validate_json(line) for line in path.read_text().splitlines() if line]
    qs = [q for q in qs if q.verified and (qtypes is None or q.qtype in qtypes)]
    if not qs:
        raise ValueError(f"no verified queries in {path}")
    return qs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--review", type=Path, default=GOLDEN_DIR / "review.csv")
    args = ap.parse_args(argv)
    stats = build_golden(args.review, GOLDEN_DIR / "golden.jsonl", GOLDEN_DIR / "review_stats.json")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
