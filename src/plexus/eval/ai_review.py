"""Merge AI-reviewer suggestions into the review CSVs, in separate ai_* columns.

    python -m plexus.eval.ai_review merge [data/golden/ai_review/*.jsonl]
    python -m plexus.eval.ai_review agreement

The human-owned columns (verdict, corrected_*, human_discussed) are never written here.
The human confirms or overrides each AI suggestion, so the golden set stays
human-verified, and AI-vs-human agreement becomes a reported number.

Suggestion lines (JSONL):
  {"file": "review.csv", "qid": "L-…", "ai_verdict": "accept|edit|reject",
   "ai_corrected_question": "", "ai_corrected_answer": "", "ai_corrected_qtype": "",
   "ai_corrected_evidence_doc_ids": "", "ai_corrected_evidence_quotes": "", "ai_notes": "…"}
  {"file": "global_themes.csv", "theme": "…", "ticker": "NVDA", "ai_discussed": "yes|no", "ai_notes": "…"}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from plexus.eval.golden import GOLDEN_DIR

QUESTION_AI_COLS = ["ai_verdict", "ai_corrected_question", "ai_corrected_answer", "ai_corrected_qtype",
                    "ai_corrected_evidence_doc_ids", "ai_corrected_evidence_quotes", "ai_notes"]  # fmt: skip
THEME_AI_COLS = ["ai_discussed", "ai_notes"]
HUMAN_COLS = {"verdict", "human_discussed"} | {
    f"corrected_{x}" for x in ("question", "answer", "qtype", "evidence_doc_ids", "evidence_quotes")
}


def _read(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def merge(suggestion_files: list[Path], golden_dir: Path = GOLDEN_DIR) -> dict:
    by_file: dict[str, dict[tuple, dict]] = {}
    for f in suggestion_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            s = json.loads(line)
            key = (s["qid"],) if "qid" in s else (s["theme"], s["ticker"])
            by_file.setdefault(s["file"], {})[key] = s
    report = {}
    for fname, sugg in by_file.items():
        path = golden_dir / fname
        fields, rows = _read(path)
        ai_cols = THEME_AI_COLS if fname == "global_themes.csv" else QUESTION_AI_COLS
        fields += [c for c in ai_cols if c not in fields]
        n = 0
        for r in rows:
            key = (r["qid"],) if "qid" in r else (r["theme"], r["ticker"])
            if key in sugg:
                for c in ai_cols:
                    r[c] = str(sugg[key].get(c, "") or "")
                n += 1
        unknown = set(sugg) - {
            ((r["qid"],) if "qid" in r else (r["theme"], r["ticker"])) for r in rows
        }
        _write(path, fields, rows)
        report[fname] = {"merged": n, "unknown_keys": len(unknown)}
    return report


def agreement(golden_dir: Path = GOLDEN_DIR) -> dict:
    out = {}
    pairs: list[tuple[str, str]] = []
    for path in sorted(golden_dir.glob("review*.csv")):
        _, rows = _read(path)
        pairs += [(r["verdict"].strip().lower(), r.get("ai_verdict", "").strip().lower()) for r in rows
                  if r.get("verdict", "").strip() and r.get("ai_verdict", "").strip()]  # fmt: skip
    if pairs:
        keep = lambda v: "keep" if v in ("accept", "edit") else "reject"  # noqa: E731
        out["questions"] = {
            "n": len(pairs),
            "exact_agreement": round(sum(h == a for h, a in pairs) / len(pairs), 4),
            "keep_vs_reject_agreement": round(
                sum(keep(h) == keep(a) for h, a in pairs) / len(pairs), 4
            ),
            "confusion": dict(Counter(f"human={h} ai={a}" for h, a in pairs)),
        }
    tp = golden_dir / "global_themes.csv"
    if tp.exists():
        _, rows = _read(tp)
        tpairs = [(r["human_discussed"].strip().lower(), r.get("ai_discussed", "").strip().lower()) for r in rows
                  if r.get("human_discussed", "").strip() and r.get("ai_discussed", "").strip()]  # fmt: skip
        if tpairs:
            out["themes"] = {
                "n": len(tpairs),
                "agreement": round(sum(h == a for h, a in tpairs) / len(tpairs), 4),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["merge", "agreement"])
    ap.add_argument("files", nargs="*", type=Path)
    a = ap.parse_args(argv)
    if a.command == "merge":
        files = a.files or sorted((GOLDEN_DIR / "ai_review").glob("*.jsonl"))
        print(json.dumps(merge(files), indent=2))
    else:
        print(json.dumps(agreement(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
