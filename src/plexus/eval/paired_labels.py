"""Blind paired labelling: is config B's answer better than config A's on the same questions?

    python -m plexus.eval.paired_labels export --a results/a6_router.json \
        --b results/a6_router_ctx10.json            # -> data/golden/paired_labels.csv
    python -m plexus.eval.paired_labels score        # after the human fills the sheet

No validated automatic judge exists (docs/failures.md), so answer changes are measured with
human labels. Each row shows the two answers as "answer_1" / "answer_2" in a random order (the
key is stored in a separate file the labeller does not open). Questions whose two answers are
identical are left out and keep their existing label from judge_labels.csv. Scoring: share of
fully correct answers per config, and an exact McNemar test on the questions where exactly
one of the two answers is correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from math import comb
from pathlib import Path

from plexus.settings import REPO_ROOT

GOLDEN = REPO_ROOT / "data" / "golden"
SHEET = GOLDEN / "paired_labels.csv"
KEY = GOLDEN / "paired_labels_key.json"  # which answer is which: do not open while labelling
VERDICTS = ("correct", "partially_correct", "incorrect")


def _answers(path: Path) -> dict[str, dict]:
    return {r["qid"]: r for r in json.loads(path.read_text())["per_query"] if r.get("answer")}


def _labels(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["qid"]: r["human_verdict"].strip().lower()
                for r in csv.DictReader(fh) if r["human_verdict"].strip()}  # fmt: skip


def export(a_path: Path, b_path: Path, labels: Path, seed: int = 0) -> tuple[int, int]:
    a, b, human = _answers(a_path), _answers(b_path), _labels(labels)
    rng = random.Random(seed)
    rows, key, same = [], {"a": str(a_path), "b": str(b_path), "swapped": {}, "identical": []}, 0
    for qid in sorted(human):
        ra, rb = a[qid], b[qid]
        if ra["answer"].strip() == rb["answer"].strip():
            key["identical"].append(qid)
            same += 1
            continue
        swap = rng.random() < 0.5
        first, second = (rb, ra) if swap else (ra, rb)
        key["swapped"][qid] = swap
        rows.append({"qid": qid, "question": ra["question"], "reference_answer": ra["reference"],
                     "answer_1": first["answer"], "verdict_1": "",
                     "answer_2": second["answer"], "verdict_2": ""})  # fmt: skip
    rng.shuffle(rows)  # question order also random
    with open(SHEET, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    KEY.write_text(json.dumps(key, indent=2) + "\n")
    return len(rows), same


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (binomial, p = 0.5)."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def score(labels: Path) -> dict:
    key = json.loads(KEY.read_text())
    human = _labels(labels)
    with open(SHEET, newline="", encoding="utf-8-sig") as fh:
        sheet = list(csv.DictReader(fh))
    va, vb = {}, {}
    for r in sheet:
        v1, v2 = r["verdict_1"].strip().lower(), r["verdict_2"].strip().lower()
        bad = [v for v in (v1, v2) if v not in VERDICTS]
        if bad:
            raise ValueError(f"{r['qid']}: verdict must be one of {VERDICTS}, got {bad}")
        swapped = key["swapped"][r["qid"]]
        va[r["qid"]], vb[r["qid"]] = (v2, v1) if swapped else (v1, v2)
    for qid in key["identical"]:  # same answer: the existing label applies to both
        va[qid] = vb[qid] = human[qid]
    qids = sorted(va)
    a_ok = [va[q] == "correct" for q in qids]
    b_ok = [vb[q] == "correct" for q in qids]
    b_only = sum(y and not x for x, y in zip(a_ok, b_ok, strict=True))
    a_only = sum(x and not y for x, y in zip(a_ok, b_ok, strict=True))
    from plexus.eval.checklist_judge import wilson

    n = len(qids)
    return {
        "n": n,
        "relabelled": len(sheet),
        "identical_answers": len(key["identical"]),
        "a": {"config": key["a"], "correct": sum(a_ok), "rate": round(sum(a_ok) / n, 3),
              "wilson_95": wilson(sum(a_ok), n)},
        "b": {"config": key["b"], "correct": sum(b_ok), "rate": round(sum(b_ok) / n, 3),
              "wilson_95": wilson(sum(b_ok), n)},
        "only_b_correct": b_only,
        "only_a_correct": a_only,
        "mcnemar_exact_p": round(mcnemar_exact(b_only, a_only), 4),
        "per_query": [{"qid": q, "a": va[q], "b": vb[q]} for q in qids],
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "score"))
    ap.add_argument("--a", type=Path, default=REPO_ROOT / "results/a6_router.json")
    ap.add_argument("--b", type=Path, default=REPO_ROOT / "results/a6_router_ctx10.json")
    ap.add_argument("--labels", type=Path, default=GOLDEN / "judge_labels.csv")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        n, same = export(a.a, a.b, a.labels)
        print(f"{SHEET}: {n} questions to label ({same} identical answers keep their label)")
        return 0
    out = score(a.labels)
    path = REPO_ROOT / "results" / f"paired_{a.b.stem}_vs_{a.a.stem}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_query"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
