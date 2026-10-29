import csv
import json

from plexus.eval import paired_labels as pl


def _res(path, answers):
    rows = [
        {"qid": q, "question": f"Q{q}", "reference": "R", "answer": a} for q, a in answers.items()
    ]
    path.write_text(json.dumps({"per_query": rows}))


def test_mcnemar_exact():
    assert pl.mcnemar_exact(0, 0) == 1.0
    assert round(pl.mcnemar_exact(6, 0), 4) == 0.0312  # 2 * 1/64
    assert pl.mcnemar_exact(3, 3) == 1.0


def test_export_blinds_and_score_unswaps(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "SHEET", tmp_path / "sheet.csv")
    monkeypatch.setattr(pl, "KEY", tmp_path / "key.json")
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "human_verdict"])
        w.writerows([[f"q{i}", "partially_correct"] for i in range(6)])
    _res(tmp_path / "a.json", {f"q{i}": f"old {i}" for i in range(6)} | {"q5": "same"})
    _res(tmp_path / "b.json", {f"q{i}": f"new {i}" for i in range(6)} | {"q5": "same"})
    n, same = pl.export(tmp_path / "a.json", tmp_path / "b.json", labels)
    assert (n, same) == (5, 1)
    with open(pl.SHEET) as fh:
        rows = list(csv.DictReader(fh))
    assert "config" not in rows[0] and all(r["verdict_1"] == "" for r in rows)
    for r in rows:  # the labeller marks every "new" answer correct, every "old" one not
        r["verdict_1"] = "correct" if r["answer_1"].startswith("new") else "incorrect"
        r["verdict_2"] = "correct" if r["answer_2"].startswith("new") else "incorrect"
    with open(pl.SHEET, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = pl.score(labels)
    assert out["b"]["correct"] == 5 and out["a"]["correct"] == 0
    assert out["only_b_correct"] == 5 and out["identical_answers"] == 1
