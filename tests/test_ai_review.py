import csv
import json

from plexus.eval.ai_review import agreement, merge
from plexus.eval.golden import (
    Evidence,
    GoldenQuery,
    build_golden,
    read_review_csv,
    write_review_csv,
)
from tests.test_golden import _set_verdicts


def test_merge_writes_ai_columns_only_and_agreement(tmp_path):
    qs = [GoldenQuery(qid=f"q{i}", qtype="local", question=f"Q{i}?", answer="A",
                      evidence=[Evidence(doc_id="D", quote="x")]) for i in range(3)]  # fmt: skip
    write_review_csv(qs, tmp_path / "review.csv")
    _set_verdicts(
        tmp_path / "review.csv", {"q0": {"verdict": "accept"}}
    )  # human already reviewed q0
    sugg = tmp_path / "a.jsonl"
    sugg.write_text("\n".join(json.dumps(s) for s in [
        {"file": "review.csv", "qid": "q0", "ai_verdict": "reject", "ai_notes": "vague", "verdict": "HACK"},
        {"file": "review.csv", "qid": "q1", "ai_verdict": "edit", "ai_corrected_answer": "Better."},
        {"file": "review.csv", "qid": "nope", "ai_verdict": "accept"},
    ]))  # fmt: skip
    rep = merge([sugg], tmp_path)
    assert rep["review.csv"] == {"merged": 2, "unknown_keys": 1}
    rows = {r["qid"]: r for r in read_review_csv(tmp_path / "review.csv")}
    assert (
        rows["q0"]["verdict"] == "accept" and rows["q0"]["ai_verdict"] == "reject"
    )  # human column untouched
    assert rows["q1"]["verdict"] == "" and rows["q1"]["ai_corrected_answer"] == "Better."
    # AI suggestions alone never make a question verified
    stats = build_golden(tmp_path / "review.csv", tmp_path / "g.jsonl", tmp_path / "s.json")
    assert stats["verdicts"]["unreviewed"] == 2
    _set_verdicts(tmp_path / "review.csv", {"q1": {"verdict": "edit"}})
    ag = agreement(tmp_path)["questions"]
    assert ag["n"] == 2 and ag["exact_agreement"] == 0.5 and ag["keep_vs_reject_agreement"] == 0.5


def test_merge_theme_rows(tmp_path):
    with open(tmp_path / "global_themes.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["theme", "keyword", "ticker", "llm_discussed", "human_discussed"]
        )
        w.writeheader()
        w.writerow(
            {
                "theme": "China",
                "keyword": "China",
                "ticker": "NVDA",
                "llm_discussed": "yes",
                "human_discussed": "",
            }
        )
    s = tmp_path / "t.jsonl"
    s.write_text(
        json.dumps(
            {"file": "global_themes.csv", "theme": "China", "ticker": "NVDA", "ai_discussed": "no"}
        )
    )
    assert merge([s], tmp_path)["global_themes.csv"]["merged"] == 1
    with open(tmp_path / "global_themes.csv", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["ai_discussed"] == "no" and row["human_discussed"] == ""
