import csv

import pytest

from plexus.eval.golden import (
    Evidence,
    GoldenQuery,
    build_golden,
    load_golden,
    read_review_csv,
    write_review_csv,
)


def _q(qid, qtype="local", n_ev=1):
    return GoldenQuery(
        qid=qid,
        qtype=qtype,
        question=f"Question {qid}?",
        answer=f"Answer {qid}.",
        evidence=[Evidence(doc_id=f"D{i}", quote=f"quote {i}") for i in range(n_ev)],
        sub_questions=["a?", "b?"] if qtype == "multi_hop" else [],
        source="llm:test",
    )


def _set_verdicts(path, verdicts: dict[str, dict]):
    rows = read_review_csv(path)
    for r in rows:
        r.update(verdicts.get(r["qid"], {}))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_only_accepted_or_edited_rows_are_scored(tmp_path):
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1"), _q("q2", "multi_hop", 2), _q("q3"), _q("q4")], review)
    _set_verdicts(
        review,
        {
            "q1": {"verdict": "accept"},
            "q2": {"verdict": "edit", "corrected_answer": "Fixed answer."},
            "q3": {"verdict": "reject"},
            # q4 left unreviewed
        },
    )
    stats = build_golden(review, tmp_path / "golden.jsonl", tmp_path / "stats.json")
    gold = load_golden(tmp_path / "golden.jsonl")
    assert [q.qid for q in gold] == ["q1", "q2"]
    assert gold[1].answer == "Fixed answer." and len(gold[1].evidence) == 2
    assert gold[1].sub_questions == ["a?", "b?"]
    assert stats["reviewed"] == 3 and stats["verdicts"]["unreviewed"] == 1
    assert stats["label_noise_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_load_golden_refuses_when_nothing_verified(tmp_path):
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1")], review)
    build_golden(review, tmp_path / "golden.jsonl", tmp_path / "stats.json")
    with pytest.raises(ValueError, match="no verified"):
        load_golden(tmp_path / "golden.jsonl")


def test_regenerating_candidates_never_clobbers_reviewed_rows(tmp_path):
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1"), _q("q2")], review)
    _set_verdicts(review, {"q1": {"verdict": "accept"}})
    with pytest.raises(ValueError, match="refusing"):
        write_review_csv([_q("q1")], review)
    write_review_csv([_q("q9")], review)  # new candidates are appended; q1 kept, q2 replaced
    assert [r["qid"] for r in read_review_csv(review)] == ["q1", "q9"]


def test_bad_verdict_and_mismatched_evidence_are_errors(tmp_path):
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1"), _q("q2")], review)
    _set_verdicts(review, {"q1": {"verdict": "maybe"}})
    with pytest.raises(ValueError, match="verdict"):
        build_golden(review, tmp_path / "g.jsonl", tmp_path / "s.json")
    _set_verdicts(review, {"q1": {"verdict": "edit", "corrected_evidence_quotes": "a ||| b"}})
    with pytest.raises(ValueError, match="quotes"):
        build_golden(review, tmp_path / "g.jsonl", tmp_path / "s.json")


def test_rejected_or_unreviewed_rows_with_broken_fields_do_not_abort_build(tmp_path):
    # review finding #6
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1"), _q("q2"), _q("q3")], review)
    _set_verdicts(
        review,
        {
            "q1": {"verdict": "accept"},
            "q2": {"verdict": "reject", "answer": "", "corrected_evidence_quotes": "a ||| b"},
            "q3": {"qtype": "nonsense"},  # unreviewed
        },
    )
    stats = build_golden(review, tmp_path / "g.jsonl", tmp_path / "s.json")
    assert stats["verdicts"] == {"accept": 1, "reject": 1, "unreviewed": 1}
    assert [q.qid for q in load_golden(tmp_path / "g.jsonl")] == ["q1"]


def test_excel_bom_csv_is_read(tmp_path):
    # review finding #7: Excel saves UTF-8 with a byte-order mark
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1")], review)
    _set_verdicts(review, {"q1": {"verdict": "Accept", "corrected_answer": "Uses TSMC’s N4 node."}})
    raw = review.read_text(encoding="utf-8")
    review.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
    build_golden(review, tmp_path / "g.jsonl", tmp_path / "s.json")
    assert load_golden(tmp_path / "g.jsonl")[0].answer == "Uses TSMC’s N4 node."


def test_skip_reviewed_keeps_existing_review(tmp_path):
    review = tmp_path / "review.csv"
    write_review_csv([_q("q1"), _q("q2")], review)
    _set_verdicts(review, {"q1": {"verdict": "accept", "reviewer_notes": "checked"}})
    write_review_csv([_q("q1"), _q("q3")], review, skip_reviewed=True)
    rows = read_review_csv(review)
    assert [r["qid"] for r in rows] == ["q1", "q3"] and rows[0]["reviewer_notes"] == "checked"
