import csv
import json

import pytest

from plexus.eval.judge import (
    Judge,
    agreement,
    cohen_kappa,
    export_for_labelling,
)
from plexus.llm import FakeProvider, LLMCache, LLMClient


def test_cohen_kappa_known_values():
    # Classic 2x2 example: 50 items, p_o = 0.7, p_e = 0.5 -> kappa = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    assert cohen_kappa(["x", "y"], ["y", "x"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa(["x"], [])


def _judge(tmp_path, reply):
    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": FakeProvider(lambda r: reply)})
    return Judge(llm, "fake", "m")


def test_faithfulness_counts_supported_claims(tmp_path):
    reply = json.dumps({"claims": [
        {"claim": "TSMC makes the GPUs", "cited": [1], "supported": True},
        {"claim": "Revenue tripled", "cited": [], "supported": False},
        {"claim": "Export controls apply", "cited": [2], "supported": True},
    ]})  # fmt: skip
    assert _judge(tmp_path, reply).faithfulness("q", "a", ["e1", "e2"]) == pytest.approx(2 / 3)


def test_refusal_has_no_faithfulness_score(tmp_path):
    assert _judge(tmp_path, '{"claims": []}').faithfulness("q", "Not enough evidence", []) is None


def test_correctness_verdict(tmp_path):
    j = _judge(tmp_path, '{"verdict": "partially_correct", "rationale": "misses China"}')
    assert j.correctness("q", "ref", "ans") == "partially_correct"


def test_export_is_blind_and_agreement_reads_labels(tmp_path):
    results = {"per_query": [
        {"qid": "q1", "question": "Q1?", "reference": "R1", "answer": "A1"},
        {"qid": "q2", "question": "Q2?", "reference": "R2", "answer": "A2"},
        {"qid": "q3", "question": "Q3?", "reference": "R3"},  # no answer -> skipped
    ]}  # fmt: skip
    out = tmp_path / "labels.csv"
    assert export_for_labelling(results, out) == 2
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["judge_verdict"] == "" and r["human_verdict"] == "" for r in rows)

    rows[0].update(human_verdict="correct", judge_verdict="correct")
    rows[1].update(human_verdict="incorrect", judge_verdict="correct")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = agreement(out)
    assert res["n"] == 2 and res["raw_agreement"] == 0.5
    assert res["confusion"] == {
        "human=correct judge=correct": 1,
        "human=incorrect judge=correct": 1,
    }
