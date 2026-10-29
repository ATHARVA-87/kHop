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


def test_export_is_blind_random_and_agreement_joins_judge_verdicts(tmp_path):
    # review finding #2: agreement used to need a judge column nobody filled
    results = {"per_query": [
        {"qid": f"q{i}", "question": f"Q{i}?", "reference": "R", "answer": f"A{i}",
         "judge_verdict": "correct" if i % 2 else "incorrect"} for i in range(10)
    ] + [{"qid": "noans", "question": "?", "reference": "R"}]}  # fmt: skip
    out = tmp_path / "labels.csv"
    assert export_for_labelling(results, out, limit=4) == 4
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert "judge_verdict" not in rows[0]  # blind
    assert [r["qid"] for r in rows] != ["q0", "q1", "q2", "q3"]  # a sample, not the head
    for r in rows:
        r["human_verdict"] = "correct"
    rows[0]["human_verdict"] = ""  # unlabelled rows are ignored
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = agreement(out, results)
    assert res["n"] == 3
    expected = sum(int(r["qid"][1:]) % 2 == 1 for r in rows[1:]) / 3
    assert res["raw_agreement"] == round(expected, 4)


def test_agreement_without_any_pairs_is_a_clear_error(tmp_path):
    out = tmp_path / "labels.csv"
    export_for_labelling({"per_query": [{"qid": "q1", "answer": "a"}]}, out)
    with pytest.raises(ValueError, match="no labelled answers"):
        agreement(out, {"per_query": []})


def test_judge_skips_a_model_after_repeated_failures(tmp_path):
    from plexus.eval.judge import Judge
    from plexus.llm import FakeProvider, LLMCache, LLMClient, ProviderError

    calls = []

    def respond(req):
        calls.append(req.model)
        if req.model == "busy":
            raise ProviderError("503 overloaded")
        return '{"verdict": "correct", "rationale": ""}'

    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"gemini": FakeProvider(respond)})
    j = Judge(llm, "gemini", "busy", 0, ["lite"])
    for i in range(6):
        assert j.correctness(f"q{i}", "ref", "ans") == "correct"
    assert j.used["lite"] == 6
    assert calls.count("busy") <= Judge.MAX_FAILURES * 4  # stops trying after the breaker opens
