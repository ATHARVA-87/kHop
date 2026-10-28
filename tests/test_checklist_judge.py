import pytest

from plexus.eval.checklist_judge import ChecklistJudge, binary_stats, wilson
from plexus.llm import FakeProvider, LLMCache, LLMClient


def test_wilson_and_binary_stats():
    assert wilson(38, 49) == pytest.approx([0.641, 0.870], abs=0.002)
    s = binary_stats([True] * 38 + [False] * 11, [True] * 37 + [False] + [True] * 8 + [False] * 3)
    assert s["tpr_catches_not_correct"] == round(3 / 11, 3)
    assert s["tnr_passes_correct"] == round(37 / 38, 3)


def test_checklist_verdict_is_correct_only_if_every_part_is_answered(tmp_path):
    def respond(req):
        text = req.messages[-1].content
        if "REQUIRED PARTS of a complete answer" in text:
            return '{"parts": ["SiTime and Apple", "Qualcomm and Apple"]}'
        if "Qualcomm says nothing" in text:
            return ('{"checks": [{"part": "SiTime and Apple", "status": "ANSWERED"},'
                    ' {"part": "Qualcomm and Apple", "status": "NOT_ANSWERED"}]}')  # fmt: skip
        return ('{"checks": [{"part": "SiTime and Apple", "status": "ANSWERED"},'
                ' {"part": "Qualcomm and Apple", "status": "ANSWERED"}]}')  # fmt: skip

    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"groq": FakeProvider(respond)})
    j = ChecklistJudge(llm, "groq", ["m"])
    assert j.judge("q", "ref", "Both answered.")["verdict"] == "correct"
    assert (
        j.judge("q", "ref", "SiTime sells to Apple; Qualcomm says nothing.")["verdict"]
        == "not_correct"
    )
