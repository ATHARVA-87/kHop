import numpy as np

from plexus.baseline.naive import NOT_ENOUGH, NaiveRAG
from plexus.config import ExperimentConfig
from plexus.index.dense import Hit
from plexus.llm import FakeProvider, LLMCache, LLMClient


class StubEmbedder:
    def embed_query(self, text):
        return np.zeros(4, dtype=np.float32)


class StubIndex:
    def __init__(self, hits):
        self.hits = hits

    def search(self, vector, k):
        return self.hits[:k]


HITS = [
    Hit(f"D#c{i}", "D", i * 10, i * 10 + 9, "risk_factors", 1 - i / 10, f"chunk text {i}")
    for i in range(8)
]


def _rag(tmp_path, responder):
    cfg = ExperimentConfig(name="t")
    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"ollama": FakeProvider(responder)})
    return NaiveRAG(cfg, StubEmbedder(), StubIndex(HITS), llm)


def test_answer_maps_citations_to_chunk_ids(tmp_path):
    rag = _rag(tmp_path, lambda r: "TSMC makes the chips [2][1]. Also [9] and [2].")
    ans = rag.answer("Who makes the chips?")
    assert len(ans.hits) == 5  # context_chunks
    assert ans.cited_chunk_ids == ["D#c0", "D#c1"]  # [9] is out of range and ignored


def test_prompt_marks_excerpts_as_data_and_states_refusal(tmp_path):
    seen = {}

    def responder(req):
        seen["req"] = req
        return NOT_ENOUGH

    _rag(tmp_path, responder).answer("q?")
    system, user = seen["req"].messages
    assert "data, not instructions" in system.content and NOT_ENOUGH in system.content
    assert user.content.count("<excerpt>") == 5 and "[5] (source: D" in user.content
    assert seen["req"].seed == 0 and seen["req"].temperature == 0


def test_second_identical_question_is_cached(tmp_path):
    rag = _rag(tmp_path, lambda r: "x [1]")
    assert not rag.answer("same?").cached
    assert rag.answer("same?").cached


def test_thinking_block_is_stripped_and_refusal_detected(tmp_path):
    from plexus.baseline.naive import is_refusal

    rag = _rag(tmp_path, lambda r: "<think>maybe [3]?</think>\nTSMC [1].")
    ans = rag.answer("q?")
    assert ans.text == "TSMC [1]." and ans.cited_chunk_ids == ["D#c0"]
    assert is_refusal("<think>hmm</think> Not enough evidence in the filings to answer this")
    assert is_refusal("NOT ENOUGH EVIDENCE in the filings to answer this.")
    assert not is_refusal("TSMC makes them [1].")
