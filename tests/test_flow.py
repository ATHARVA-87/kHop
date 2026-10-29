from plexus.agents.flow import ask, build_flow
from plexus.baseline.naive import NaiveRAG
from plexus.retrieve.router import FixedRouter, RoutedRetriever
from tests.test_naive import HITS, _rag


class ListRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, question, k):
        return self.hits[:k]


def test_flow_gives_the_same_answer_as_naive_rag(tmp_path):
    rag = _rag(tmp_path, lambda r: "TSMC makes the chips [2][1].")
    direct = rag.answer("Who makes the chips?")
    ans, trace = ask(build_flow(rag), "Who makes the chips?")
    assert ans.text == direct.text and ans.cited_chunk_ids == direct.cited_chunk_ids
    assert ans.cached  # the same request hit the LLM cache: identical prompt
    assert [t["step"] for t in trace] == ["route", "retrieve", "answer"]
    assert trace[0]["route"] == "fixed" and trace[2]["cited"] == ["D#c0", "D#c1"]


def test_flow_uses_the_routed_retriever(tmp_path):
    base = _rag(tmp_path, lambda r: "ok [1]")
    routed = RoutedRetriever(FixedRouter("global"), {
        "local": ListRetriever(HITS[:1]), "multi_hop": ListRetriever(HITS[:2]),
        "global": ListRetriever(HITS[3:]),
    })  # fmt: skip
    rag = NaiveRAG(base.cfg, routed, base.llm)
    ans, trace = ask(build_flow(rag), "Which companies?")
    assert trace[0]["route"] == "global" and trace[1]["n_hits"] == 5
    assert ans.hits[0].chunk_id == "D#c3"
