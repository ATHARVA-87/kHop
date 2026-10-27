from plexus.extract.resolve import Resolver
from plexus.index.dense import Hit
from plexus.index.graph import Edge
from plexus.retrieve.graph_search import (
    ChunkLocator,
    EntityLinker,
    GraphHybridRetriever,
    GraphRetriever,
    relation_cues,
)

FILERS = {"SLAB": {"company": "SILICON LABORATORIES INC."}, "CRUS": {"company": "CIRRUS LOGIC, INC."},
          "ON": {"company": "ON SEMICONDUCTOR CORP"}}  # fmt: skip
RES = Resolver(FILERS, {"company": {"TSMC": ["Taiwan Semiconductor Manufacturing Company"]}})
ENTS = [{"id": "company:tsmc", "type": "company", "name": "TSMC", "aliases": []}]
CHUNKS = [
    {"chunk_id": "SLAB#s1", "doc_id": "SLAB_10K", "start": 0, "end": 100, "section": "business", "text": "a"},
    {"chunk_id": "SLAB#s2", "doc_id": "SLAB_10K", "start": 100, "end": 200, "section": "business", "text": "b"},
    {"chunk_id": "CRUS#s1", "doc_id": "CRUS_10K", "start": 0, "end": 80, "section": "business", "text": "c"},
]  # fmt: skip


def _ev(doc, start):
    return [{"doc_id": doc, "passage_id": "p", "start": start, "end": start + 5, "quote": "q"}]


class FakeGraph:
    def __init__(self):
        self.e = [Edge("company:SLAB", "manufactured_by", "company:tsmc", 2, [], _ev("SLAB_10K", 120)),
                  Edge("company:CRUS", "manufactured_by", "company:tsmc", 1, [], _ev("CRUS_10K", 10))]  # fmt: skip

    def edges_of(self, x, relations=None):
        return [
            e
            for e in self.e
            if x in (e.src, e.dst) and (relations is None or e.relation in relations)
        ]

    def edges_between(self, a, b):
        return [e for e in self.e if {e.src, e.dst} == {a, b}]

    def common_neighbours(self, a, b, types=None):
        return [("company:tsmc", self.e)]


def test_linker_finds_names_aliases_and_capital_tickers_only():
    link = EntityLinker(RES, ENTS)
    assert link.link("Which foundry makes wafers for both Silicon Labs and Cirrus Logic?") == [
        "company:SLAB", "company:CRUS"]  # fmt: skip
    assert link.link("What do onsemi and Silicon Laboratories rely on TSMC for?") == [
        "company:ON", "company:SLAB", "company:tsmc"]  # fmt: skip
    assert "company:ON" not in link.link("What depends on demand?")  # lowercase 'on' is a word


def test_relation_cues():
    assert "manufactured_by" in relation_cues("Which foundry makes their wafers?")
    assert relation_cues("Who are Lam Research's competitors?") == {"competes_with"}


def test_chunk_locator():
    loc = ChunkLocator(CHUNKS)
    assert loc.find("SLAB_10K", 150)["chunk_id"] == "SLAB#s2"
    assert loc.find("SLAB_10K", 500) is None and loc.find("X", 1) is None


def test_bridge_question_returns_both_evidence_chunks_and_fuses():
    g = GraphRetriever(FakeGraph(), EntityLinker(RES, ENTS), CHUNKS)
    hits = g.retrieve("Which foundry makes wafers for both Silicon Labs and Cirrus Logic?", 10)
    assert [h.chunk_id for h in hits] == ["SLAB#s2", "CRUS#s1"]
    assert g.last_trace["mode"] == "bridge"

    class Base:
        def retrieve(self, q, k):
            return [Hit("X#1", "X", 0, 1, "s", 1.0, "x")]

    fused = GraphHybridRetriever(g, Base()).retrieve(
        "Which foundry serves both Silicon Labs and Cirrus Logic?", 3
    )
    assert {h.chunk_id for h in fused} == {"SLAB#s2", "CRUS#s1", "X#1"}
    # no linked company -> plain base list
    assert [
        h.chunk_id for h in GraphHybridRetriever(g, Base()).retrieve("What is inflation?", 3)
    ] == ["X#1"]
