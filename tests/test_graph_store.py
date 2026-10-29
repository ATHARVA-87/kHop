import json

import pytest

pytest.importorskip("ladybug")  # optional `graph` extra; CI skips

from plexus.index.graph import LadybugGraph


def _ev(doc, q):
    return [{"doc_id": doc, "passage_id": f"{doc}#p1", "start": 0, "end": 5, "quote": q}]


@pytest.fixture
def graph(tmp_path):
    ents = [
        {"id": "company:SLAB", "type": "company", "name": "Silicon Labs", "ticker": "SLAB", "mentions": 3},
        {"id": "company:CRUS", "type": "company", "name": "Cirrus Logic", "ticker": "CRUS", "mentions": 3},
        {"id": "company:tsmc", "type": "company", "name": "TSMC", "ticker": None, "mentions": 5},
        {"id": "company:smic", "type": "company", "name": "SMIC", "ticker": None, "mentions": 1},
        {"id": "geography:china", "type": "geography", "name": "China", "ticker": None, "mentions": 2},
    ]  # fmt: skip
    edges = [
        {"src": "company:SLAB", "relation": "manufactured_by", "dst": "company:tsmc", "support": 2,
         "qualifiers": ["wafers"], "evidence": _ev("SLAB_10K", "TSMC makes our wafers")},
        {"src": "company:CRUS", "relation": "manufactured_by", "dst": "company:tsmc", "support": 1,
         "qualifiers": [], "evidence": _ev("CRUS_10K", "we use TSMC")},
        {"src": "company:SLAB", "relation": "manufactured_by", "dst": "company:smic", "support": 1,
         "qualifiers": [], "evidence": _ev("SLAB_10K", "and SMIC")},
        {"src": "company:SLAB", "relation": "operates_in", "dst": "geography:china", "support": 1,
         "qualifiers": [], "evidence": _ev("SLAB_10K", "sales in China")},
        {"src": "company:CRUS", "relation": "operates_in", "dst": "geography:china", "support": 1,
         "qualifiers": [], "evidence": _ev("CRUS_10K", "China sales")},
    ]  # fmt: skip
    (tmp_path / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in ents) + "\n")
    (tmp_path / "edges.jsonl").write_text("\n".join(json.dumps(e) for e in edges) + "\n")
    info = LadybugGraph.build(tmp_path, tmp_path / "g.lbug")
    assert info == {"nodes": 5, "edges": 5, "db": str(tmp_path / "g.lbug")}
    g = LadybugGraph(tmp_path / "g.lbug")
    yield g
    g.close()


def test_edges_of_filters_by_relation_and_keeps_evidence(graph):
    es = graph.edges_of("company:SLAB", {"manufactured_by"})
    assert {e.dst for e in es} == {"company:tsmc", "company:smic"}
    tsmc = next(e for e in es if e.dst == "company:tsmc")
    assert tsmc.support == 2 and tsmc.qualifiers == ["wafers"]
    assert tsmc.evidence[0]["quote"] == "TSMC makes our wafers"


def test_common_neighbours_answers_bridge_questions(graph):
    both = graph.common_neighbours("company:SLAB", "company:CRUS")
    assert [n for n, _ in both] == ["company:tsmc", "geography:china"]  # by total support
    only_companies = graph.common_neighbours("company:SLAB", "company:CRUS", types={"company"})
    assert [n for n, _ in only_companies] == ["company:tsmc"]
    assert {e.src for e in only_companies[0][1]} == {"company:SLAB", "company:CRUS"}


def test_edges_between_and_entity_lookup(graph):
    assert [e.relation for e in graph.edges_between("company:tsmc", "company:CRUS")] == [
        "manufactured_by"
    ]
    assert graph.entity("company:SLAB")["ticker"] == "SLAB"
    assert graph.entity("company:nope") is None
