"""A4: graph traversal retrieval, fused with A2 hybrid retrieval.

For a question, link the companies it names to graph nodes, then pick edges by its shape:
- two filers + a third named entity ("What do AMD and SiTime each rely on TSMC for?"):
  the edges between each filer and that entity;
- two filers, nothing else ("Which foundry makes wafers for both Silicon Labs and Cirrus
  Logic?"): their common neighbours (a bridge), filtered by relation words in the question;
- one company + relation words ("Who are Lam Research's competitors?"): its edges of that kind.
Each edge carries the exact spans it was extracted from; those spans are mapped to the A1
chunks that contain them and fused with the hybrid list by reciprocal rank fusion. A question
the graph cannot link (no named company) simply gets the hybrid list.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict
from pathlib import Path

from plexus.extract.resolve import Resolver, norm_key
from plexus.index.dense import Hit
from plexus.index.graph import Edge, GraphStore
from plexus.retrieve.fusion import rrf

# question words -> relations they ask about
RELATION_CUES: list[tuple[re.Pattern, set[str]]] = [
    (re.compile(r"\bfoundr|\bfab\b|\bwafer|manufactur|assembl|packag|\bmake[s]?\b", re.I),
     {"manufactured_by", "supplied_by"}),
    (re.compile(r"\bsuppl|\bsource|\bvendor|\brel(y|ies|iance) on\b", re.I),
     {"supplied_by", "manufactured_by", "licenses_from"}),
    (re.compile(r"\bcustomer|\bbuy|\bsell|\bsales to\b|\brevenue from\b", re.I), {"sells_to"}),
    (re.compile(r"\bcompet|\brival", re.I), {"competes_with"}),
    (re.compile(r"\bpartner|\bcollaborat|\balliance|\bjoint", re.I), {"partners_with"}),
    (re.compile(r"\blicen[cs]", re.I), {"licenses_from"}),
    (re.compile(r"\bacquir|\bbought|\bpurchas", re.I), {"acquired", "sold_business_to"}),
    (re.compile(r"\bsold\b|\bsale of\b|\bdivest", re.I), {"sold_business_to", "acquired"}),
    (re.compile(r"\bentity list|\bexport|\brestrict|\bsanction", re.I), {"exposed_to", "subject_to"}),
]  # fmt: skip
COMPANY_RELATIONS = {"sells_to", "supplied_by", "manufactured_by", "competes_with",
                     "partners_with", "licenses_from", "acquired", "sold_business_to", "exposed_to"}  # fmt: skip


def relation_cues(question: str) -> set[str]:
    out: set[str] = set()
    for rx, rels in RELATION_CUES:
        if rx.search(question):
            out |= rels
    return out


class EntityLinker:
    """Surface names -> entity ids, longest match first. Tickers count only when written in
    capitals ("ON" yes, "on" no); other names match case-insensitively on word boundaries."""

    def __init__(self, resolver: Resolver, entities: list[dict]):
        self.names: dict[str, str] = {}
        for (etype, key), (eid, _canon) in resolver.index.items():
            if etype == "company" and key and len(key) > 2:
                self.names.setdefault(key, eid)
        for e in entities:
            if e["type"] != "company" or e["id"].startswith("company:anon:"):
                continue
            for n in [e["name"], *e.get("aliases", [])]:
                k = norm_key(n)
                if len(k) > 2:
                    self.names.setdefault(k, e["id"])
        self.tickers = {eid.split(":")[1]: eid for eid in resolver.filer_ids.values()}
        keys = sorted(self.names, key=len, reverse=True)
        self._rx = (
            re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b") if keys else None
        )

    def link(self, question: str) -> list[str]:
        found: list[tuple[int, str]] = []
        text = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
        if self._rx:
            for m in self._rx.finditer(text):
                found.append((m.start(), self.names[m.group(1)]))
        for m in re.finditer(r"\b([A-Z]{2,5})\b", question):
            if m.group(1) in self.tickers:
                found.append((m.start() * 10_000, self.tickers[m.group(1)]))  # after names
        out: list[str] = []
        for _, eid in sorted(found):
            if eid not in out:
                out.append(eid)
        return out


class ChunkLocator:
    """Maps (doc_id, char offset) to the A1 chunk containing it."""

    def __init__(self, chunks: list[dict]):
        self.by_doc: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            self.by_doc[c["doc_id"]].append(c)
        self.starts = {}
        for d, cs in self.by_doc.items():
            cs.sort(key=lambda c: c["start"])
            self.starts[d] = [c["start"] for c in cs]

    def find(self, doc_id: str, offset: int) -> dict | None:
        cs = self.by_doc.get(doc_id)
        if not cs:
            return None
        i = bisect.bisect_right(self.starts[doc_id], offset) - 1
        if i >= 0 and cs[i]["start"] <= offset < cs[i]["end"]:
            return cs[i]
        return None


def _hit(c: dict, score: float) -> Hit:
    fields = {f: c[f] for f in ("chunk_id", "doc_id", "start", "end", "section", "text")}
    extra = {f: c[f] for f in ("parent_id", "parent_start", "parent_end", "header") if f in c}
    return Hit(score=score, **fields, **extra)


class GraphRetriever:
    def __init__(self, graph: GraphStore, linker: EntityLinker, chunks: list[dict],
                 max_edges: int = 12):  # fmt: skip
        self.graph = graph
        self.linker = linker
        self.locator = ChunkLocator(chunks)
        self.max_edges = max_edges
        self.last_trace: dict = {}

    def select_edges(self, question: str) -> tuple[list[Edge], dict]:
        linked = self.linker.link(question)
        cues = relation_cues(question)
        filers = [e for e in linked if e in self.linker.tickers.values()]
        others = [e for e in linked if e not in filers]
        edges: list[Edge] = []
        mode = "none"
        if len(filers) >= 2 and others:
            mode = "filers_to_entity"
            for f in filers:
                for o in others:
                    edges += self.graph.edges_between(f, o)
        elif len(filers) >= 2:
            mode = "bridge"
            a, b = filers[:2]
            edges += self.graph.edges_between(a, b)
            for _n, es in self.graph.common_neighbours(a, b, types={"company"}):
                if cues and not any(e.relation in cues for e in es):
                    continue
                edges += es
        elif len(linked) >= 2:
            # one filer + another named company the graph knows ("Apple and Qualcomm")
            mode = "pair"
            edges += self.graph.edges_between(linked[0], linked[1])
            rels = cues & COMPANY_RELATIONS
            if rels:
                for e in linked[:2]:
                    edges += self.graph.edges_of(e, rels)
        else:
            # a single named company: the graph returned generic "competitors"/"suppliers"
            # passages that displaced the specific one asked for (dev: local recall -0.20 on
            # the 10 such questions), so single-entity questions use the hybrid list only
            mode = "single_skipped" if linked else "none"
        # relation-matching edges first, then better-supported ones
        edges.sort(key=lambda e: (e.relation not in cues, -e.support))
        trace = {"linked": linked, "cues": sorted(cues), "mode": mode,
                 "edges": [(e.src, e.relation, e.dst, e.support) for e in edges[: self.max_edges]]}  # fmt: skip
        return edges[: self.max_edges], trace

    def retrieve(self, question: str, k: int) -> list[Hit]:
        edges, self.last_trace = self.select_edges(question)
        out: list[Hit] = []
        seen: set[str] = set()
        for rank, e in enumerate(edges):
            for ev in e.evidence:
                c = self.locator.find(ev["doc_id"], ev["start"])
                if c and c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    out.append(_hit(c, 1.0 / (rank + 1)))
        return out[:k]


class GraphHybridRetriever:
    """A4 = RRF(graph hits, hybrid hits). With no graph hits it equals the hybrid list."""

    def __init__(self, graph_retriever: GraphRetriever, base, rrf_k: int = 60):
        self.graph_retriever = graph_retriever
        self.base = base
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[Hit]:
        g = self.graph_retriever.retrieve(question, k)
        b = self.base.retrieve(question, max(k, 100))
        if not g:
            return b[:k]
        return rrf([g, b], self.rrf_k)[:k]

    def close(self) -> None:
        for r in (self.base, getattr(self.graph_retriever, "graph", None)):
            if hasattr(r, "close"):
                r.close()


def load_entities(graph_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (graph_dir / "entities.jsonl").read_text().splitlines()]
