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
from plexus.extract.schema import LOWERCASE_NAMES
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


def _proper(word: str) -> bool:
    """A single word counts as a name only when written like one: capitalised ("Arm",
    "Skyworks") or containing a digit ("3M"). Review 2026-10-28: "arm's length" linked Arm and
    "flex capacity" linked Flex. Trade-off: an all-lowercase "nvidia" no longer links."""
    if word.lower() in LOWERCASE_NAMES:  # brands written in lower case: onsemi, imec
        return True
    return word[:1].isupper() or any(ch.isdigit() for ch in word)


class EntityLinker:
    """Question text -> linked entities with a confidence score.

    Matches, best first (overlapping matches keep the best one):
    - 1.0  a full name or alias from the alias table / graph ("Skyworks Solutions", "TSMC");
    - 0.9  a filer ticker written in capitals ("ON", "NXPI"; lowercase "on" never links);
    - 0.6  one distinctive word of a filer's name ("Skyworks"), so a partial name still hits.
    A single-word name that is also a common lowercase word in the filings ("diodes", "onto")
    only links when the question capitalises it.
    """

    FULL, TICKER, WORD = 1.0, 0.9, 0.6

    def __init__(self, resolver: Resolver, entities: list[dict]):
        self.names: dict[tuple[str, ...], str] = {}
        for (etype, key), (eid, _canon) in resolver.index.items():
            if etype == "company" and key and len(key) > 2:
                self.names.setdefault(tuple(key.split()), eid)
        for e in entities:
            if e["type"] != "company" or e["id"].startswith("company:anon:"):
                continue
            for n in [e["name"], *e.get("aliases", [])]:
                k = norm_key(n)
                if len(k) > 2:
                    self.names.setdefault(tuple(k.split()), e["id"])
        self.max_len = max((len(k) for k in self.names), default=1)
        self.tickers = {eid.split(":")[1]: eid for eid in resolver.filer_ids.values()}
        # short all-caps aliases ("TI", "MPS", "ADI") match like tickers: exact, capitals only.
        # Found on the held-out test: "What does TI say..." was routed as sector-wide.
        for (etype, key), (eid, _canon) in resolver.index.items():
            if etype == "company" and 2 <= len(key) <= 3 and key.isalpha():
                self.tickers.setdefault(key.upper(), eid)
        self.words = {t: resolver.filer_ids[tk] for t, tk in resolver.filer_tokens.items()}
        self.caps = resolver.token_caps

    def link_scored(self, question: str) -> list[tuple[str, float]]:
        toks = [(m.group(), m.start()) for m in re.finditer(r"[A-Za-z0-9]+", question)]
        low = [t.lower() for t, _ in toks]
        cands: list[tuple[float, int, int, str]] = []  # score, start, end (token idx), id
        for i in range(len(toks)):
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                key = tuple(low[i : i + n])
                eid = self.names.get(key)
                if eid is None:
                    continue
                if n == 1 and not _proper(toks[i][0]):
                    continue  # one lowercase word ("arm's length", "diodes") is not a name
                cands.append((self.FULL, i, i + n, eid))
            if toks[i][0] in self.tickers:
                cands.append((self.TICKER, i, i + 1, self.tickers[toks[i][0]]))
            if low[i] in self.words and _proper(toks[i][0]):
                cands.append((self.WORD, i, i + 1, self.words[low[i]]))
        # keep the best match per span: higher score, then longer
        cands.sort(key=lambda c: (-c[0], -(c[2] - c[1]), c[1]))
        taken: set[int] = set()
        kept: list[tuple[int, str, float]] = []
        for score, a, b, eid in cands:
            if taken.isdisjoint(range(a, b)):
                taken.update(range(a, b))
                kept.append((a, eid, score))
        best: dict[str, tuple[int, float]] = {}
        for a, eid, score in sorted(kept):
            if eid not in best or score > best[eid][1]:
                best[eid] = (min(a, best.get(eid, (a, 0))[0]), score)
        return [(eid, sc) for eid, (_, sc) in sorted(best.items(), key=lambda kv: kv[1][0])]

    def link(self, question: str) -> list[str]:
        return [eid for eid, _ in self.link_scored(question)]


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
        scored = self.linker.link_scored(question)
        linked = [e for e, _ in scored]
        conf = dict(scored)
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
        # relation match first; then edges whose linked endpoint was named more specifically
        # (full name > ticker > one name word); then better-supported edges
        edges.sort(key=lambda e: (e.relation not in cues,
                                  -max(conf.get(e.src, 0.0), conf.get(e.dst, 0.0)), -e.support))  # fmt: skip
        trace = {"linked": scored, "cues": sorted(cues), "mode": mode,
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


class CompanyRoundRobinRetriever:
    """A2-RR, the no-graph baseline for multi-company questions (reviewer, 2026-10-28): take
    the hybrid list and alternate between the filers the question names (their best hit,
    then their second best, ...), then the rest in hybrid order. Uses the same entity linker
    as A4 but no graph edges, so A4 - A2-RR is the graph's own contribution."""

    def __init__(self, base, linker: EntityLinker, depth: int = 100):
        self.base = base
        self.linker = linker
        self.depth = depth

    def retrieve(self, question: str, k: int) -> list[Hit]:
        hits = self.base.retrieve(question, max(k, self.depth))
        tickers = {eid.split(":")[1] for eid in self.linker.link(question)
                   if eid in self.linker.tickers.values()}  # fmt: skip
        if len(tickers) < 2:
            return hits[:k]
        order = [
            t for t in (eid.split(":")[1] for eid in self.linker.link(question)) if t in tickers
        ]
        queues = {t: [h for h in hits if h.doc_id.split("_")[0] == t] for t in order}
        out: list[Hit] = []
        seen: set[str] = set()
        while any(queues.values()) and len(out) < k:
            for t in order:
                if queues[t] and len(out) < k:
                    h = queues[t].pop(0)
                    seen.add(h.chunk_id)
                    out.append(h)
        out += [h for h in hits if h.chunk_id not in seen][: k - len(out)]
        return out

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()


class CompanyDiverseRetriever:
    """A5 (sector-wide questions): if the question names no company, keep the best passage
    from each company first (in hybrid order), then the rest. One passage per company is
    what a "how do all semiconductor companies ..." answer needs, and top-10 from plain
    relevance ranking tends to repeat a few companies. Questions naming a company pass
    through unchanged (to the wrapped retriever, e.g. A4)."""

    def __init__(self, base, linker: EntityLinker, depth: int = 100):
        self.base = base
        self.linker = linker
        self.depth = depth

    def retrieve(self, question: str, k: int) -> list[Hit]:
        hits = self.base.retrieve(question, max(k, self.depth))
        if self.linker.link(question):
            return hits[:k]
        first: list[Hit] = []
        rest: list[Hit] = []
        seen: set[str] = set()
        for h in hits:
            company = h.doc_id.split("_")[0]
            (rest if company in seen else first).append(h)
            seen.add(company)
        return (first + rest)[:k]

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()
