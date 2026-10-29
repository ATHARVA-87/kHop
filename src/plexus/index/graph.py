"""Graph store for the resolved entity graph (Phase 3, ADR 0006).

    python -m plexus.index.graph build            # entities.jsonl + edges.jsonl -> LadybugDB

`GraphStore` is the small interface the retrievers use. `LadybugGraph` implements it on
LadybugDB (the maintained Kuzu fork, Cypher). Swapping engines means re-implementing these
few methods, nothing else.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from plexus.settings import Settings


@dataclass
class Edge:
    src: str
    relation: str
    dst: str
    support: int
    qualifiers: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # doc_id, passage_id, start, end, quote


class GraphStore(Protocol):
    def edges_of(self, entity_id: str, relations: set[str] | None = None) -> list[Edge]: ...
    def edges_between(self, a: str, b: str) -> list[Edge]: ...
    def common_neighbours(
        self, a: str, b: str, types: set[str] | None = None
    ) -> list[tuple[str, list[Edge]]]: ...


class LadybugGraph:
    def __init__(self, path: Path, read_only: bool = True):
        import ladybug

        self.db = ladybug.Database(str(path), read_only=read_only)
        self.conn = ladybug.Connection(self.db)

    # ---- building ----
    @classmethod
    def build(cls, graph_dir: Path, db_path: Path) -> dict:
        if db_path.exists():
            shutil.rmtree(db_path) if db_path.is_dir() else db_path.unlink()
        g = cls(db_path, read_only=False)
        c = g.conn
        c.execute("CREATE NODE TABLE Entity(id STRING, name STRING, type STRING, ticker STRING, "
                  "mentions INT64, PRIMARY KEY(id))")  # fmt: skip
        c.execute("CREATE REL TABLE Rel(FROM Entity TO Entity, relation STRING, support INT64, "
                  "qualifiers STRING, evidence STRING)")  # fmt: skip
        n_nodes = n_edges = 0
        for line in (graph_dir / "entities.jsonl").read_text().splitlines():
            e = json.loads(line)
            c.execute(
                "CREATE (:Entity {id: $id, name: $name, type: $type, ticker: $ticker, mentions: $m})",
                {"id": e["id"], "name": e["name"], "type": e["type"],
                 "ticker": e.get("ticker") or "", "m": e["mentions"]},
            )  # fmt: skip
            n_nodes += 1
        for line in (graph_dir / "edges.jsonl").read_text().splitlines():
            e = json.loads(line)
            c.execute(
                "MATCH (a:Entity {id: $s}), (b:Entity {id: $d}) "
                "CREATE (a)-[:Rel {relation: $r, support: $n, qualifiers: $q, evidence: $ev}]->(b)",
                {"s": e["src"], "d": e["dst"], "r": e["relation"], "n": e["support"],
                 "q": json.dumps(e["qualifiers"]), "ev": json.dumps(e["evidence"])},
            )  # fmt: skip
            n_edges += 1
        g.close()
        return {"nodes": n_nodes, "edges": n_edges, "db": str(db_path)}

    # ---- queries ----
    @staticmethod
    def _edge(row) -> Edge:
        s, rel, d, n, q, ev = row
        return Edge(s, rel, d, int(n), json.loads(q or "[]"), json.loads(ev or "[]"))

    _RET = "RETURN a.id, r.relation, b.id, r.support, r.qualifiers, r.evidence"

    def edges_of(self, entity_id: str, relations: set[str] | None = None) -> list[Edge]:
        rows = self.conn.execute(
            f"MATCH (a:Entity)-[r:Rel]->(b:Entity) WHERE a.id = $x OR b.id = $x {self._RET}",
            {"x": entity_id},
        ).get_all()
        edges = [self._edge(r) for r in rows]
        return [e for e in edges if relations is None or e.relation in relations]

    def edges_between(self, a: str, b: str) -> list[Edge]:
        rows = self.conn.execute(
            f"MATCH (a:Entity)-[r:Rel]->(b:Entity) "
            f"WHERE (a.id = $x AND b.id = $y) OR (a.id = $y AND b.id = $x) {self._RET}",
            {"x": a, "y": b},
        ).get_all()
        return [self._edge(r) for r in rows]

    def common_neighbours(
        self, a: str, b: str, types: set[str] | None = None
    ) -> list[tuple[str, list[Edge]]]:
        """Entities linked to both a and b (either direction), with the connecting edges,
        best-supported first. This answers bridge questions ("which foundry do X and Y use")."""
        ea, eb = self.edges_of(a), self.edges_of(b)

        def other(e: Edge, me: str) -> str:
            return e.dst if e.src == me else e.src

        na: dict[str, list[Edge]] = {}
        for e in ea:
            na.setdefault(other(e, a), []).append(e)
        out = []
        for e in eb:
            n = other(e, b)
            if n in na and n not in (a, b):
                out.append((n, e))
        merged: dict[str, list[Edge]] = {}
        for n, e in out:
            merged.setdefault(n, list(na[n])).append(e)
        if types:
            ok = {r[0] for r in self.conn.execute(
                "MATCH (e:Entity) WHERE e.id IN $ids AND e.type IN $t RETURN e.id",
                {"ids": list(merged), "t": list(types)}).get_all()}  # fmt: skip
            merged = {k: v for k, v in merged.items() if k in ok}
        return sorted(merged.items(), key=lambda kv: -sum(e.support for e in kv[1]))

    def entity(self, entity_id: str) -> dict | None:
        rows = self.conn.execute(
            "MATCH (e:Entity {id: $x}) RETURN e.id, e.name, e.type, e.ticker, e.mentions",
            {"x": entity_id},
        ).get_all()
        if not rows:
            return None
        i, n, t, tk, m = rows[0]
        return {"id": i, "name": n, "type": t, "ticker": tk or None, "mentions": m}

    def close(self) -> None:
        self.conn.close()
        self.db.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--snapshot", default="semis-2025")
    ap.add_argument("--graph-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    gdir = a.graph_dir or Settings.load().data_dir / "graph" / a.snapshot
    print(json.dumps(LadybugGraph.build(gdir, gdir / "graph.lbug"), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
