"""A6: route each question to the retriever that suits its type (Phase 4).

    0 companies named        -> "global"     company-diverse ranking (A5)
    2+ corpus filers named   -> "multi_hop"  graph traversal + hybrid (A4)
    otherwise                -> "local"      hybrid + cross-encoder reranker (A3)

Rules on the existing entity linker, no LLM call: the idea of Adaptive-RAG (Jeong et al.
2024, route by query complexity) without training a classifier. Routing also lets each
type use its best retriever: the reranker helped single-company questions but hurt
two-company ones (Phase 2), so it now runs on "local" only.
"""

from __future__ import annotations

from collections.abc import Callable

from plexus.index.dense import Hit

ROUTES = ("local", "multi_hop", "global")


class RuleRouter:
    def __init__(self, linker):
        self.linker = linker
        self.filer_ids = set(linker.tickers.values())

    def route(self, question: str) -> str:
        linked = self.linker.link(question)
        if not linked:
            return "global"
        filers = [e for e in linked if e in self.filer_ids]
        return "multi_hop" if len(filers) >= 2 else "local"


class FixedRouter:
    """Always the same route: the 'always-local' / 'always-global' baselines of the gate."""

    def __init__(self, route: str):
        self._route = route

    def route(self, question: str) -> str:
        return self._route


class OracleRouter:
    """The golden question type: the upper bound that shows what routing errors cost."""

    def __init__(self, qtype_of: Callable[[str], str]):
        self.qtype_of = qtype_of

    def route(self, question: str) -> str:
        return self.qtype_of(question)


class RoutedRetriever:
    def __init__(self, router, retrievers: dict[str, object]):
        missing = set(ROUTES) - set(retrievers)
        if missing:
            raise ValueError(f"no retriever for routes {sorted(missing)}")
        self.router = router
        self.retrievers = retrievers
        self.last_route = ""

    def retrieve(self, question: str, k: int) -> list[Hit]:
        self.last_route = self.router.route(question)
        return self.retrievers[self.last_route].retrieve(question, k)

    def close(self) -> None:
        seen = set()
        for r in self.retrievers.values():
            if id(r) not in seen and hasattr(r, "close"):
                seen.add(id(r))
                r.close()
