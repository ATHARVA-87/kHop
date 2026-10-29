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

import re
from collections.abc import Callable

from plexus.index.dense import Hit

ROUTES = ("local", "multi_hop", "global")

SECTOR_CUE = re.compile(
    r"\b(which|what|how many|how do|do|across)\s+(the\s+|all\s+|other\s+)?"
    r"(companies|firms|chipmakers|chip makers|semiconductor companies|manufacturers|filers|peers)\b"
    r"|\b(industry|sector)[- ]wide\b|\bacross the (industry|sector)\b"
    r"|\b(chip ?makers|chip companies|semiconductor companies)\b",
    re.IGNORECASE,
)


class RuleRouter:
    def __init__(self, linker):
        self.linker = linker
        self.filer_ids = linker.filer_ids

    def route(self, question: str) -> str:
        linked = self.linker.link(question)
        filers = [e for e in linked if e in self.filer_ids]
        if len(filers) >= 2:
            return "multi_hop"
        # "Which companies depend on TSMC?" names a company but asks across the sector
        if not filers and (not linked or SECTOR_CUE.search(question)):
            return "global"
        return "local"


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
        # the routes wrap ONE shared hybrid retriever: close it once (review 2026-10-28)
        for r in getattr(self, "closables", None) or self.retrievers.values():
            if hasattr(r, "close"):
                r.close()
