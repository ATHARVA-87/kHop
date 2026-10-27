"""Retrieval metrics with chunker-independent relevance.

Gold evidence is a (doc_id, quote) pair, not a chunk ID, so the same golden set scores
any chunking strategy (A0 fixed windows vs A1 structure-aware). A retrieved span counts as
a hit on an evidence item when it covers at least half of the evidence span, or when at
least half of the retrieved span lies inside it (a small chunk inside a long quote).

Each evidence item can be "found" once: duplicate hits on the same evidence earn nothing.
That keeps recall and nDCG in [0, 1] and stops overlapping chunks inflating scores.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes, collapse whitespace. Returns (norm, norm_idx -> orig_idx)."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text.translate(_TRANSLATE)):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def locate_quote_all(doc_text: str, quote: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of `quote` (case/whitespace/typography-insensitive)."""
    norm_doc, mapping = _normalise_with_map(doc_text)
    norm_q, _ = _normalise_with_map(quote)
    # Table cells were joined with " | " by the parser; let quotes match across that.
    norm_q = re.sub(r"\s*\|\s*", " | ", norm_q)
    if not norm_q:
        return []
    spans, pos = [], norm_doc.find(norm_q)
    while pos >= 0:
        spans.append((mapping[pos], mapping[pos + len(norm_q) - 1] + 1))
        pos = norm_doc.find(norm_q, pos + len(norm_q))
    return spans


def locate_quote(doc_text: str, quote: str) -> tuple[int, int] | None:
    """First occurrence of `quote` (see locate_quote_all), or None."""
    spans = locate_quote_all(doc_text, quote)
    return spans[0] if spans else None


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)


# A gold evidence item is one span, or several alternative spans when the quoted text
# occurs more than once in the filing (boilerplate). Hitting any alternative counts.
GoldItem = Span | Sequence[Span]


def _alternatives(g: GoldItem) -> Sequence[Span]:
    return (g,) if isinstance(g, Span) else g


def is_hit(retrieved: Span, gold: GoldItem, min_cover: float = 0.5) -> bool:
    for g in _alternatives(gold):
        if retrieved.doc_id != g.doc_id:
            continue
        overlap = min(retrieved.end, g.end) - max(retrieved.start, g.start)
        if overlap > 0 and (overlap >= min_cover * len(g) or overlap >= min_cover * len(retrieved)):
            return True
    return False


def first_hits(
    retrieved: Sequence[Span], gold: Sequence[GoldItem], min_cover: float = 0.5
) -> list[int | None]:
    """For each rank, the gold item newly covered at that rank (or None).

    Uses incremental maximum bipartite matching (Kuhn's augmenting paths): each retrieved
    chunk can be credited to at most one gold item and vice versa, and after every rank the
    number of covered gold items is the best possible for that prefix. A greedy
    first-match would undercount when one chunk hits two nearby gold items.
    """
    edges = [[g for g, gs in enumerate(gold) if is_hit(r, gs, min_cover)] for r in retrieved]
    owner: dict[int, int] = {}  # gold index -> retrieved index

    def augment(r: int, seen: set[int]) -> int | None:
        for g in edges[r]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner:
                owner[g] = r
                return g
            if augment(owner[g], seen) is not None:
                owner[g] = r
                return g
        return None

    out: list[int | None] = []
    for r in range(len(retrieved)):
        before = len(owner)
        g = augment(r, set())
        out.append(g if len(owner) > before else None)
    return out


def recall_at_k(
    retrieved: Sequence[Span], gold: Sequence[GoldItem], k: int, min_cover: float = 0.5
) -> float:
    if not gold:
        raise ValueError("recall is undefined without gold evidence")
    hits = first_hits(retrieved[:k], gold, min_cover)
    return sum(h is not None for h in hits) / len(gold)


def mrr(retrieved: Sequence[Span], gold: Sequence[GoldItem], k: int | None = None) -> float:
    hits = first_hits(retrieved[:k] if k else retrieved, gold)
    for rank, h in enumerate(hits, start=1):
        if h is not None:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[Span], gold: Sequence[GoldItem], k: int = 10) -> float:
    """Binary-gain nDCG where each gold item can contribute once."""
    if not gold:
        raise ValueError("nDCG is undefined without gold evidence")
    hits = first_hits(retrieved[:k], gold)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, h in enumerate(hits, 1) if h is not None)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal


def retrieval_metrics(
    retrieved: Sequence[Span], gold: Sequence[GoldItem], ks: Sequence[int] = (1, 5, 10, 20)
) -> dict[str, float]:
    out = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in ks}
    out["ndcg@10"] = ndcg_at_k(retrieved, gold, 10)
    out["mrr"] = mrr(retrieved, gold)
    return out


SENSITIVITY_COVERS = (0.3, 0.7)


def threshold_sensitivity(
    retrieved: Sequence[Span], gold: Sequence[GoldItem], k: int = 10
) -> dict[str, float]:
    """recall@k under stricter/looser hit thresholds (ADR 0004 sensitivity check)."""
    return {f"recall@{k}@cover{c}": recall_at_k(retrieved, gold, k, c) for c in SENSITIVITY_COVERS}
