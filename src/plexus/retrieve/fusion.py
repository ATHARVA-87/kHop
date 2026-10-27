"""Reciprocal rank fusion (Cormack, Clarke & Buettcher 2009).

score(chunk) = sum over ranked lists of 1 / (k + rank), rank starting at 1. Uses ranks only,
so BM25 scores and cosine similarities never need to be put on the same scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from plexus.index.dense import Hit


def rrf(ranked_lists: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    order: dict[str, tuple[int, int]] = {}  # tie-break: best rank seen, then list index
    for li, hits in enumerate(ranked_lists):
        for rank, h in enumerate(hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(h.chunk_id, h)
            order[h.chunk_id] = min(order.get(h.chunk_id, (rank, li)), (rank, li))
    ids = sorted(scores, key=lambda c: (-scores[c], order[c], c))
    return [replace(first[c], score=scores[c]) for c in ids]
