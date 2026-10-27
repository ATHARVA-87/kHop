"""Cross-encoder reranking (A3).

A bi-encoder embeds question and passage separately; a cross-encoder reads them together
and scores relevance directly. It is too slow to run over 31k chunks, so it reorders only
the first `candidates` hits of the base retriever. Hits beyond that keep their base order,
so the reranked prefix is the same whatever k the caller asks for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from plexus.config import RerankConfig
from plexus.index.dense import Hit

ScoreFn = Callable[[str, Sequence[str]], Sequence[float]]


class CrossEncoderScorer:
    def __init__(self, cfg: RerankConfig):
        from sentence_transformers import CrossEncoder  # heavy import

        self.cfg = cfg
        self.model = CrossEncoder(cfg.model, max_length=cfg.max_length, device=cfg.device)
        if self.model.model.device.type == "cuda":
            self.model.model.half()  # fp16 on GPU: same ranking in practice, half the memory

    def __call__(self, question: str, passages: Sequence[str]) -> list[float]:
        pairs = [(question, p) for p in passages]
        scores = self.model.predict(pairs, batch_size=self.cfg.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


class RerankingRetriever:
    def __init__(self, base, score_fn: ScoreFn, candidates: int = 50):
        self.base = base
        self.score_fn = score_fn
        self.candidates = candidates

    def retrieve(self, question: str, k: int) -> list[Hit]:
        hits = self.base.retrieve(question, max(k, self.candidates))
        head, tail = hits[: self.candidates], hits[self.candidates :]
        if not head:
            return []
        scores = self.score_fn(question, [h.text for h in head])
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))  # ties keep base order
        reranked = [replace(head[i], score=scores[i]) for i in order]
        return (reranked + tail)[:k]

    def close(self) -> None:
        if hasattr(self.base, "close"):
            self.base.close()
