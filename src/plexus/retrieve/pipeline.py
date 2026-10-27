"""Retrievers selected by `retrieval.type` in an experiment config.

dense  : bge embeddings in Qdrant (A0)
bm25   : keyword-only baseline
hybrid : dense + BM25, fused with reciprocal rank fusion (A2)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

from plexus.config import ExperimentConfig
from plexus.index.dense import Hit
from plexus.retrieve.bm25 import BM25Index
from plexus.retrieve.fusion import rrf

log = logging.getLogger("plexus.retrieve")


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index

    def retrieve(self, question: str, k: int) -> list[Hit]:
        return self.index.search(self.embedder.embed_query(question), k)

    def close(self) -> None:
        if hasattr(self.index, "close"):
            self.index.close()  # Qdrant local mode holds a file lock


def load_chunks(index_path: Path) -> list[dict]:
    with open(index_path / "chunks.jsonl") as fh:
        return [json.loads(line) for line in fh]


class BM25Retriever:
    def __init__(self, chunks: list[dict], index: BM25Index):
        if len(chunks) != index.n_docs:
            raise ValueError(f"{len(chunks)} chunks but BM25 index has {index.n_docs} docs")
        self.chunks = chunks
        self.index = index

    @classmethod
    def from_index_dir(cls, index_path: Path, k1: float = 1.2, b: float = 0.75) -> BM25Retriever:
        """Loads data/index/<key>/bm25.pkl, building it from chunks.jsonl on first use."""
        chunks = load_chunks(index_path)
        cache = index_path / "bm25.pkl"
        index = BM25Index.load(cache, k1, b) if cache.exists() else None
        if index is None or index.n_docs != len(chunks):
            t0 = time.perf_counter()
            index = BM25Index.build((c["text"] for c in chunks), k1, b)
            index.save(cache)
            log.info("built BM25 over %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)
        return cls(chunks, index)

    def retrieve(self, question: str, k: int) -> list[Hit]:
        out = []
        for i, score in self.index.search(question, k):
            c = self.chunks[i]
            fields = {f: c[f] for f in ("chunk_id", "doc_id", "start", "end", "section", "text")}
            parent = {f: c[f] for f in ("parent_id", "parent_start", "parent_end") if f in c}
            out.append(Hit(score=score, **fields, **parent))
        return out


class HybridRetriever:
    """Each retriever returns `candidates` hits; RRF fuses them; the top k are kept."""

    def __init__(self, retrievers: list[Retriever], candidates: int = 100, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int) -> list[Hit]:
        n = max(k, self.candidates)
        return rrf([r.retrieve(question, n) for r in self.retrievers], self.rrf_k)[:k]

    def close(self) -> None:
        for r in self.retrievers:
            if hasattr(r, "close"):
                r.close()


def make_retriever(cfg: ExperimentConfig, data_dir: Path) -> Retriever:
    from plexus.index.build import index_dir

    r = cfg.retrieval
    path = index_dir(data_dir, cfg)
    if not (path / "chunks.jsonl").exists():
        raise FileNotFoundError(f"no index at {path}: run `make index CONFIG=...` first")

    def dense() -> DenseRetriever:
        from plexus.index.dense import DenseIndex, Embedder

        emb = Embedder(cfg.embedder, cache_path=data_dir / "cache" / "embeddings.sqlite")
        return DenseRetriever(emb, DenseIndex(path / "qdrant", "chunks", emb.dim))

    base: Retriever
    if r.type == "dense":
        base = dense()
    else:
        bm25 = BM25Retriever.from_index_dir(path, r.bm25_k1, r.bm25_b)
        base = bm25 if r.type == "bm25" else HybridRetriever([dense(), bm25], r.candidates, r.rrf_k)
    if r.rerank is None:
        return base
    from plexus.retrieve.rerank import CrossEncoderScorer, RerankingRetriever

    return RerankingRetriever(base, CrossEncoderScorer(r.rerank), r.rerank.candidates)
