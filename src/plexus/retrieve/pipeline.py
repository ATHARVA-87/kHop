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

import numpy as np

from plexus.config import ExperimentConfig
from plexus.index.dense import Hit
from plexus.ingest.headers import index_text
from plexus.retrieve.bm25 import BM25Index
from plexus.retrieve.fusion import rrf

log = logging.getLogger("plexus.retrieve")


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index

    def retrieve(self, question: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        if doc_ids is not None and not doc_ids:
            return []  # an empty filter means "no filings", not "all filings"
        return self.index.search(self.embedder.embed_query(question), k, doc_ids)

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
            index = BM25Index.build(
                (index_text(c.get("header", ""), c["text"]) for c in chunks), k1, b
            )
            index.save(cache)
            log.info("built BM25 over %d chunks in %.1fs", len(chunks), time.perf_counter() - t0)
        return cls(chunks, index)

    def retrieve(self, question: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        allowed = None
        if doc_ids is not None:
            keep = set(doc_ids)
            allowed = np.array([c["doc_id"] in keep for c in self.chunks])
        out = []
        for i, score in self.index.search(question, k, allowed):
            c = self.chunks[i]
            fields = {f: c[f] for f in ("chunk_id", "doc_id", "start", "end", "section", "text")}
            parent = {
                f: c[f] for f in ("parent_id", "parent_start", "parent_end", "header") if f in c
            }
            out.append(Hit(score=score, **fields, **parent))
        return out


class HybridRetriever:
    """Each retriever returns `candidates` hits; RRF fuses them; the top k are kept."""

    def __init__(self, retrievers: list[Retriever], candidates: int = 100, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        n = max(k, self.candidates)
        return rrf([r.retrieve(question, n, doc_ids) for r in self.retrievers], self.rrf_k)[:k]

    def close(self) -> None:
        for r in self.retrievers:
            if hasattr(r, "close"):
                r.close()


class OneCompanyFilter:
    """If the question names exactly one corpus company, search only that company's filings.
    Found by using the app (2026-10-29): "Which foundry does NVIDIA rely on?" was answered from
    MaxLinear passages, because the single-company route searched every filing."""

    def __init__(self, base, linker, doc_ids: list[str]):
        self.base, self.linker = base, linker
        self.by_ticker: dict[str, list[str]] = {}
        for d in sorted(set(doc_ids)):
            self.by_ticker.setdefault(d.split("_", 1)[0], []).append(d)

    def retrieve(self, question: str, k: int) -> list[Hit]:
        filers = [e for e in self.linker.link(question) if e in self.linker.filer_ids]
        if len(filers) == 1:
            ids = self.by_ticker.get(filers[0].split(":", 1)[-1])
            if ids:
                return self.base.retrieve(question, k, ids)
        return self.base.retrieve(question, k)


def make_linker(cfg: ExperimentConfig, data_dir: Path):
    """One entity linker for A2-RR, A4 and A5: alias table + SEC/filer names + graph entities
    (when the graph exists), so all three see the same companies in a question."""
    from plexus.extract.resolve import load_resolver
    from plexus.retrieve.graph_search import EntityLinker, load_entities

    r = cfg.retrieval
    gdir = Path(r.graph_dir) if r.graph_dir else data_dir / "graph" / cfg.snapshot
    ents = load_entities(gdir) if (gdir / "entities.jsonl").exists() else []
    return EntityLinker(load_resolver(data_dir, cfg.snapshot, r.aliases_file), ents)


def _graph_hybrid(cfg: ExperimentConfig, data_dir: Path, index_path: Path, base: Retriever):
    from plexus.index.graph import LadybugGraph
    from plexus.retrieve.graph_search import GraphHybridRetriever, GraphRetriever

    r = cfg.retrieval
    gdir = Path(r.graph_dir) if r.graph_dir else data_dir / "graph" / cfg.snapshot
    if not (gdir / "graph.lbug").exists():
        raise FileNotFoundError(f"no graph at {gdir}: run `make graph` first")
    graph = GraphRetriever(LadybugGraph(gdir / "graph.lbug"), make_linker(cfg, data_dir),
                           load_chunks(index_path), r.graph_max_edges)  # fmt: skip
    return GraphHybridRetriever(graph, base, r.rrf_k)


def _routed(cfg: ExperimentConfig, data_dir: Path, index_path: Path, hybrid: Retriever):
    """A6: one hybrid retriever shared by three routes (one copy of the models on the GPU)."""
    from plexus.retrieve.graph_search import CompanyDiverseRetriever
    from plexus.retrieve.rerank import CrossEncoderScorer, RerankingRetriever
    from plexus.retrieve.router import FixedRouter, RoutedRetriever, RuleRouter

    r = cfg.retrieval
    if r.rerank is None:
        raise ValueError("routed retrieval needs retrieval.rerank (used on the local route)")
    linker = make_linker(cfg, data_dir)
    retrievers = {
        "local": RerankingRetriever(
            OneCompanyFilter(hybrid, linker, [c["doc_id"] for c in load_chunks(index_path)])
            if r.local_company_filter
            else hybrid,
            CrossEncoderScorer(r.rerank),
            r.rerank.candidates,
        ),
        "multi_hop": _graph_hybrid(cfg, data_dir, index_path, hybrid),
        "global": CompanyDiverseRetriever(hybrid, linker),
    }
    router = RuleRouter(linker) if r.router == "rules" else FixedRouter(r.router)
    routed = RoutedRetriever(router, retrievers)
    graph = retrievers["multi_hop"].graph_retriever.graph
    routed.closables = [hybrid, graph]  # each shared resource closed exactly once
    return routed


def make_retriever(cfg: ExperimentConfig, data_dir: Path) -> Retriever:
    from plexus.index.build import index_dir

    r = cfg.retrieval
    path = index_dir(data_dir, cfg)
    if not (path / "chunks.jsonl").exists():
        raise FileNotFoundError(f"no index at {path}: run `make index CONFIG=...` first")

    def dense() -> DenseRetriever:
        from plexus.index.dense import DenseIndex, Embedder

        if not (path / "qdrant").exists():
            raise FileNotFoundError(f"no dense index at {path}: built with --chunks-only?")
        emb = Embedder(cfg.embedder, cache_path=data_dir / "cache" / "embeddings.sqlite")
        return DenseRetriever(emb, DenseIndex(path / "qdrant", "chunks", emb.dim))

    base: Retriever
    if r.type == "dense":
        base = dense()
    else:
        bm25 = BM25Retriever.from_index_dir(path, r.bm25_k1, r.bm25_b)
        keyword_only = r.type == "bm25" or (r.type == "graph_hybrid" and r.graph_base == "bm25")
        base = bm25 if keyword_only else HybridRetriever([dense(), bm25], r.candidates, r.rrf_k)
    if r.type == "routed":
        return _routed(cfg, data_dir, path, base)
    if r.type == "graph_hybrid":
        base = _graph_hybrid(cfg, data_dir, path, base)
    elif r.type == "hybrid_rr":
        from plexus.retrieve.graph_search import CompanyRoundRobinRetriever

        base = CompanyRoundRobinRetriever(base, make_linker(cfg, data_dir))
    if r.company_diverse:
        from plexus.retrieve.graph_search import CompanyDiverseRetriever

        base = CompanyDiverseRetriever(base, make_linker(cfg, data_dir))
    if r.rerank is None:
        return base
    from plexus.retrieve.rerank import CrossEncoderScorer, RerankingRetriever

    return RerankingRetriever(base, CrossEncoderScorer(r.rerank), r.rerank.candidates)
