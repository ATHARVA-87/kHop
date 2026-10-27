"""Build the chunk index for an experiment config.

    python -m plexus.index.build --config configs/a0_naive.yaml

Output: data/index/<index_key>/{chunks.jsonl, qdrant/, build_info.json}. Configs with the
same chunker + embedder share one index (see ExperimentConfig.index_key).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from plexus.config import ExperimentConfig
from plexus.ingest.chunk import Chunk, fixed_chunks, structured_chunks
from plexus.ingest.headers import chunk_header, index_text
from plexus.ingest.process import iter_docs
from plexus.settings import Settings

log = logging.getLogger("plexus.index.build")


def index_dir(data_dir: Path, cfg: ExperimentConfig) -> Path:
    return data_dir / "index" / cfg.index_key


def make_chunks(cfg: ExperimentConfig, data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    c = cfg.chunker
    for meta, doc in iter_docs(data_dir / "processed", cfg.snapshot):
        if c.type == "fixed":
            doc_chunks = fixed_chunks(doc, c.size_words, c.overlap_words)
        elif c.type == "structured":
            doc_chunks = structured_chunks(doc, c.child_words, c.child_chars, c.parent_words)
        else:  # pragma: no cover - guarded by the config Literal
            raise ValueError(c.type)
        if c.headers:
            doc_chunks = [replace(x, header=chunk_header(meta, x.section)) for x in doc_chunks]
        chunks.extend(doc_chunks)
    return chunks


def build(cfg: ExperimentConfig, data_dir: Path, chunks_only: bool = False) -> dict:
    """chunks_only: write chunks.jsonl (all BM25 needs) and skip embeddings, so the CI
    retrieval gate runs without torch or a GPU."""
    out = index_dir(data_dir, cfg)
    out.mkdir(parents=True, exist_ok=True)
    chunks = make_chunks(cfg, data_dir)
    if not chunks:
        raise RuntimeError(
            f"no chunks: run `python -m plexus.ingest.process --snapshot {cfg.snapshot}`"
        )
    with open(out / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict()) + "\n")

    summary_path = data_dir / "snapshots" / f"{cfg.snapshot}.summary.json"
    checksum = (
        json.loads(summary_path.read_text())["corpus_checksum"] if summary_path.exists() else None
    )
    info = {
        "index_key": cfg.index_key,
        "snapshot": cfg.snapshot,
        "corpus_checksum": checksum,
        "documents": len({c.doc_id for c in chunks}),
        "chunks": len(chunks),
        "parents": len({c.parent_id for c in chunks if c.parent_id}),
        "mean_chunk_words": round(sum(len(c.text.split()) for c in chunks) / len(chunks), 1),
    }
    info_path = out / "build_info.json"
    if chunks_only:
        old = json.loads(info_path.read_text()) if info_path.exists() else {}
        if old.get("chunks") == info["chunks"]:
            info = {**old, **info}  # keep the dense fields of an earlier full build
    else:
        info.update(_embed(cfg, data_dir, out, chunks))
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return info


def _embed(cfg: ExperimentConfig, data_dir: Path, out: Path, chunks: list[Chunk]) -> dict:
    from plexus.index.dense import DenseIndex, Embedder

    t0 = time.perf_counter()
    embedder = Embedder(cfg.embedder, cache_path=data_dir / "cache" / "embeddings.sqlite")
    texts = [index_text(c.header, c.text) for c in chunks]
    vectors = embedder.embed_passages(texts)
    embed_s = time.perf_counter() - t0
    index = DenseIndex(out / "qdrant", "chunks", embedder.dim)
    index.reset()
    index.add(chunks, vectors)
    n = index.count()
    index.close()
    return {
        "indexed_points": n,
        "embedder": cfg.embedder.model,
        "embedding_dim": embedder.dim,
        "embed_seconds": round(embed_s, 1),
        "chunks_truncated_by_embedder": embedder.count_truncated(texts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunks-only", action="store_true", help="skip embeddings (BM25 / CI)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = build(ExperimentConfig.load(args.config), Settings.load().data_dir, args.chunks_only)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
