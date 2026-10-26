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
from pathlib import Path

from plexus.config import ExperimentConfig
from plexus.ingest.chunk import Chunk, fixed_chunks
from plexus.ingest.process import iter_docs
from plexus.settings import Settings

log = logging.getLogger("plexus.index.build")


def index_dir(data_dir: Path, cfg: ExperimentConfig) -> Path:
    return data_dir / "index" / cfg.index_key


def make_chunks(cfg: ExperimentConfig, data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for _, doc in iter_docs(data_dir / "processed", cfg.snapshot):
        if cfg.chunker.type == "fixed":
            chunks.extend(fixed_chunks(doc, cfg.chunker.size_words, cfg.chunker.overlap_words))
        else:  # pragma: no cover - guarded by the config Literal
            raise ValueError(cfg.chunker.type)
    return chunks


def build(cfg: ExperimentConfig, data_dir: Path) -> dict:
    from plexus.index.dense import DenseIndex, Embedder

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

    t0 = time.perf_counter()
    embedder = Embedder(cfg.embedder, cache_path=data_dir / "cache" / "embeddings.sqlite")
    vectors = embedder.embed_passages([c.text for c in chunks])
    embed_s = time.perf_counter() - t0

    index = DenseIndex(out / "qdrant", "chunks", embedder.dim)
    index.reset()
    index.add(chunks, vectors)
    n = index.count()
    index.close()

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
        "indexed_points": n,
        "embedder": cfg.embedder.model,
        "embedding_dim": embedder.dim,
        "embed_seconds": round(embed_s, 1),
    }
    (out / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = build(ExperimentConfig.load(args.config), Settings.load().data_dir)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
