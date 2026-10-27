"""Dense embeddings (sentence-transformers) and a Qdrant local-mode index.

Embeddings are cached on disk per (model, chunk text) so re-indexing after a crash, or
building a second config with the same chunks, does not re-embed anything.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plexus.config import EmbedderConfig
from plexus.ingest.chunk import Chunk


class Embedder:
    def __init__(self, cfg: EmbedderConfig, cache_path: Path | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, device=cfg.device)
        get_dim = (
            getattr(self.model, "get_embedding_dimension", None)
            or self.model.get_sentence_embedding_dimension
        )
        self.dim = get_dim()
        self._cache = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = sqlite3.connect(str(cache_path))
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)"
            )

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.cfg.model}\x00{text}".encode()).hexdigest()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,  # cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        if self._cache is None:
            return self._encode(texts)
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):  # SQLite parameter limit
            batch = keys[i : i + 900]
            q = f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(batch))})"
            for k, blob in self._cache.execute(q, batch):
                found[k] = np.frombuffer(blob, dtype=np.float32)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            vecs = self._encode([texts[i] for i in todo])
            self._cache.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                [(keys[i], v.tobytes()) for i, v in zip(todo, vecs, strict=True)],
            )
            self._cache.commit()
            for i, v in zip(todo, vecs, strict=True):
                found[keys[i]] = v
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.dim), np.float32)

    def count_truncated(self, texts: Sequence[str]) -> int:
        """How many passages exceed the model's max sequence length (silently truncated)."""
        limit = self.model.get_max_seq_length() or 512
        tok = self.model.tokenizer
        return sum(len(tok(t, add_special_tokens=True)["input_ids"]) > limit for t in texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.cfg.query_prefix + text])[0]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    section: str
    score: float
    text: str = ""
    parent_id: str = ""
    parent_start: int = -1
    parent_end: int = -1
    header: str = ""


class DenseIndex:
    """Thin wrapper over a Qdrant collection in local (embedded, on-disk) mode."""

    def __init__(self, path: Path | str, collection: str, dim: int):
        from qdrant_client import QdrantClient

        self.client = (
            QdrantClient(location=":memory:")
            if str(path) == ":memory:"
            else QdrantClient(path=str(path))
        )
        self.collection = collection
        self.dim = dim

    def reset(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=i + j, vector=vectors[i + j].tolist(), payload=c.to_dict())
                for j, c in enumerate(chunks[i : i + batch])
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(self, vector: np.ndarray, k: int) -> list[Hit]:
        res = self.client.query_points(
            self.collection, query=vector.tolist(), limit=k, with_payload=True
        ).points
        return [
            Hit(
                chunk_id=p.payload["chunk_id"],
                doc_id=p.payload["doc_id"],
                start=p.payload["start"],
                end=p.payload["end"],
                section=p.payload["section"],
                score=float(p.score),
                text=p.payload["text"],
                parent_id=p.payload.get("parent_id", ""),
                parent_start=p.payload.get("parent_start", -1),
                parent_end=p.payload.get("parent_end", -1),
                header=p.payload.get("header", ""),
            )
            for p in res
        ]

    def close(self) -> None:
        self.client.close()
