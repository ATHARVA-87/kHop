"""Experiment configs (configs/a*.yaml). One file = one ablation run, fully specified."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from plexus.settings import REPO_ROOT


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a config must fail loudly


class ChunkerConfig(_Strict):
    type: Literal["fixed", "structured"] = "fixed"
    # fixed (A0)
    size_words: int = 300
    overlap_words: int = 50
    # structured (A1): see plexus.ingest.chunk.structured_chunks
    child_words: int = 200
    child_chars: int = 1400
    parent_words: int = 800


class EmbedderConfig(_Strict):
    model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this prefix on queries (not on passages)
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 64
    device: str | None = None  # None = cuda if available


class RetrievalConfig(_Strict):
    type: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = 20
    candidates: int = 100  # hybrid: hits taken from each retriever before fusion
    rrf_k: int = 60  # reciprocal rank fusion constant (the paper's default)
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


class GenerationConfig(_Strict):
    provider: str = "ollama"
    model: str = "qwen3.5:4b"
    temperature: float = 0.0
    max_tokens: int = 512
    context_chunks: int = 5


class JudgeConfig(_Strict):
    provider: str = "gemini"
    model: str = "gemini-3.6-flash"  # 3.8-flash overloaded on 2026-10-27, ADR 0005


class ExperimentConfig(_Strict):
    name: str
    description: str = ""
    snapshot: str = "semis-2025"
    seed: int = 0
    chunker: ChunkerConfig = ChunkerConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    judge: JudgeConfig = JudgeConfig()

    @classmethod
    def load(cls, path: Path | str) -> ExperimentConfig:
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @property
    def index_key(self) -> str:
        """Configs that share chunker + embedder share an index."""
        c, e = self.chunker, self.embedder
        model = e.model.replace("/", "_")
        from plexus.ingest.parse import VERSION as PARSER_VERSION

        # parser version included: a parser fix must not silently reuse an old index
        if c.type == "fixed":
            shape = f"fixed-{c.size_words}-{c.overlap_words}"
        else:
            shape = f"structured-{c.child_words}-{c.child_chars}-{c.parent_words}"
        return f"{self.snapshot}__p{PARSER_VERSION}__{shape}__{model}"
