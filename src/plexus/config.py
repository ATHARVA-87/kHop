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
    type: Literal["fixed"] = "fixed"
    size_words: int = 300
    overlap_words: int = 50


class EmbedderConfig(_Strict):
    model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this prefix on queries (not on passages)
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 64
    device: str | None = None  # None = cuda if available


class RetrievalConfig(_Strict):
    type: Literal["dense"] = "dense"
    top_k: int = 20


class GenerationConfig(_Strict):
    provider: str = "ollama"
    model: str = "qwen3.5:4b"
    temperature: float = 0.0
    max_tokens: int = 512
    context_chunks: int = 5


class JudgeConfig(_Strict):
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"  # checked 2026-10-27, ADR 0005


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
        return f"{self.snapshot}__{c.type}-{c.size_words}-{c.overlap_words}__{model}"
