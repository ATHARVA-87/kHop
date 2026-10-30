"""Typed request/response objects for every LLM call in kHop."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class LLMRequest(BaseModel):
    """Everything that can change an LLM's output. The cache key is derived from all of it."""

    model_config = ConfigDict(frozen=True)

    provider: str  # "ollama" | "gemini" | "groq" | "claude_sdk" | "fake"
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int | None = 0
    # Optional JSON schema the response must follow (structured output).
    json_schema: dict[str, Any] | None = None
    # Provider-specific extras (e.g. top_p). Included in the cache key.
    params: dict[str, Any] = Field(default_factory=dict)

    def cache_key(self) -> str:
        """sha256 over a canonical JSON dump: same request -> same key, on any machine."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0  # latency of the original provider call
    cached: bool = False  # True when served from the disk cache
    cache_key: str = ""
