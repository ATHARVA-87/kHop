"""The one entry point for LLM calls: cache first, provider second, typed output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from plexus.llm.cache import LLMCache
from plexus.llm.providers import (
    ClaudeSDKProvider,
    LiteLLMProvider,
    Provider,
    ProviderError,
)
from plexus.llm.types import LLMRequest, LLMResponse
from plexus.settings import Settings

T = TypeVar("T", bound=BaseModel)


class CacheMissError(RuntimeError):
    """Raised in offline mode when a request is not cached (keeps CI free)."""


class StructuredOutputError(RuntimeError):
    pass


@dataclass
class CallStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.calls if self.calls else 0.0


class LLMClient:
    def __init__(
        self,
        cache: LLMCache,
        providers: dict[str, Provider] | None = None,
        offline: bool = False,
    ):
        self.cache = cache
        self.providers: dict[str, Provider] = dict(providers or {})
        self.offline = offline
        self.stats = CallStats()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LLMClient:
        s = settings or Settings.load()
        providers: dict[str, Provider] = {
            "ollama": LiteLLMProvider("ollama", api_base=s.ollama_api_base),
            "gemini": LiteLLMProvider("gemini"),
            "groq": LiteLLMProvider("groq"),
        }
        if s.enable_claude_sdk:
            providers["claude_sdk"] = ClaudeSDKProvider()
        return cls(LLMCache(s.llm_cache_path), providers, offline=s.llm_offline)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.stats.calls += 1
        self.stats.by_provider[request.provider] = (
            self.stats.by_provider.get(request.provider, 0) + 1
        )
        cached = self.cache.get(request)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.offline:
            raise CacheMissError(
                f"offline mode: no cached response for {request.provider}/{request.model} "
                f"(key {request.cache_key()[:12]})"
            )
        provider = self.providers.get(request.provider)
        if provider is None:
            hint = (
                " (set PLEXUS_ENABLE_CLAUDE_SDK=1 for personal local runs only)"
                if request.provider == "claude_sdk"
                else ""
            )
            raise ProviderError(f"provider {request.provider!r} is not enabled{hint}")
        response = self._call_with_retry(provider, request)
        if not response.text.strip():  # defence in depth: an empty answer is never cached
            raise ProviderError(f"{request.provider}/{request.model}: empty response")
        self.stats.prompt_tokens += response.usage.prompt_tokens
        self.stats.completion_tokens += response.usage.completion_tokens
        self.cache.put(request, response)
        return response.model_copy(update={"cache_key": request.cache_key()})

    @staticmethod
    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _call_with_retry(provider: Provider, request: LLMRequest) -> LLMResponse:
        return provider.complete(request)

    def complete_json(self, request: LLMRequest, schema: type[T]) -> T:
        """Call with the model's JSON schema attached and validate the reply."""
        if request.json_schema is None:
            request = request.model_copy(update={"json_schema": schema.model_json_schema()})
        response = self.complete(request)
        try:
            return schema.model_validate_json(_extract_json(response.text))
        except (ValidationError, ValueError) as exc:
            raise StructuredOutputError(
                f"{request.provider}/{request.model} returned invalid {schema.__name__}: {exc}"
            ) from exc


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Small models often wrap JSON in ``` fences or add a preamble; strip that."""
    match = _FENCE.search(text)
    candidate = match.group(1) if match else text
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON object found")
    json.loads(candidate[start:])  # raises ValueError on bad JSON
    return candidate[start:]
