"""LLM providers. Each one turns an LLMRequest into an LLMResponse. No caching here."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Protocol

from plexus.llm.types import LLMRequest, LLMResponse, Usage


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class FakeProvider:
    """Deterministic provider for tests. Never touches the network.

    `responder` maps a request to its text; by default it echoes the last user message.
    """

    name = "fake"

    def __init__(self, responder: Callable[[LLMRequest], str] | None = None):
        self.responder = responder or (lambda r: f"echo: {r.messages[-1].content}")
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        text = self.responder(request)
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=text,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=len(text.split())),
        )


# LiteLLM model-string prefix per provider. Keys are Plexus provider names.
_LITELLM_PREFIX = {"ollama": "ollama", "gemini": "gemini", "groq": "groq"}


class LiteLLMProvider:
    """Ollama, Gemini and Groq through LiteLLM, so agents can switch provider by config."""

    def __init__(self, name: str, api_base: str | None = None):
        if name not in _LITELLM_PREFIX:
            raise ValueError(f"unknown LiteLLM provider {name!r}")
        self.name = name
        self.api_base = api_base

    def complete(self, request: LLMRequest) -> LLMResponse:
        import litellm  # imported lazily: it is slow to import

        kwargs: dict = {
            "model": f"{_LITELLM_PREFIX[self.name]}/{request.model}",
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            **request.params,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if self.api_base and self.name == "ollama":
            kwargs["api_base"] = self.api_base
        # one attempt per call with a hard timeout; LLMClient owns the retry policy
        kwargs.setdefault("timeout", 120)
        kwargs.setdefault("num_retries", 0)
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema, "strict": True},
            }
        start = time.perf_counter()
        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:  # LiteLLM wraps many provider errors
            raise ProviderError(f"{self.name}/{request.model}: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        text = resp.choices[0].message.content or ""
        if not text.strip():
            # e.g. a thinking model spent max_tokens on hidden reasoning; never cache this
            raise ProviderError(f"{self.name}/{request.model}: empty response (raise max_tokens?)")
        return LLMResponse(
            text=text,
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=latency,
        )


class ClaudeSDKProvider:
    """OPTIONAL, OFF BY DEFAULT. Personal local comparison runs only.

    Uses the developer's own Claude login through the Claude Agent SDK
    (`pip install claude-agent-sdk`). It must never serve other users, so the
    client refuses to build it unless PLEXUS_ENABLE_CLAUDE_SDK=1.
    Status: written but not yet exercised against the real SDK.
    """

    name = "claude_sdk"

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from claude_agent_sdk import (  # type: ignore[import-not-found]
                AssistantMessage,
                ClaudeAgentOptions,
                TextBlock,
                query,
            )
        except ImportError as exc:
            raise ProviderError("claude_sdk provider needs `pip install claude-agent-sdk`") from exc

        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        prompt = "\n\n".join(m.content for m in request.messages if m.role != "system")
        if request.json_schema is not None:
            prompt += "\n\nReply with JSON only, matching this schema:\n" + json.dumps(
                request.json_schema
            )
        options = ClaudeAgentOptions(system_prompt=system or None, max_turns=1, model=request.model)

        async def _run() -> str:
            parts: list[str] = []
            async for msg in query(prompt=prompt, options=options):
                if isinstance(msg, AssistantMessage):
                    parts.extend(b.text for b in msg.content if isinstance(b, TextBlock))
            return "".join(parts)

        start = time.perf_counter()
        text = asyncio.run(_run())
        return LLMResponse(text=text, latency_ms=(time.perf_counter() - start) * 1000)
