# ADR 0001: One LLM wrapper with a SQLite disk cache and an offline mode

- Status: Accepted (owner sign-off 2026-10-30)
- Date: 2026-10-26

## Context
Plexus makes LLM calls in extraction, golden-set generation, five agents and the LLM judge. Rules: reruns and CI must cost nothing; providers must be switchable per agent by config; outputs must be typed.

## Decision
- All calls go through `plexus.llm.LLMClient`. Providers (Ollama, Gemini, Groq via LiteLLM; optional `claude_sdk`; `fake` for tests) implement one `complete(LLMRequest) -> LLMResponse` method.
- Cache key = sha256 of the canonical JSON of the whole request (provider, model, messages, temperature, max_tokens, seed, JSON schema, extra params). Anything that can change the output is in the key.
- Storage is one SQLite file in WAL mode with a lock, safe for parallel Researcher calls and easy to ship as a CI artifact.
- `PLEXUS_LLM_OFFLINE=1` makes a cache miss an error. CI uses it, so CI can never call a provider.
- `complete_json` attaches a Pydantic model's JSON schema and validates the reply, stripping the code fences that small models often add.

## Alternatives considered
- LiteLLM's built-in cache: ties the key format to LiteLLM, and it does not cover the `claude_sdk` provider.
- One JSON file per call: simple, but thousands of small files and non-atomic writes.
- `diskcache`: fine, but another dependency for something about 80 lines of SQLite does.

## Consequences
- Changing a prompt by one character is a cache miss. That is intended.
- A cached response records the original call's latency. Latency benchmarks must report cached and uncached runs separately.
