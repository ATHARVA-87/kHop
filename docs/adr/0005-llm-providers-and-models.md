# ADR 0005: LLM providers and models (free tiers checked 2026-10-27)

- Status: Accepted (owner sign-off 2026-10-30)
- Date checked: 2026-10-27 (IST), by Claude Code using the official pages below.

## Findings
**Google Gemini API** ([models](https://ai.google.dev/gemini-api/docs/models), updated 2026-10-22; [terms](https://ai.google.dev/gemini-api/terms), updated 2026-04-28)
- `gemini-2.0-flash` is **shut down**. The first draft of the configs used it, and they have been corrected.
- Current stable: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`. `gemini-2.5-flash` is served only to past users.
- Free-tier rate limits are **not published** on the [rate-limit page](https://ai.google.dev/gemini-api/docs/rate-limits) (updated 2026-10-15). Google says to view them in AI Studio. Third-party blogs disagree (roughly 5–15 RPM, 20–1,500 RPD). Check AI Studio after creating the key.
- Terms for unpaid services: prompts and outputs may be used to improve Google products and may be read by human reviewers. Do not submit sensitive or personal data. You must be 18 or older. Only public SEC text goes there, which is acceptable.

**Groq** ([rate limits](https://console.groq.com/docs/rate-limits), no date shown)
- The public table lists Developer-plan base limits. For example, `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` are at 30 RPM, 1K RPD, 8K TPM and 200K TPD. Free-plan limits are only shown in the account's limits page. Llama models no longer appear.
- Adding a card moves the account to a paid tier. **Do not add a card** (rule 1: free only).

**Ollama, local** (RTX 4050, 6 GB VRAM)
- `qwen3.5:4b` (about 3.4 GB, tools and thinking support) is the most-recommended model that fits 6 GB with room for context. The fallback is `phi4-mini`. Thinking mode must be off for JSON agents (to be verified in Phase 3).

## Decision
| Use | Provider / model | Why |
| --- | --- | --- |
| Golden candidates, LLM judge | `gemini/gemini-3.6-flash` (see update below) | Strongest free option. Low call volume (about 200 candidates, about 300 judge calls), all cached |
| A0 generation, small agents (Planner, Researcher, Verifier) | `ollama/qwen3.5:4b` | Local, free, no rate limit |
| Writer, extraction (Phase 3+) | Gemini Flash, or Groq `openai/gpt-oss-120b` as the fallback | Stronger free option; decide by measured quality |

## Update 2026-10-27 (first live calls)
- `gemini-3.8-flash` timed out and returned 503 "high demand" on repeated tries. `gemini-3.6-flash` answered in about 4 s, `gemini-3.5-flash-lite` in about 1.5 s. **Default is now `gemini-3.6-flash`**, with `gemini-3.5-flash-lite` as the fallback.
- Gemini 3 models spend output tokens on hidden reasoning (about 90 tokens to answer "OK"). `max_tokens` for candidates and the judge is raised to 2048. Empty responses are now errors and are never cached.

## Consequences
- Daily caps make the disk cache essential. A capped run resumes the next day from the cache.
- Judge and generation should not share one model, to limit self-preference bias. Here they differ (Gemini vs Qwen).

## Update 2026-10-28 (Phase 3 extraction)
- Graph extraction used `ollama/qwen3.5:4b`, Ollama model digest `d8b0f5e9760c` (from `ollama list`), with `think: false`, temperature 0, seed 0, prompt v5. The raw output is published as release asset `ci-graph-semis-2025` (checksums in `data/snapshots/semis-2025.ci_graph.json`), so the graph can be rebuilt without a GPU or LLM.
