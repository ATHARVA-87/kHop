# ADR 0008: The Phase 5 comparison baseline is A6, not A7

Status: **proposed, needs the owner's sign-off** (2026-10-29)

## Context
The brief's Phase 5 gate is "A9 (multi-agent) vs A7 (single graph with grade-and-rewrite), with latency and cost". A7 was never built:
- Phase 4 measured the fixes A7 would make and none helped (docs/failures.md: per-company sub-queries, the A7 company-coverage fix, 10 writer passages).
- With 44/49 dev answers fully correct (human labels), a rewrite loop had too few failures to show a measurable gain (review 2026-10-29).
- Research (Ferrazzi et al. 2026) found extra retrieval/refinement rounds give nothing over a tuned single pipeline with a reranker.

## Decision
Compare A9 against **A6** (rule router + per-type retrieval + writer), the best measured single-graph system. A9 = A6 + input gate + datamarking + verifier marks. Report: guardrail metrics (tuning and held-out sets), claim support (validated MiniCheck, cited sentences only), cold latency (p50/p95) and LLM calls/tokens per question.

## Consequences
- The brief's A7 rung is recorded as "not built, measured unnecessary"; it can be added later if new failure data appears.
- A9 adds no LLM calls over A6 (the verifier is a 770M local model), so the cost comparison is latency, not money.
