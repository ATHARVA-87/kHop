# ADR 0009: A rule router, not an LLM planner, picks the search route

Status: Proposed (awaiting owner sign-off)
Date: 2026-10-29

## Context
Three kinds of question need different search: one company (local), two or more companies (multi-hop), and sector-wide (global). Each route was measured on its own in Phases 2–3. The reranker helps local questions but hurt multi-hop ones, so one pipeline for all questions loses on some kind.

## Decision
Route by counting the corpus companies the entity linker finds in the question:
- 0 companies: global (company-diverse ranking, A5);
- 2 or more: multi-hop (hybrid + graph links, A4);
- otherwise: local (hybrid + reranker).

The app adds one more step on the local route: search only that company's filings (`configs/app.yaml`, measured in `results/a6_filter.json`).

No LLM call. Results: `results/a6_router.json` (dev), `results/test/a6_router.json` (held out), and the always-local, always-multi-hop, always-global and oracle runs next to them.

## Alternatives considered
- **LLM planner that classifies the question.** It costs a call, adds latency, and is not deterministic. On our question sets the rules match the oracle route on dev and nearly on test. Partly this is by construction: the golden questions separate cleanly by how many companies they name, and real users' questions will not always. A planner might help there; that is untested.
- **One pipeline for every question.** Measured as the always-X baselines. Each one loses on at least one question type.

## Consequences
- The router is only as good as the entity linker. A misspelled company name sends a question to the wrong route, which is why the spelling layer (ADR 0012) exists.
- Honest reading: on single-company questions the router's gain over A5 is mostly "the reranker now runs there". The router is the explainable switch that makes this possible.
