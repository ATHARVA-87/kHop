# Architecture Decision Records (ADRs)

An **ADR (Architecture Decision Record)** is a short note that records one important design choice. Each note covers four things:
- the problem (context);
- what was decided;
- the alternatives that were considered;
- the consequences.

Every ADR here was signed off by the project owner. If a decision changes, a new ADR replaces the old one rather than editing history.

| ADR | Decision |
| --- | --- |
| [0001](0001-llm-wrapper-and-disk-cache.md) | One LLM wrapper with a SQLite disk cache and an offline mode |
| [0002](0002-environment-and-dependency-split.md) | Reuse the `torch` conda env; split core and ML dependencies |
| [0003](0003-corpus-snapshot.md) | Corpus = 40 US semiconductor filers, 10-K + 10-Q filed in calendar 2025 |
| [0004](0004-chunker-independent-relevance.md) | Gold evidence is a quote, not a chunk ID |
| [0005](0005-llm-providers-and-models.md) | LLM providers and models (free tiers checked 2026-10-27) |
| [0006](0006-graph-store.md) | Graph store (LadybugDB instead of Kuzu) |
| [0007](0007-phase5-baseline-is-a6.md) | The Phase 5 comparison baseline is A6, not A7 |
| [0008](0008-web-app-static-nextjs.md) | The web app is a static Next.js export served by FastAPI |
| [0009](0009-rule-router.md) | A rule router, not an LLM planner, picks the search route |
| [0010](0010-guardrails-and-fact-checker.md) | Rules for refusals, datamarking for passages, a small fact-checker for answers |
| [0011](0011-semantic-cache.md) | Semantic cache keyed by meaning AND by the companies and periods asked about |
| [0012](0012-spelling-layer.md) | A plain-code spelling layer that asks before it changes anything |
| [0013](0013-demo-hosting.md) | Where the public demo runs |
