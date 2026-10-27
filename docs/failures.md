# Failures and negative results

Things that did not work, or did not help as much as expected. Negative results are part of the deliverable.


## Parser: Intel filings are unstructured (2026-10-27)
Intel's 10-K and 10-Qs use their own section headings ("Risk Factors and Other Key Information", repeated as page headers) and put Item numbers only in a cross-reference index at the end. The Item-heading parser cannot map them. A title-based fallback first labelled 98% of each Intel filing as `risk_factors`, which is wrong. The parser now marks a filing `unknown` when one section would cover more than 80% of it.
- Impact: 4 of 156 filings (2.6%). Their text is fully indexed and retrievable (A0 ignores sections), but they are excluded from section-based golden-candidate sampling and get no section metadata (relevant for A1).
- Possible fix later: parse Intel's cross-reference index to map its own headings to Items.

## Reranker hurts two-company questions (2026-10-27)
With `bge-reranker-base` over the top 50 hybrid hits, single-company recall@10 rises (0.846 → 0.923) but multi-hop recall@10 falls (0.675 → 0.544, paired Δ −0.132 [−0.237, −0.026]); overall it is flat. A cross-encoder scores one question against one passage. For "How do Lam Research and KLA each describe Huawei?" it prefers passages that fit the whole question loosely over the two passages that each answer half. Without chunk headers it was worse still: it ranked generic "Lam Research Corporation is a global supplier…" overview passages first because they were the only ones naming the company.
- Fix planned: decompose multi-hop questions per company (router in Phase 4, Planner in Phase 5) and rerank each sub-question. Until then the reranker should run only on single-company questions.

## Equal-weight hybrid below BM25 alone on A0 chunks (2026-10-27)
On A0's 300-word chunks, RRF of dense + BM25 (recall@10 0.374) was worse than BM25 alone (0.432, Δ −0.057 [−0.110, −0.007]), because the weaker dense list pushed good BM25 hits out of the top 10. With A1 chunks plus headers the dense side improved enough that hybrid wins (0.704 vs 0.612).
