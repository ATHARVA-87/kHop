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

## Extraction prompt v6 and gleaning did not help (2026-10-28)
Scored on 40 gold-labelled 10-K passages (187 relations; AI labels, human confirmation pending), relation-family level:

| variant | precision | recall | company-company P / R |
| --- | --- | --- | --- |
| v5 (current) | 0.67 | 0.18 | 0.90 / 0.45 |
| v6: worked examples + `licenses_to` | 0.32 | 0.14 | 0.46 / 0.25 |
| v5 + one gleaning pass (GraphRAG `max_gleanings`) | 0.45 | 0.32 | 0.56 / 0.50 |

- v6's examples made the 4B model list more generic objects ("applicable executive orders") and repeat edges. Worse on every measure.
- Gleaning raised recall mainly for products, segments and exposures, but company-company precision fell from 0.90 to 0.56; false company edges create false bridges for A4. Not adopted.
- Kept: v5. Known v5 error class: `licenses_from` written where the filer licenses TO the object (5 of 15 bad edges in the quote calibration). Fixing it needs a stronger model or a rule, not more prompting.
- Sample is small (20 company-company gold relations); CIs in `results/extraction_*.json`.

## HippoRAG-style Personalized PageRank lost to simple edge rules (2026-10-28)
PPR over an entity + evidence-chunk graph (seeded with the question's entities, fused with A2 by RRF), dev multi-hop complete@10:

| variant | all multi-hop | bridge (14) | comparison (43) |
| --- | --- | --- | --- |
| A2-RR (no graph) | 0.597 | 0.357 | 0.674 |
| **A4 edge rules** | **0.772** | **0.643** | **0.814** |
| PPR, all edges, damping 0.5 or 0.85 | 0.596 | 0.571 | 0.605 |
| PPR, company edges only | 0.649 | 0.714 | 0.628 |
| graph weight 2 in RRF (any PPR variant) | 0.54–0.56 | | |

PageRank mass leaks through hub nodes (NVIDIA has 126 edges; 1,041 exposure edges join most companies to shared risk nodes). The simple rules (edges between the named entities, or their common neighbours for bridge questions) are better and easier to explain. Removed PPR and weighted RRF; kept plain RRF.

## Per-company sub-queries did not beat the graph (2026-10-28)
For questions naming 2+ filers: the hybrid search was run once per company, restricted to that company's filings, and the results interleaved (decomposition without an LLM; the research scout cited IRCoT / Self-Ask / least-to-most). Dev multi-hop complete@10: per-company search alone 0.561 (below the no-graph round-robin 0.597; bridge 0.214); graph alone (A6) 0.789; graph + per-company 0.807 (+0.018 [−0.059, +0.111], noise). The graph already retrieves the right passages for these questions. Removed; the per-filing filter is kept for the A7 corrective step ("a named company is missing: search its filings").
