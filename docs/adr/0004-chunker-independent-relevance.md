# ADR 0004: Gold evidence is a quote, not a chunk ID

- Status: Accepted (owner sign-off 2026-10-30)
- Date: 2026-10-26

## Context
The ablation ladder changes chunking (A0 fixed windows → A1 structure-aware parent-child). If gold labels were chunk IDs, every chunker change would invalidate the golden set, and A0 vs A1 could not be compared.

## Decision
- Each golden query stores evidence as `(doc_id, quote)`, with the quote copied verbatim from the filing.
- At scoring time the quote is located in the parsed document text, ignoring case, whitespace and curly-vs-straight quotes or dashes (`metrics.locate_quote`). That gives a character span.
- A retrieved chunk (also a character span) **hits** an evidence item when the overlap is at least 50% of the evidence span, or at least 50% of the chunk.
- Each evidence item can be found once. Later chunks hitting the same item earn nothing, so recall and nDCG stay in [0, 1] and overlapping windows cannot inflate scores.
- nDCG@10 uses binary gain with ideal DCG = min(10, number of evidence items).
- Confidence intervals: percentile bootstrap over queries, 2000 resamples, seed 0. Config-vs-config comparisons use a paired bootstrap over the same queries.

## Alternatives considered
- Chunk-ID labels: simplest, but tied to one chunker.
- Exact quote containment in the chunk: fails when a quote straddles a chunk boundary.
- LLM-judged relevance: not deterministic, and it costs calls. That rules it out for the CI gate.

## Consequences
- A quote that cannot be located (e.g. the LLM paraphrased it) is a labelling error. The scorer reports these, and review must fix them.
- 50% is a judgement call. A sensitivity check at 30% and 70% was run and dropped: gold quotes are single sentences that always sit inside one chunk, so all three thresholds gave the same scores (docs/ai_review_log.md, 2026-10-27). In practice the rule means "the chunk contains the quote".
