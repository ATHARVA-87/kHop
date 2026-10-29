# Reframe layer: keep rules (fixed 2026-10-29, before the layer was written)

Owner request: one canonical question in the filings' own wording, so equivalent questions get the same answer; the user sees the rewritten question first and chooses "Use this version" or "Ask as I typed" (nothing is searched until they choose).

Step 2, spelling (kept only if ALL hold):
- 0 changes on all 225 golden questions (134 dev + 91 test) and on every trap word in traps.json;
- company names restored on typos.jsonl >= 90%; other words restored >= 70%;
- wrong replacements (changed to a different word) <= 3%;
- p95 time < 20 ms per question.

Step 3, company-name canonicalisation (kept only if): typo / lowercase / plural variants get the same route as the clean question >= 95% (consistency.jsonl); median top-5 passage overlap (Jaccard) with the clean question rises; golden recall@10 drops at most 0.5 points on dev and test.

Step 6, optional filing-language phrase map (<= 20 hand-written phrases): kept only if dev recall@10 rises >= 1 point and test does not fall. Expected to fail (Kotte 2026, arXiv:2603.13301: prompt rewriting cut nDCG on finance data).

Not built: LLM rewriting (cost, nondeterminism, invented facts), word reordering, corrections of capitals / numbers / tokens of 3 characters or fewer.
