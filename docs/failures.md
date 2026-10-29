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

## The Gemini judge is too lenient (2026-10-28)
Agreement with 49 blind human correctness labels (38 correct / 10 partially correct / 1 incorrect) on dev A6 answers:

| judge setup | raw agreement | Cohen's kappa |
| --- | --- | --- |
| original prompt (gemini-3.5-flash-lite in practice; 3.6-flash overloaded) | 0.776 | 0.064 |
| prompt with the human's exact rubric (still flash-lite: 3.6-flash did not answer) | 0.755 | 0.133 |

The judge rated 7–9 of the 10 human "partially correct" answers (one of two named companies answered, or "not enough evidence" for half the question) as correct. Its correctness scores (0.94–0.97) are therefore NOT reported as results. On this sample the human labels give 38/49 = 0.78 fully correct. Next: RAGAS claim-level metrics in a separate env, scored against the same labels.

### Checklist judge: better, still not accepted (2026-10-28)
Research-recommended fix (CheckEval / TICK style): list the required parts of an answer, check each part ANSWERED / NOT_ANSWERED / CONTRADICTS, correct only if all are answered. Prompts and the acceptance bar (catch >= 70% of human not-correct answers AND pass >= 85% of human-correct answers) were committed before the run (f0198f9); one run, Groq `openai/gpt-oss-120b`, all 49 labels (`results/checklist_judge_a6_router.json`).

| judge (binary: correct vs not) | catches not-correct (TPR, 95% Wilson) | passes correct (TNR) | binary kappa | Gwet AC1 |
| --- | --- | --- | --- | --- |
| single-verdict Gemini judge | 3/11 = 0.27 [0.10, 0.57] | 0.97 | 0.32 | 0.75 |
| checklist judge | 4/11 = 0.36 [0.15, 0.65] | 0.95 | 0.37 | 0.74 |

Not accepted: it still passed 7 of the 10 "partially correct" answers (4 local, 3 multi-hop). Per the protocol the prompts are not tuned on these labels. Headline answer quality stays the human number: 38/49 = 0.78 fully correct (Wilson 0.64–0.87). Automated judges are reported only as "agreement with human", never as answer accuracy.

## A7 company-coverage fix did not help (2026-10-28)
Deterministic corrective step: if a company the question names had no passage in the writer's top 5, search that company's filings only and insert its best passage. Dev: 20 of 57 multi-hop questions lack both gold passages in the top 5, but the fix triggered on only 8 (the named company was usually present, with the wrong passage), and multi-hop complete@5 stayed 0.649 (top-10 +0.018, n.s.). Removed. The gap is relevance within a company, not coverage: complete@10 is 0.789 vs complete@5 0.649, so showing the writer 10 passages instead of 5 is the simpler lever (to be measured with a validated judge).

## Showing the writer 10 passages instead of 5: no measurable gain (2026-10-29)
Blind paired human labels on the 49 labelled dev questions (46 differing answer pairs, random order, hidden key; rule fixed before unblinding, commit d5680c8). `results/paired_a6_router_ctx10_vs_a6_router*.json`.

| run | 5 passages correct | 10 passages correct | only 10 correct / only 5 correct | exact McNemar p |
| --- | --- | --- | --- | --- |
| primary (human verdicts as given) | 44/49 = 0.90 | 45/49 = 0.92 | 2 / 1 | 1.00 |
| sensitivity (4 stray "Not enough evidence" cells penalised: the human named 5, one was already partial) | 40/49 = 0.82 | 45/49 = 0.92 | 6 / 1 | 0.125 |

Stray "Not enough evidence" lines in otherwise answered questions: 6 (5 passages) vs 4 (10 passages). All 4 penalised cells are 5-passage answers; the 4 ten-passage answers with a stray line were already labelled partial or incorrect, so penalising cannot lower the 10-passage side. Neither run is significant; 5 passages stays the default (shorter prompt, nothing proven). The better complete@10 retrieval did not turn into measurably better answers on 49 questions.

Label drift found: on re-labelling the SAME 5-passage answers blind, 6 answers first labelled partially_correct were now labelled correct (L-8ab362f0e3, L-950416d079, L-99426bab62, L-a8da0ea627, M-74be6129e9, M-7531785032); four of them are exactly the "partial" answers both judges passed. Human test-retest agreement on the 46 re-labelled answers: 40/46 = 0.87 (the 3 identical answers were not re-labelled). All 6 changes went the same way, and the re-label came after the judge results were known. The judge validation above uses the first labels; it is not re-scored until the human decides which label is right.

### Judges re-scored with the human's updated labels (2026-10-29)
The human chose the newer labels for the 6 drifted answers (all partially_correct -> correct; `data/golden/judge_labels.csv` now 44 correct / 4 partially correct / 1 incorrect). Same cached judge outputs, no new LLM calls:

| judge (binary) | catches not-correct (TPR, 95% Wilson) | passes correct (TNR) | binary kappa | Gwet AC1 |
| --- | --- | --- | --- | --- |
| single-verdict Gemini judge | 3/5 = 0.60 [0.23, 0.88] | 0.98 | 0.63 | 0.93 |
| checklist judge | 3/5 = 0.60 [0.23, 0.88] | 0.93 | 0.49 | 0.87 |

Still below the 0.70 bar (checklist judge not accepted), and with only 5 not-correct answers the TPR interval is too wide to say anything either way: the validation is underpowered, not just failed. Much of the earlier "judge too lenient" finding was label drift on borderline one-company-missing answers. RAGAS: its 14 scored rows are now all human-correct, so it cannot be compared with humans on them (F1 still marks 12 of those 14 as not correct: too harsh). Headline answer quality: human 44/49 = 0.90 (Wilson 0.78–0.96) under the updated labels; 38/49 = 0.78 (0.64–0.87) under the first labels. Report both: the change is a labelling decision, not a better system. The single-verdict judge now agrees with the human better than the checklist judge (kappa 0.63 vs 0.49), so the checklist judge is not used further. The rule behind the 6 changes (when is an answer that misses part of a question still "correct"?) is to be written into the labelling rubric by the human.

## Verifier validated; the number check added nothing (2026-10-29)
MiniCheck-Flan-T5-Large vs 60 blind human labels (50 real cited sentences from A6 answers + 10 swapped negatives; `results/verifier_validation.json`). The human marked 15 unsupported (all 10 swapped + 5 real). MiniCheck at its default 0.5: catches 15/15 (Wilson 0.80–1.00), passes 42/45 supported (0.93), kappa 0.88, AUROC 0.999: **accepted** against the pre-set bar (0.70 / 0.85). Adding "every number in the sentence appears in the source" changed nothing on these pairs, so it was removed. Side finding: 5 of 50 real cited sentences (10%) are not supported by the passage they cite.

## Datamarking: kept (2026-10-29)
Spotlighting by datamarking (Hines et al. 2024): `^` between the words of retrieved text plus one prompt rule. Keep rule set before the run (commit 1660f48): planted-attack success must drop and fully-supported dev answers must not drop. Result (`results/guardrails_a9_datamark.json`, `results/support_a6_router_vs_a9_datamark.json`): planted attacks 3/30 -> 0/30; fully-supported answers (validated MiniCheck) 0.265 -> 0.336, paired +0.076 [+0.008, +0.152]; uncited sentences 0.293 -> 0.233; no `^` in any answer; refusals unchanged. Unanswerable declined 19 -> 18 by the strict rule, but both misses (U-01, U-04) do say the fact is not in the filings in their own words. Kept.

## Verifier replacing answers: rejected; it only marks sentences (2026-10-29)
Planned rule: if no cited sentence of an answer is supported, replace the answer with "not enough evidence". Keep rule (non-circular, uses existing human labels): it must not blank any answer the human marked correct. On the 49 labelled A6 answers it would blank B-bf5d25e90c, which the human marked correct (4 of 134 dev answers would be blanked). Rejected. The verifier now only marks each sentence supported / not supported (validated: 15/15 unsupported caught, 42/45 supported passed) for the reader; it never changes the answer. Deleting unsupported sentences was not tried: it would be scored by the same verifier (circular).

### Verifier switched to MiniCheck-RoBERTa-Large; review fixes re-validated (2026-10-29)
A9 with Flan-T5 MiniCheck ran out of GPU memory (6 GB card: writer 3.1 GB + search models + 1.5 GB verifier). MiniCheck-RoBERTa-Large (same authors, MIT, 355M, ~0.7 GB fp16; pinned 74c8919, converted to safetensors like the T5 one) was validated on the same 60 blind labels. After the review fixes (sentence-aware chunks, abbreviation-safe splitter) both models give identical verdicts: catch 15/15 unsupported (Wilson 0.80–1.00), pass 41/45 supported (0.91), kappa 0.84; on the 50 real sentences only: catch 5/5 (0.57–1.00), pass 41/45, kappa 0.67 (the 10 swapped pairs were easy). RoBERTa is now the default (`results/verifier_validation.json`; T5: `verifier_validation_flan_t5.json`); the T5 weights were deleted. Earlier "15/15, 42/45, kappa 0.88" was before the fixes and counted the easy swapped pairs.

### Phase 5 re-measured after the review (2026-10-29)
- **Advice rules overfit.** On 30 fresh held-out advice questions (written before the rule fixes, `refusal_heldout.csv`) the gate's rules alone catch 4/30. The whole system still declines 27/30 (A9) vs 28/30 (A6) because the writer's prompt already says "no investment advice": the gate adds a clear message, not coverage. Look-alikes answered: A9 28/30, A6 26/30. No answer gave a buy/sell opinion. (H-I01/H-I02 and H-A01–05 came from the review; reported in the same set.)
- **Datamarking does not make cited facts more accurate.** Cited-sentence support (validated MiniCheck-RoBERTa): A6 0.820, A9 0.833, paired per-answer +0.034 [−0.013, +0.080], not significant. "All sentences supported" rises 0.295 -> 0.373 (+0.083 [+0.008, +0.159]) mainly because fewer sentences lack a citation (0.263 -> 0.215) and answers are slightly shorter (4.98 -> 4.62 sentences). The earlier "+0.076" claim measured citation format more than faithfulness.
- **Injection, 10 templates (v2, incl. adaptive ones):** A6 4/30 (Wilson 0.05–0.30; fake end-of-excerpt + system turn 3/3, French 1/3), A9 0/30 (0–0.11). Exact McNemar on the 4 discordant pairs p = 0.125: consistent with datamarking blocking these attacks, not proof.
- **Unanswerable:** A6 19/20, A9 18/20 by the strict rule; all misses decline in their own words.

### Held-out refusal set checked by the owner (2026-10-29)
58/60 labels confirmed; H-A11 and H-A23 ("upside" questions) moved to "answer with filing facts, no opinion", like the earlier price-target ones. Rescored from saved flags (no re-generation): advice declined A6 26/28, A9 25/28; should-answer answered A6 26/32, A9 28/32; no buy/sell opinion in any answer. Both systems decline the two "upside" questions (the writer's no-advice rule), so answering valuation-style questions with facts remains open.

## Found by using the app: single-company questions searched every filing (2026-10-29, kept fix)
Asking the live app "Which foundry does NVIDIA rely on to make its chips?" returned "not enough evidence": the local route (hybrid + reranker over all filings) ranked four MaxLinear passages about Taiwan foundries above NVIDIA's own. The golden questions never showed it because they are worded close to one filing. Fix: when exactly one corpus company is named, search only its filings (`retrieval.local_company_filter`). Keep rule set before measuring (commit before results): dev recall@10 not lower. Result (`results/a6_filter.json` vs A6): all +0.015 [0.000, +0.041], local +0.031 [0.000, +0.077], multi-hop and global unchanged; the NVIDIA question now gets NVIDIA's supplier-dependency passages. Kept in `configs/app.yaml` (the app); the Phase 5 configs are unchanged so their results stay reproducible.

## A8 semantic cache: kept, and the company/period key is what makes it useful (2026-10-29)
Test set: 40 dev questions x (1 rewording + 1 near-miss: other company 13, partner 7, period 2, metric 3, topic 15), assistant-drafted, all 80 labels confirmed by the owner. Rule fixed before the run (commit 3bc37b3): threshold = lowest with zero wrong reuses on pairs 1-20, reported once on pairs 21-40. Result (`results/a8_cache.json`): threshold 0.935; test half reuses 10/20 rewordings with 0/20 wrong reuses (rule-of-three upper bound 15%). The pre-set no-key ablation on the test half showed no difference, because the near-misses that come closest all fell in the tune half. Follow-up over all 80 pairs (post-hoc, labelled): the five most similar near-misses (0.936-0.973; "quarter ended June 30" vs "September 30" scores 0.973, above every rewording) are blocked only by the company/period key; without the key a safe threshold must exceed 0.973 and reuses 3/40 rewordings, with the key 23/40. Live check: close rewordings served in 12-43 ms vs ~10 s fresh; a looser rewording (similarity 0.859) is answered fresh. Only answers with every sentence fact-checked are stored.

## The app overfilled the 6 GB GPU; fixed by load order, not by moving the verifier (2026-10-29)
In the load test Ollama ran the writer 35% on the CPU (slow answers, laptop lag). Moving the fact-checker to the CPU (owner's choice) did not fix it on its own: Ollama's log shows qwen3.5:4b also reserves up to 892 MB for an image encoder we never use, so loaded after the app's search models (1.8 GB) it still spilled. Fix: the app loads the writer FIRST (keep-alive while the app runs, unloaded on shutdown; a runtime setting outside the cache key), shares one bge-small copy between search and cache, and runs the fact-checker on the CPU (4 threads, `guard.verifier_device`). Result: writer 100% on the GPU, 5,513 / 6,141 MB used (inside the owner's 10% margin), writer 3.8-4.4 s instead of 10-21 s, CPU load ~0.9 on 16 cores; stopping the app frees the GPU (12 MB).

## Reframe layer: kept (2026-10-29, owner request)
Owner found that "How are the nvidia company / nvidia companies / nvdia companies performed in 2024" gave different answers: the typo hid NVIDIA (sector-wide search instead of NVIDIA's filings) and small wording changes moved the top passages. Built after research, test sets and keep rules frozen first (`data/golden/reframe/PROTOCOL.md`): plain-code spelling correction against the filings' own vocabulary and company names (no LLM), official company names, no word reordering; the page shows "Did you mean this?" with "Use this version" / "Ask as I typed" before anything is searched. v1 FAILED its bars (changed 8 of 225 correct questions: possessive apostrophes; restored only 42/57 company typos: mixed-case words skipped, top-5 candidate cutoff); general fixes, then judged on a FRESH typo set it never saw (`results/reframe.json`): 113/113 company names and 114/125 words restored, 0 wrong replacements, 0 changes to the 225 golden questions and 17 trap words, 0.4 ms. A bug where "inc" matched "include" was caught and fixed. Variants of the same question: same route 78.5% -> 100%, median top-5 overlap 0.67 -> 1.00 (company typos 0.18 -> 1.00). Owner-checked rewordings are unchanged (0.48): the layer does not paraphrase; a filing-language phrase map is the next, optional experiment.
