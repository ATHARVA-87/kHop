# Plexus — progress

## Phases

| Phase | Work | Gate | Status |
| --- | --- | --- | --- |
| 0. Setup | Toolchain, repo, CLAUDE.md, PROGRESS.md, Makefile, `make doctor`, CI skeleton, LLM wrapper + cache | `make doctor` and CI pass | **Passed 2026-10-27.** Doctor phase 0 all OK; GitHub CI green (run 40440596122) |
| 1. Harness | EDGAR snapshot, golden-set candidates, human verification, metrics, A0 | A0 scored with CIs | **Gate met 2026-10-27** (134 verified queries; A0 recall@10 0.276). Branch `phase1-harness`, PR #1 needs your review |
| 2. Retrieval + CI gate | A1–A3, retrieval gate on PRs | A1–A3 recorded; gate blocks a planted regression | **Gate met 2026-10-27.** A2 hybrid + headers recall@10 0.704 (A0 0.276); CI gate blocks planted regression (run 40527190366). Branch `phase2-retrieval`, PR #2 needs your review |
| 3. Graph | Extraction, resolution, Kuzu, Leiden, summaries, A4, A5 | Extraction P/R; A4, A5 recorded | — |
| 4. Single-graph agent | Router, grade-and-rewrite, A6, A7 | Router vs always-local/global | **Gate met 2026-10-28** (A6 router beats always-X on dev and test; = oracle). A7 loop in progress |
| 5. Multi-agent | 4 agents, guardrails, A9 | A9 vs A7 with latency/cost; guardrail metrics | — |
| 6. Systems + UI | Semantic cache (A8), SSE, Phoenix, Next.js, Locust | p95, cache hit rate, cost/query | — |
| 7. Ship | README, ADRs, demo Space, drift check, demo script | Definition of done | — |

## Resume here

**State (2026-10-28, paused for the night):** Phases 0–3 done (PR #5 for Phase 3 open, you merge). Phase 4 on branch `phase4-router` (pushed, clean). Phase 4 gate met: the A6 rule router beats always-X on dev and test and equals the oracle. Honest reading: the gain is the reranker on single-company questions.

**Waiting on you:** label `data/golden/judge_labels.csv`, 49 answers (blind, stratified). Fill `human_verdict` with `correct` / `partially_correct` / `incorrect`, save it in place, then say "labels done".

**Next Claude steps, in order (agreed plan):**
1. `python -m plexus.eval.judge agreement --results results/a6_router.json` gives kappa and the confusion matrix. If the judge is lenient (it rated two half-answers correct), make it stricter (gemini-3.6-flash when not overloaded, or a stricter prompt), re-measure, and report both.
2. A7 grade-and-rewrite: deterministic coverage grade (every named company present? ≥N companies for global?) → corrective search in the missing company's filings (per-filing filter already built) → one yes/no LLM grade → at most 1 rewrite keeping the company names. Compare loop vs no loop (answer metrics, rewrite rate, latency).
3. LangGraph wiring last: pinned version ≥ 2 weeks old, in-memory state only, no CLI/server (security advisories noted in the research report).
4. A reviewer agent after each step; delete what loses (keep it explainable).

## Gate evidence

### Phase 0 (2026-10-26)
- `make doctor`: all phase-0 checks pass, GitHub remote included (2026-10-27).
- GitHub CI run 40440596122 on `phase1-harness`: 77 packages from `uv.lock` (no torch), ruff clean, 89 passed / 3 ml deselected.
- Note: GitHub did not fire the push and pull_request events for the first pushes to the new repo. `ci.yml` now also has `workflow_dispatch`. If PR #1 shows no checks, run `gh workflow run ci --ref phase1-harness`.
- `make test`: passes. `make lint`: clean.
- CI dry run: fresh venv from `uv.lock` (core + dev, no torch), ruff clean, 66 passed / 3 ml deselected. Done 2026-10-26 23:54, environment deleted afterwards.

### Phase 1: gate met 2026-10-27 (retrieval). `results/a0_naive.json`, 134 queries, 0 label errors, 95% bootstrap CIs
| slice | n | recall@10 | nDCG@10 | MRR |
| --- | --- | --- | --- | --- |
| all | 134 | 0.276 (0.208–0.343) | 0.211 | 0.238 |
| local | 65 | 0.408 (0.292–0.523) | 0.286 | 0.259 |
| multi_hop | 57 | 0.175 (0.114–0.246) | 0.160 | 0.240 |
| global (company_coverage@10) | 12 | 0.508 (0.417–0.617) | | |
- strict_recall@10 (exact source filing only) 0.204. Retrieval p50 46.46 ms.
- Generation metrics not run yet: they need Ollama (sudo install), so those are deferred and not part of the gate.
- Global questions were reworded by you: "How do all semiconductor companies …" instead of "… in the corpus". All 122 local/multi-hop/bridge questions were then rephrased the way a user would type them (short company names, no filing dates unless the answer depends on the period; your request 2026-10-27). Same answers and evidence. A0 recall@10 rose from 0.216 to 0.276: the legal names and filing dates were noise for the dense embedder.

### Phase 2: gate met 2026-10-27 (branch `phase2-retrieval`, PR #2 stacked on #1)
- **Gate evidence.** CI job `retrieval-gate` (BM25 path, no torch) passes on the branch: run 40526855835. A planted regression (S-stemmer disabled, throwaway branch `gate-demo-planted-regression`, now deleted) fails it: run 40527190366. Failing metrics: all recall@10 0.5808 < 0.6119, local 0.7000 < 0.7308, multi-hop 0.5614 < 0.6053. The unit tests also failed (2). `make gate` runs the full gate locally (BM25, hybrid, reranker).
- CI data: release `ci-data-semis-2025` (processed filings, 9.0 MB gz); both checksums are verified by `plexus.ingest.ci_data`.
- Retrieval is fully deterministic: re-running all three gated configs, including the fp16 GPU reranker, reproduced every per-query score exactly.
- A1–A3 recorded (tables below).
Step 1, 2026-10-27: BM25 keyword retriever, reciprocal rank fusion and a retriever interface (`src/plexus/retrieve/`), `make compare` (paired bootstrap B − A). Tested on A0's chunks, so only the retriever changes:

| config | recall@10 (95% CI) | nDCG@10 | MRR | Δ recall@10 vs A0 (paired 95% CI) |
| --- | --- | --- | --- | --- |
| a0_naive (dense) | 0.276 (0.208–0.343) | 0.211 | 0.238 | |
| a0_bm25 | 0.432 | 0.318 | 0.328 | +0.155 [+0.080, +0.231] |
| a0_hybrid (RRF k=60, 100+100) | 0.374 | 0.290 | 0.318 | +0.098; vs bm25 −0.057 [−0.110, −0.007] |

- BM25 alone beats dense bge-small clearly on local and multi-hop questions.
- Negative result: equal-weight RRF is worse than BM25 alone, because the weaker dense list pulls good BM25 hits out of the top 10. Weighted fusion and a reranker (A3) are the next fixes.
- Bias check: the golden questions were written from their passages, which favours keyword matching. On average 40% of a question's words appear in its gold quote. On the high-overlap half BM25 scores 0.60 vs dense 0.32; on the low-overlap half it still wins, 0.34 vs 0.28. So the BM25 advantage is partly an artefact of how the golden set was made, but not only that.
- Global questions: no retriever reaches more than ~55% of the relevant companies in the top 10 (company_coverage@10). That is the job of A5 (community summaries).

Step 2, 2026-10-27: A1 structure-aware parent-child chunks (`structured_chunks`): children are section-bounded and paragraph-aligned, at most 200 words / 1,400 chars; parents group up to 800 words. 31,530 children, 8,347 parents, mean 161 words. **Chunks truncated by the embedder: 10 (A0: 1,031).**
New metric `recall@{B}w`: recall within the first B words retrieved. Chunkers are only comparable at an equal reading budget (A0's top 10 ≈ 3,000 words, A1's ≈ 1,650).

| config | R@500w | R@1000w | R@2000w | R@3000w | R@5000w | R@10 |
| --- | --- | --- | --- | --- | --- | --- |
| a0_naive | 0.128 | 0.199 | 0.245 | 0.276 | 0.325 | 0.276 |
| a0_bm25 | 0.190 | 0.302 | 0.387 | 0.432 | 0.486 | 0.432 |
| a0_hybrid | 0.184 | 0.255 | 0.341 | 0.374 | 0.454 | 0.374 |
| a1_structured | 0.175 | 0.235 | 0.269 | 0.333 | 0.376 | 0.250 |
| a1_bm25 | 0.202 | 0.310 | 0.377 | 0.435 | 0.485 | 0.347 |
| a1_hybrid | 0.190 | 0.261 | 0.378 | 0.438 | 0.512 | 0.356 |

- Paired Δ at 3,000 words, A1 − A0: dense +0.057 [0.000, +0.117]; hybrid +0.064 [+0.006, +0.128]; BM25 +0.004 [−0.050, +0.060].
- A1 helps the embedding-based retrievers (whole paragraphs, no truncation) and does nothing for BM25. At equal top-k it looks worse only because each chunk is about half the size.
- From here the ladder runs on A1 chunks: A2 = a1_hybrid, A3 = A2 + reranker. Primary Phase 2 metric: recall@3000w, with recall@10 alongside.

Step 3, 2026-10-27: A3 cross-encoder reranker (`bge-reranker-base`, fp16, top 50 of A2 = a1_hybrid). Eval run 3.7 min including the model download.
- vs A2: nDCG@10 +0.065 [+0.009, +0.121], MRR +0.072 [+0.006, +0.138]; local recall@10 0.538 → 0.646 (+0.108 [+0.015, +0.185]).
- Multi-hop recall@10 0.219 → 0.140 (−0.079 [−0.175, +0.018], not significant but a warning). It is not company crowding: both companies appear in the top 10 more often after reranking (44/57 vs 40/57).
- Cause, from inspecting the 15 multi-hop losses: the short user-style questions are dominated by company names, so the cross-encoder scores generic "Lam Research Corporation is a global supplier…" overview passages near 1.0. The answer passages say "we", not the company name. Next step: contextual chunk headers (company, form, section) on every chunk, for all retrievers.

Step 4, 2026-10-27: contextual chunk headers (`ingest/headers.py`): "Lam Research Corp (LRCX) 10-K for the period ended 2025-06-29. Section: Risk Factors." is prepended to the text that is embedded, BM25-indexed and reranked. The chunk text and offsets are unchanged, and no gold information is involved. Small alias table for common names (onsemi, TI, Silicon Labs, …).

| config | R@1000w | R@3000w | R@10 | local | multi-hop | nDCG@10 | MRR | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a0_naive (A0) | 0.199 | 0.276 | 0.276 | 0.408 | 0.175 | 0.211 | 0.238 | 30 |
| a1h_structured (A1: chunks + headers, dense) | | 0.640 | 0.541 | 0.769 | 0.386 | 0.442 | 0.471 | |
| a1h_bm25 | 0.504 | 0.693 | 0.612 | 0.731 | 0.605 | 0.461 | 0.454 | 1 |
| **a1h_hybrid (A2)** | **0.623** | **0.778** | **0.704** | 0.846 | **0.675** | 0.560 | 0.563 | 45 |
| a3h_rerank (A3) | 0.602 | 0.770 | 0.684 | **0.923** | 0.544 | **0.582** | **0.613** | 231 |

- Headers alone, hybrid: recall@10 +0.348 [+0.273, +0.423]; multi-hop +0.456. Same pattern with the reranker (+0.309).
- A3 vs A2: local +0.077 [+0.015, +0.139] and nDCG +0.141; multi-hop −0.132 [−0.237, −0.026]. Net flat (see docs/failures.md). Decision: rerank single-company questions only, once the router exists (Phase 4).
- A3h vs A0: recall@10 +0.408 [+0.335, +0.484].
- Global company_coverage@10 stays at ~0.45–0.52 for every retriever. That is A5's job.
- Ladder from here: A1 = a1h_structured, A2 = a1h_hybrid, A3 = a3h_rerank. The configs without headers stay as the header ablation.

## What was built in Phase 1 (branch `phase1-harness`)
| Module | What | Review needed |
| --- | --- | --- |
| `ingest/edgar.py` | Rate-limited (5 req/s), disk-cached, checksummed snapshot downloader | — |
| `ingest/parse.py`, `process.py` | iXBRL HTML → text + Item sections; longest span wins over TOC; parse report | Validate on real data |
| `ingest/chunk.py` | A0 fixed 300-word windows, 50 overlap | — |
| `eval/golden.py` | review.csv → golden.jsonl; only accept/edit scored; noise rate | **Yes** (golden-set code) |
| `eval/metrics.py`, `bootstrap.py` | Quote-located, chunker-independent relevance; recall/nDCG/MRR; bootstrap + paired bootstrap; 30/70% threshold sensitivity | **Yes** (metrics) |
| `eval/candidates.py` | LLM candidate generation (local / multi-hop / global) with automatic quote filter | **Yes** (golden-set code) |
| `eval/judge.py` | Faithfulness + correctness judge; blind export; Cohen's kappa | **Yes** (metrics) |
| `eval/run.py` | Results JSON with CIs per query type, latency, label errors, git commit, checksum | **Yes** |
| `index/dense.py`, `build.py`, `baseline/naive.py` | bge-small + Qdrant local; A0 pipeline with citations | — |

## Planned extras (agreed with you)
- Phase 5: **A9-claude-writer**, a personal local comparison run with the Writer on the optional `claude_sdk` provider vs the default Gemini Writer (same cache, same golden set). Report the quality gap and latency/cost. Never the default, never served to others (brief rule 3).
- Agent model split (ADR 0005): Planner/Researcher/Verifier on local Ollama; Writer, judge and extraction on Gemini (Groq fallback). One account per provider; no multi-account quota stretching.

- Phase 2 (your idea, 2026-10-27): add a **BM25-only baseline** alongside dense-only and hybrid (BM25 + dense + RRF), and a **token-budget curve**: recall@k and answer quality vs number of chunks sent to the Writer (3/5/10), reporting tokens per query. Goal: show that hybrid retrieval reaches the same quality with fewer context tokens (less LLM usage).

## Design review (researcher subagent, 2026-10-27): decisions
Fixed now (Phase 1):
- #1 Same-company verbatim copies of a gold quote count as hits; `strict_recall@10` kept (10-Qs repeat 10-K risk factors, ~75% of spans).
- #2 Global questions are scored at company level: `global_themes.csv` (39 companies × 12 themes, Gemini pre-fill with verified quotes, ~20% human spot-check), `company_coverage@k`, bootstrap resampled by theme.
- #4 21 bridge multi-hop questions with the linking company hidden (`review_bridge.csv`); multi-hop is tagged bridge vs comparison and reported separately.

Planned (by phase):
- #5 Statistical power: with ~40–60 questions per type only gains of about 12–15 recall points are detectable at 80% power. Report pooled results first (per-type exploratory), paired diffs + counts of improved/worse queries, and state the minimum detectable effect in the README. Optionally grow local questions toward 100.
- #6 Self-preference: use a Groq (gpt-oss) judge whenever the Writer is Gemini. Add ~25 human-paraphrased, analyst-style questions as a separate subset (you write them). Add a "metadata filter by company" rung (likely the biggest cheap win).
- #3 Phase 3: community summaries store provenance (source chunk spans). A4/A5 judged mainly by company coverage + answer quality, and "provenance recall" is reported separately.
- #7 Phase 3: extract entities/relations only from 10-K Business/Risk Factors/MD&A (≈1.5M tokens, not the full ≈6.9M). Local qwen for entities, Gemini only for organisation-pair relations. Measure P/R on ~100 edges.
- #8 Phase 5: A9 vs A7 at the same Writer model, tools, top-k and context-token budget. Add an "A7 + single-call decomposition" rung. Measure the Verifier's miss and false-alarm rates. Expect p50 ≈ 20–40 s with a local 4B model (serial on one GPU).
- #9 Scope and pitch: no numeric/table questions (state it). Fiscal years differ (10-Ks filed Jan–Dec 2025). Pitch novelty as controlled ablations under free and small-model constraints with honest negative results (HybridRAG, GraphRAG and GraphRAG-Bench already exist).
- Ladder change: A8 (semantic cache) becomes a systems metric, not a quality rung.

## Golden-set review status (2026-10-27)
- AI pre-review done (3 Claude reviewer agents); suggestions are in the `ai_*` columns.
- **Human review done (2026-10-27).** You read the AI suggestions and agreed with all but one: L-d950b1c9df (Vishay capex), where you kept the original question (the AI edit was unnecessary). Verdicts were filled from `ai_verdict` and `ai_corrected_*`; `reviewer_notes` records which. Themes: 93 spot-check rows set from `ai_discussed`.
- `make golden`: 201 reviewed → **134 verified** (65 local, 57 multi-hop [43 comparison + 14 bridge], 12 global). Verdicts: 84 accept / 50 edit / 67 reject → **candidate label-noise rate 58.2%** (Gemini candidates needing a fix or a reject).
- Human vs AI reviewer: 99.5% exact (200/201), 100% keep-vs-reject. Theme table, human vs Gemini: 95.7% agreement, kappa 0.48 (low kappa because "yes" dominates: 3 Gemini-yes/human-no, 1 the other way).
- Caveat for the README: the human review was AI-assisted (confirm-or-override), so the human/AI agreement number measures that process, not independent labelling.
- Theme table rebuilt with synonyms: 434 yes / 34 no. Gemini vs AI reviewer agreement on 93 spot-check rows: 96% (was 83% with literal keywords). Gemini 3.6-flash was overloaded, so all labels come from 3.5-flash-lite.
- `company_coverage@k` is normalised by min(k, relevant companies).

## Security (user rule 2026-10-27: keys local only, nothing sensitive exposed)
- Audit 2026-10-27: no API keys or tokens in any commit on any branch. `.env` is gitignored and `.env.example` has blanks only. The CI release asset has no keys or emails. GitHub secret scanning and push protection are on. `main` is protected (no force-push or deletion, PRs only, admins included).
- Finding: the author email on the first 51 commits is a personal address, public in commit metadata. Future commits use the GitHub noreply address (repo-local git config). Rewriting history needs your decision.
- New pre-commit hook `no-secrets` blocks API-key and token patterns and personal email addresses.

## Phase 3 progress (2026-10-28)
- Extraction v5, `make extract` (you ran it): 1,665 10-K passages in 91 min (3.3 s each), 0 failures. 3,513 relations proposed, 2,348 kept (67%). Drops: quote not found 588, quote length 202, type mismatch 148, generic entity 103, object not in quote 99. **Only 397 of 1,665 passages yielded any relation**; possible recall problem, to be measured by the P/R labelling.
- Graph (`make graph`): 1,363 entities (292 companies, all 39 filers), 1,992 edges (exposed_to 1,041, competes_with 197, has_segment 194, makes_product 154, operates_in 119, supplied_by 63, sells_to 51, manufactured_by 48, acquired 38). Alias resolution covered 2,818 of 4,696 mentions; 11 fuzzy merges, all logged. A merge of "Country Groups D4/D5" into "D1" was caught in the log and fixed (codes must match).
- **A4 (graph traversal + A2, RRF) on dev** vs A2: multi-hop complete@10 0.509 → **0.667** (+0.158 [+0.058, +0.246]), multi-hop recall@10 +0.140, all recall@10 0.704 → 0.764 (+0.060 [+0.014, +0.101]); local unchanged; global unchanged (A5's job).
- Dev tuning (dev set only): (1) filer short names ("Skyworks", "NXP", "Lattice") were not linked, and mentions of Skyworks in other filings formed a second, disconnected node; fixed by `short_key`. (2) The graph is skipped for single-entity questions: it cost local recall −0.20 on the 10 such questions.

### Research-driven fixes (scout report 2026-10-28; user approved all 5 steps)
- **Step 1, quote grounding:** normalise, split on "...", rapidfuzz `partial_ratio_alignment` (cutoff 95), else an in-order word match for lists with skipped names; the stored citation is the passage's own span. Calibrated on 80 stratified dropped quotes (AI-labelled; human confirmation pending): the alignment score did NOT predict support (almost every quote scoring under 90 was a real edge). Fuzzy ≥95 OR in-order kept 54/63 good edges and let in 9/15 bad (precision 0.86), against 14/63 kept at ≥95 alone. The remaining bad ones are direction/relation-type errors (e.g. `licenses_from` where Rambus licenses TO them): a prompt problem for step 3, not a quote problem.
- **Step 2, resolution:** SEC `company_tickers.json` (10,430 registrants, fetched 2026-10-28, sha256 in `data/reference/*.meta.json`) as a canonical-name layer; `cleanco` suffix stripping; rapidfuzz token_sort_ratio plus "same name except generic words" merging, including into filer, alias and SEC entities. Group/place/nationality names are no longer companies. Not fully closed-world: Hon Hai, Wistron etc. are not SEC registrants.
- Result: relations kept 2,348 → 2,928; edges 1,992 → 2,413; junk company nodes gone; merges logged (Samsung Semiconductor → Samsung, Monolithic Power System → MPS).
- **A4 on dev after steps 1–2:** multi-hop complete@10 0.772 (A2 0.509, +0.263 [+0.100, +0.417]); all recall@10 0.790. Still to compare against the no-graph round-robin baseline (step 4).

- **Step 3, extraction P/R** (40 gold passages, human-confirmed labels; family level): v5 P 0.67 R 0.18, company-company P 0.90 R 0.45. Prompt v6 (worked examples) and gleaning were measured and rejected (docs/failures.md). No re-extraction.
- **Step 4, multi-hop, with the honest baseline (dev complete@10):**

| config | all multi-hop (57) | bridge (14) | comparison (43) | all recall@10 |
| --- | --- | --- | --- | --- |
| A2 hybrid | 0.509 | 0.214 | 0.605 | 0.704 |
| A2-RR (no graph, alternate named filers) | 0.597 | 0.357 | 0.674 | 0.730 |
| **A4 graph edges + A2** | **0.772** | **0.643** | **0.814** | **0.790** |

  A4 vs A2-RR: +0.175 [−0.020, +0.341] (cluster bootstrap by linking entity); comparison +0.140 [0.000, +0.262]; bridge +0.286 (n = 14, wide CI). HippoRAG-style PPR and weighted RRF lost and were removed (user rule: keep it explainable).

- **Step 5, sector-wide questions (A5 = A4 + company-diverse ranking):** when a question names no company, the best passage of each company goes first. Dev global company_coverage@10 0.475 → 0.892. **Honest reading (Phase 3 review):** the base rate for ANY 10 different companies is 0.923, because most themes apply to 32–39 of 39 companies. So A5's gain is diversity: it makes the top 10 span 10 companies. Whether those passages answer the question is left to the Phase 4 answer judge. Deviation from the brief's "A5 community search": Leiden communities give peer groups, not themes (reviewer and research scout).

### Phase 3 review fixes (2026-10-28; user: "fix all 7, then test")
1. A5 coverage reported against its base rate (`coverage_base_rate` in results), and the claim reworded as above.
2. **Alias ablation:** graph and linker rebuilt without the hand-listed aliases of the 17 dev multi-hop linking entities (SEC list kept). Multi-hop complete@10: A2-RR 0.597 / **A4 without those aliases 0.737** / A4 full 0.789; bridge 0.643 either way. About three quarters of the graph's gain does not depend on the hand-listed aliases.
3. SEC gazetteer matches full names only (≥ 2 words, pure legal suffixes removed). This fixed Merck KGaA → MRK, STAr Technologies → Star Group, "Target" → TGT. sha256 checked on load; `python -m plexus.ingest.sec_reference` re-fetches.
4. In-order quote match: one sentence only, no skipped or preceding negation, low density only for lists.
5. Single-word company mentions must be capitalised ("arm's length" no longer links Arm); filer short-name aliases; one shared linker for A2-RR/A4/A5.
6. (see below)
7. Unused igraph/leidenalg removed.
- After fixes: A4/A5 multi-hop complete@10 **0.789** (bridge 0.643, comparison 0.837), all recall@10 0.795.

**How the system works, in 6 steps (for interviews):**
1. Split each filing into sections and passages, and label each passage with company, form and section.
2. Search two ways, by keywords (BM25) and by meaning (embeddings), and merge the rankings (RRF).
3. Offline, a small local LLM lists company relationships in each passage ("AMD is manufactured by TSMC"); a relationship is kept only if its quote is found in the filing.
4. Spellings of a company ("TSMC", "Taiwan Semiconductor") are merged with an alias list and the SEC company list, giving a graph.
5. For a question naming companies: find them in the graph, follow the links between them (or to a shared partner), and add those links' source passages to the results.
6. For a sector-wide question naming no company: put one best passage per company first.

- **CI green on `phase3-graph`: run 40637527821.** `lint-test` passes (graph extra installed) and `retrieval-gate` passes for both a1h_bm25 and a5_ci_bm25. The second rebuilds the graph from the published extractions and gates graph search + A5, with no torch.

## Phase 4 (branch `phase4-router`): router + grade-and-rewrite
Plan agreed 2026-10-28 (research scout: Adaptive-RAG, CRAG, IRCoT/Self-Ask, RAGAS/ARES, LangGraph advisories): route → retrieve (per type) → grade → at most 1 rewrite → generate with citations; LangGraph last, as wiring only.

**Step 1, A6 rule router (gate met).** 0 companies named → global (A5 ranking); 2+ corpus filers → multi-hop (A4 graph); else local (hybrid + reranker, which helps local but hurt multi-hop in Phase 2). No LLM call.

| | router accuracy | all R@10 | all both@10 | local R@10 | multi-hop both@10 | global cov@10 |
| --- | --- | --- | --- | --- | --- | --- |
| dev: always-local / multi-hop / global | | 0.684 / 0.794 / 0.706 | 0.560 / 0.739 / 0.619 | | | |
| dev: **A6 router** (= oracle) | **1.000** (134) | **0.833** | **0.776** | 0.923 | 0.789 | 0.892 |
| test: always-local / multi-hop / global | | 0.532 / 0.563 / 0.518 | 0.462 / 0.494 / 0.418 | | | |
| test: **A6 router** (= oracle) | **0.978** (91) | **0.584** | **0.516** | 1.000 | 0.435 | 0.671 |

- Caveat: the golden questions separate cleanly by how many companies they name; real user questions will not always. Both test misroutes were "TI" not linking, fixed after the test run (docs/ai_review_log.md); the 0.978 is pre-fix.
- **Honest reading (router review, 2026-10-28):** A6 differs from A5 only on single-company questions (dev 5 of 65 changed, test 2 of 37), where the reranker now runs. The gain over A5 is "+reranker on local questions"; the router is the explainable switch that makes it possible. "Router = oracle" holds by construction, because golden questions were generated per type with 0 / 1 / 2+ filers named.
- **Router review fixes:** non-filer short aliases (Amazon, Arm, DOC, BIS) had leaked into the filer set and silently lowered six dev multi-hop rankings. Fixed, and all dev configs re-run at clean commit 35b6498 (multi-hop nDCG@10 0.726 again). Lowercase names now link when the filings capitalise them ≥ 90% ("amd" 0.995, "on" 0.06), and "which companies / chip companies / industry-wide" route to global.
- **Routing on realistic phrasings** (`data/golden/routing_robustness.jsonl`: 30 assistant-written paraphrases of dev questions with lowercase names, tickers, "vs", sector cues): 0.933 before the last two general fixes, 1.000 after. Small and self-written: a diagnostic, not a benchmark. The test set is spent for linker tuning (the "TI" fix came from it).
- **Step 2, per-company sub-queries:** measured and removed (docs/failures.md). The per-filing search filter is kept for the A7 corrective step.
- Also fixed at Phase 4 start: generation returned empty answers (qwen3.5 thinking), now `generation.think: false`.

## Held-out TEST results (2026-10-28, scored once, 91 questions: 37 local, 23 multi-hop [9 bridge, 14 comparison], 31 global)
`results/test/*.json`. No design decision was made on these numbers.

| config | all R@10 | local R@10 | multi-hop R@10 | multi-hop both@10 | bridge | comparison | global coverage@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 naive | 0.179 | 0.324 | 0.130 | 0.000 | 0.000 | 0.000 | 0.442 |
| A2 hybrid + headers | 0.519 | 0.946 | 0.500 | 0.130 | 0.000 | 0.214 | 0.394 |
| A2-RR (no graph) | 0.541 | 0.946 | 0.587 | 0.348 | 0.222 | 0.429 | 0.394 |
| A3 reranker | 0.532 | 1.000 | 0.457 | 0.217 | 0.111 | 0.286 | 0.397 |
| A4 graph | 0.563 | 0.946 | 0.674 | 0.435 | 0.222 | 0.571 | 0.394 |
| **A5 (final)** | **0.561** | **0.946** | **0.674** | **0.435** | 0.222 | 0.571 | **0.671** |

Honest reading:
- Clear on test: A5 vs A0 all recall@10 +0.383 [+0.260, +0.499]; local +0.62; multi-hop both@10 0.000 → 0.435; A5's company diversity +0.277 [+0.188, +0.362] over A4.
- **Not established on test:** the graph's own gain over the no-graph baseline. A4 vs A2-RR multi-hop both@10 +0.087 [−0.273, +0.297] (n = 23, cluster bootstrap by linking entity). On dev it was +0.192. Direction is consistent, the size is not.
- **Dev → test drop on multi-hop** (A4 both@10 0.789 → 0.435). The test questions are paraphrased user-style and their links are not dev links, and dev drove the rule design. That gap is the overfitting the held-out set exists to show.
- A5 global coverage 0.671 is below the 0.93 base rate for any 10 different companies; it measures diversity, not answer quality.
- The reranker made local recall@10 1.000 on test (+0.054 [0.000, +0.135]).

## Held-out test set (2026-10-27)
- Same procedure as dev: Gemini drafts (user-style phrasing, dev passages and company pairs excluded) -> 3 AI reviewer agents -> human. You approved the AI pre-review as a whole ("most edits not required, same meaning"), so human/AI agreement is 100% by construction. Report it that way.
- 130 candidates -> **91 verified** (37 local, 23 multi-hop: 14 comparison + 9 bridge, 31 global ≈ 2–3 angles × 12 themes). `data/golden/golden_test.jsonl`, built with `make golden-test`.
- Draft quality: 39/130 rejected (30%). Counting cosmetic edits as errors gives 88% (`review_test_stats.json`). The reviewers rejected executive-bio links, weak Ambarella–TSMC links, non-unique bridge answers and boilerplate.
- **Rule: final configs only, scored once.** Every design decision uses the dev set (`golden.jsonl`).

## Open questions for you
1. Sign off or amend ADRs 0001–0005. 0003 (the ticker list) and 0004 (definition of a retrieval hit) matter most.
2. The brief puts the LLM key and Ollama in Phase 3, but Phase 1 needs an LLM for candidates and the judge. I pulled the Gemini key forward. Is that OK?
3. Golden-set generation could alternatively be done by Claude Code inside a session (on your Pro plan) instead of Gemini. That is better quality but less reproducible. Current default: Gemini, cached.

## Log
- 2026-10-26 23:13: Session 1 start. Env: conda `torch` (Python 3.11.13, torch 2.5.1 + CUDA, RTX 4050 6 GB), git 2.53, Docker 29.1.3, Node 22.15, `ANTHROPIC_API_KEY` unset, gh logged in as ATHARVA-87. Ollama not installed.
- Phase 0 committed on `main`. Phase 1 committed in small steps on `phase1-harness` (see `git log`).
- Environment incidents (both logged in `docs/ai_review_log.md`, both fixed): broken `packaging` metadata after a pin; transformers 5.x needs torch ≥ 2.6, so it is pinned to <5.
- 2026-10-27 00:00: free tiers checked (ADR 0005): `gemini-2.0-flash` is shut down, so configs moved to `gemini-3.8-flash` and `qwen3.5:4b`.
- 2026-10-27 00:05–00:12: reviewer subagent pass on the Phase 1 branch. 9 confirmed bugs fixed: partial-evidence recall inflation, greedy matching, TOC sections, heading regex, section gaps, judge agreement, golden build crashes, BOM CSVs, repeated quotes. Plus EDGAR robustness and a versioned index key.
- Known limits, not fixed: chunks longer than bge's 512 tokens are truncated (now counted in `build_info.json`); the parser is validated on synthetic HTML only.
- 2026-10-27 09:20–09:40: `make ingest` (156 filings, 39 cos; QRVO missing after the Skyworks merger; YAML `ON` ticker bug fixed). `make process`: parser v4, 152/156 fully sectioned. `make index` for A0: 20,360 chunks, 156 s on GPU, **1,031 chunks (5.1%) exceed bge's 512 tokens** (table-heavy), a known A0 weakness for A1 to address with token-based sizing.
- User chose to keep the brief's phase order (multi-agent = Phase 5). Gemini + Groq keys added. `make candidates` done: 180 candidates (10% auto-rejected) + 21 bridge questions. **Next: you review `review.csv` (local + multi-hop first) and `review_bridge.csv`; spot-check `global_themes.csv`; then `make golden` and `make eval`.**
