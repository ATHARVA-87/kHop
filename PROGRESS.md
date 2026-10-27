# Plexus — progress

## Phases

| Phase | Work | Gate | Status |
| --- | --- | --- | --- |
| 0. Setup | Toolchain, repo, CLAUDE.md, PROGRESS.md, Makefile, `make doctor`, CI skeleton, LLM wrapper + cache | `make doctor` and CI pass | **Passed 2026-10-27.** Doctor phase 0 all OK; GitHub CI green (run 40440596122) |
| 1. Harness | EDGAR snapshot, golden-set candidates, human verification, metrics, A0 | A0 scored with CIs | **Code complete on branch `phase1-harness`** (not merged; needs your review). Waits on your steps below |
| 2. Retrieval + CI gate | A1–A3, retrieval gate on PRs | A1–A3 recorded; gate blocks a planted regression | Not started (gate rule) |
| 3. Graph | Extraction, resolution, Kuzu, Leiden, summaries, A4, A5 | Extraction P/R; A4, A5 recorded | — |
| 4. Single-graph agent | Router, grade-and-rewrite, A6, A7 | Router vs always-local/global | — |
| 5. Multi-agent | 4 agents, guardrails, A9 | A9 vs A7 with latency/cost; guardrail metrics | — |
| 6. Systems + UI | Semantic cache (A8), SSE, Phoenix, Next.js, Locust | p95, cache hit rate, cost/query | — |
| 7. Ship | README, ADRs, demo Space, drift check, demo script | Definition of done | — |

## Resume here

**State:** branch `phase1-harness` (checked out). `main` holds Phase 0. 92 tests pass (`make test-all`). A reviewer subagent checked the branch; all 9 confirmed bugs are fixed with regression tests (see `docs/ai_review_log.md`). No real data yet: every number so far comes from synthetic test fixtures.

**Your steps, in order** (each one unblocks the next; run `make doctor PHASE=1` after each):

1. ~~**GitHub repo**~~ Done 2026-10-27: https://github.com/ATHARVA-87/plexus, PR #1 open (phase1-harness → main). **Review PR #1** (metrics and golden-set code) before merging.
2. ~~**SEC User-Agent**~~ Done 2026-10-27. (step 3): `cp .env.example .env`, set `SEC_USER_AGENT="Your Name your@email"`.
3. ~~**Download + parse**~~ Done 2026-10-27: 156 filings / 39 companies; 152 fully sectioned, 4 Intel unstructured (docs/failures.md). A0 index built (20,360 chunks). Originally:
   `conda activate torch && make ingest && make process`. Then read `data/snapshots/semis-2025.parse_report.json`. Section coverage below ~90% for `risk_factors` or `mdna` means the parser needs fixing on real filings. Tell me which doc IDs fail.
4. **Gemini key** (step 5, pulled forward because candidates and the judge need it): create a free key at https://aistudio.google.com/apikey and paste it into `.env` as `GEMINI_API_KEY` yourself. Check your free-tier limits in AI Studio (ADR 0005).
5. **Candidates:** `make candidates` writes about 200 rows to `data/golden/review.csv`. Then `make index CONFIG=configs/a0_naive.yaml`.
6. **Review** (step 6, the core human task): follow `docs/golden_review_guide.md` and verify at least 100 rows. Then `make golden`.
7. **A0:** `make eval CONFIG=configs/a0_naive.yaml` gives retrieval metrics with CIs, which is the Phase 1 gate. For generation metrics, install Ollama (step 4, needs sudo) and `ollama pull qwen3.5:4b`, then `make eval-full`, `make judge-export`, label about 50 answers, and `make judge-agreement`.

**Next Claude task once data exists:** fix the parser on real filings using the parse report, then start Phase 2 (A1 structure-aware parent-child chunks).

## Gate evidence

### Phase 0 (2026-10-26)
- `make doctor`: all phase-0 checks pass, GitHub remote included (2026-10-27).
- GitHub CI run 40440596122 on `phase1-harness`: 77 packages from `uv.lock` (no torch), ruff clean, 89 passed / 3 ml deselected.
- Note: GitHub did not fire the push and pull_request events for the first pushes to the new repo. `ci.yml` now also has `workflow_dispatch`. If PR #1 shows no checks, run `gh workflow run ci --ref phase1-harness`.
- `make test`: passes. `make lint`: clean.
- CI dry run: fresh venv from `uv.lock` (core + dev, no torch), ruff clean, 66 passed / 3 ml deselected. Done 2026-10-26 23:54, environment deleted afterwards.

### Phase 1 (in progress)
- Code complete and tested on synthetic filings (`tests/test_pipeline_e2e.py` runs the whole A0 path with the real bge-small model).
- Not yet run on real EDGAR data (needs your steps 2–3).

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
- AI pre-review done (3 Claude reviewer agents); suggestions are in the `ai_*` columns. **Human verdicts pending.**
- Theme table rebuilt with synonyms: 434 yes / 34 no. Gemini vs AI reviewer agreement on 93 spot-check rows: 96% (was 83% with literal keywords). Gemini 3.6-flash was overloaded, so all labels come from 3.5-flash-lite.
- `company_coverage@k` is normalised by min(k, relevant companies).

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
