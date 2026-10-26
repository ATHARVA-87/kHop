# Plexus — progress

## Phases

| Phase | Work | Gate | Status |
| --- | --- | --- | --- |
| 0. Setup | Toolchain, repo, CLAUDE.md, PROGRESS.md, Makefile, `make doctor`, CI skeleton, LLM wrapper + cache | `make doctor` and CI pass | **Done locally.** CI simulated locally from `uv.lock` (passes); real GitHub run waits on step 2 |
| 1. Harness | EDGAR snapshot, golden-set candidates, human verification, metrics, A0 | A0 scored with CIs | **Code complete on branch `phase1-harness`** (not merged; needs your review). Waits on your steps below |
| 2. Retrieval + CI gate | A1–A3, retrieval gate on PRs | A1–A3 recorded; gate blocks a planted regression | Not started (gate rule) |
| 3. Graph | Extraction, resolution, Kuzu, Leiden, summaries, A4, A5 | Extraction P/R; A4, A5 recorded | — |
| 4. Single-graph agent | Router, grade-and-rewrite, A6, A7 | Router vs always-local/global | — |
| 5. Multi-agent | 4 agents, guardrails, A9 | A9 vs A7 with latency/cost; guardrail metrics | — |
| 6. Systems + UI | Semantic cache (A8), SSE, Phoenix, Next.js, Locust | p95, cache hit rate, cost/query | — |
| 7. Ship | README, ADRs, demo Space, drift check, demo script | Definition of done | — |

## Resume here

**State:** branch `phase1-harness` (checked out). `main` holds Phase 0. 70 tests pass (`make test-all`). No real data yet: every number so far comes from synthetic test fixtures.

**Your steps, in order** (each one unblocks the next; run `make doctor PHASE=1` after each):

1. **GitHub repo** (step 2): create an empty public repo `plexus`, then
   `git remote add origin <url> && git push -u origin main phase1-harness`. Then open a PR from `phase1-harness` → `main` and check CI is green. That closes the Phase 0 gate.
2. **SEC User-Agent** (step 3): `cp .env.example .env`, set `SEC_USER_AGENT="Your Name your@email"`.
3. **Download + parse** (you run these, about 5 minutes):
   `conda activate torch && make ingest && make process`. Then read `data/snapshots/semis-2025.parse_report.json`. Section coverage below ~90% for `risk_factors` or `mdna` means the parser needs fixing on real filings. Tell me which doc IDs fail.
4. **Gemini key** (step 5, pulled forward because candidates and the judge need it): create a free key at https://aistudio.google.com/apikey and paste it into `.env` as `GEMINI_API_KEY` yourself. Check your free-tier limits in AI Studio (ADR 0005).
5. **Candidates:** `make candidates` writes about 200 rows to `data/golden/review.csv`. Then `make index CONFIG=configs/a0_naive.yaml`.
6. **Review** (step 6, the core human task): follow `docs/golden_review_guide.md` and verify at least 100 rows. Then `make golden`.
7. **A0:** `make eval CONFIG=configs/a0_naive.yaml` gives retrieval metrics with CIs, which is the Phase 1 gate. For generation metrics, install Ollama (step 4, needs sudo) and `ollama pull qwen3.5:4b`, then `make eval-full`, `make judge-export`, label about 50 answers, and `make judge-agreement`.

**Next Claude task once data exists:** fix the parser on real filings using the parse report, then start Phase 2 (A1 structure-aware parent-child chunks).

## Gate evidence

### Phase 0 (2026-10-26)
- `make doctor`: all phase-0 checks pass except **GitHub remote** (your step 2).
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

## Open questions for you
1. Sign off or amend ADRs 0001–0005. 0003 (the ticker list) and 0004 (definition of a retrieval hit) matter most.
2. The brief puts the LLM key and Ollama in Phase 3, but Phase 1 needs an LLM for candidates and the judge. I pulled the Gemini key forward. Is that OK?
3. Golden-set generation could alternatively be done by Claude Code inside a session (on your Pro plan) instead of Gemini. That is better quality but less reproducible. Current default: Gemini, cached.

## Log
- 2026-10-26 23:13: Session 1 start. Env: conda `torch` (Python 3.11.13, torch 2.5.1 + CUDA, RTX 4050 6 GB), git 2.53, Docker 29.1.3, Node 22.15, `ANTHROPIC_API_KEY` unset, gh logged in as ATHARVA-87. Ollama not installed.
- Phase 0 committed on `main`. Phase 1 committed in small steps on `phase1-harness` (see `git log`).
- Environment incidents (both logged in `docs/ai_review_log.md`, both fixed): broken `packaging` metadata after a pin; transformers 5.x needs torch ≥ 2.6, so it is pinned to <5.
- 2026-10-27 00:00: free tiers checked (ADR 0005): `gemini-2.0-flash` is shut down, so configs moved to `gemini-3.8-flash` and `qwen3.5:4b`.
