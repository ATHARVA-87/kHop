# Plexus — progress

## Phases

| Phase | Work | Gate | Status |
| --- | --- | --- | --- |
| 0. Setup | Toolchain, repo, CLAUDE.md, PROGRESS.md, Makefile, `make doctor`, CI skeleton, LLM wrapper + cache | `make doctor` and CI pass | **Done locally**; CI waits on GitHub remote (step 2) |
| 1. Harness | EDGAR snapshot, golden-set candidates, human verification, metrics, A0 | A0 scored with CIs | Not started |
| 2. Retrieval + CI gate | A1–A3, retrieval gate on PRs | A1–A3 recorded; gate blocks a planted regression | — |
| 3. Graph | Extraction, resolution, Kuzu, Leiden, summaries, A4, A5 | Extraction P/R; A4, A5 recorded | — |
| 4. Single-graph agent | Router, grade-and-rewrite, A6, A7 | Router vs always-local/global | — |
| 5. Multi-agent | 4 agents, guardrails, A9 | A9 vs A7 with latency/cost; guardrail metrics | — |
| 6. Systems + UI | Semantic cache (A8), SSE, Phoenix, Next.js, Locust | p95, cache hit rate, cost/query | — |
| 7. Ship | README, ADRs, demo Space, drift check, demo script | Definition of done | — |

## Gate evidence

### Phase 0 (2026-10-26)
- `make doctor`: all phase-0 checks pass except **GitHub remote** (human step 2).
- `make test`: 17 passed (LLM cache: hit/miss, key covers every parameter, persistence, offline mode, provider gating, JSON validation; doctor logic).
- `make lint`: clean.
- CI: workflow written (`.github/workflows/ci.yml`, uses `uv.lock`), **not yet run**, because there is no remote.

## Log
- 2026-10-26: Session 1. Env check: Python 3.11.13 (conda `torch`), git 2.53, Docker 29.1.3 (daemon up), Node 22.15, `ANTHROPIC_API_KEY` unset, gh logged in as ATHARVA-87. Ollama not installed (needed in Phase 3).
- 2026-10-26: Phase 0 scaffold, LLM wrapper + SQLite cache, doctor, CI, ADR 0001–0002.

## Open questions for you
1. Sign off or amend ADR 0001 (LLM cache) and ADR 0002 (env + dependency split).

## Blockers (need you)
- **Step 2:** create an empty public GitHub repo `plexus` and give me the URL (or run `git remote add origin <url> && git push -u origin main`). Then CI can run.

## Resume here
Next task: Phase 1, EDGAR ingestion (see branch notes below once started).
