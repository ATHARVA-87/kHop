# Plexus — standing rules for Claude Code

Full brief: `PLEXUS_BUILD_PROMPT.md`. Live state: `PROGRESS.md`. Read both at session start.

## Environment
- Python work runs in the conda env **`torch`** (`~/miniconda3/envs/torch`, Python 3.11, torch 2.5.1 + CUDA, RTX 4050 6 GB).
- **Never change torch.** Always install with `-c constraints.txt` (pins torch 2.5.1, numpy 1.26.4). Dry-run first if unsure. If something cannot fit, make a separate env.
- Repo path contains a trailing space (`.../Desktop/RAG /`): always quote it.
- npm registry is flaky here: use `NODE_OPTIONS=--dns-result-order=ipv4first` and `--fetch-timeout=120000`.

## Non-negotiable rules (brief §3)
1. Free only: local tools and free tiers. Anything that costs money → stop and ask.
2. Claude Code runs on the Pro plan: check `ANTHROPIC_API_KEY` is **unset** each session (`make doctor` does this).
3. Runtime agents never depend on the developer's Claude login. Defaults: Ollama, Gemini/Groq free tier. `claude_sdk` provider is off by default (`PLEXUS_ENABLE_CLAUDE_SDK=1`), personal local comparison only, never serves other users.
4. Every LLM call goes through `plexus.llm.LLMClient` (disk cache keyed by provider, model, prompt, params). CI runs with `PLEXUS_LLM_OFFLINE=1` (cache-only).
5. Reproducible: `uv.lock`, fixed seeds, corpus checksum, one command per experiment (`make eval CONFIG=...`).
6. No secrets in git. `.env` is gitignored; `.env.example` is the template.
7. Small steps: one task, tests with it, commit after each passing step.
8. Keep the `Co-Authored-By` trailer. Log AI mistakes caught by tests/evals in `docs/ai_review_log.md`.
9. Never fabricate numbers. Every README number comes from a script in the repo.

## Session protocol (brief §4)
- Start: read CLAUDE.md + PROGRESS.md, check `git status`, state where we are / next task / done-condition.
- Work in 20–40 min tasks, each ending in a commit; update PROGRESS.md after every commit.
- Keep context cheap: read only needed files, write long logs to files and summarise.
- Checkpoint-commit before any long or risky step.
- Every ~2–3 h (or when asked) write a "Resume here" entry in PROGRESS.md and commit.
- Long jobs (ingest, embed, extract) are scripts the human runs (`make ingest`), resumable from their own cache.

## Human-only steps (brief §5) — ask one at a time, when needed, then verify with `make doctor`
| # | When | Step |
|---|------|------|
| 1 | S1 | Toolchain (Python 3.11+, Git, Docker, Node LTS) — all present as of 2026-10-26 |
| 2 | S1 | Create empty public GitHub repo `plexus`, give URL |
| 3 | Phase 1 | `SEC_USER_AGENT="Name email"` in `.env` |
| 4 | Phase 3 | Install Ollama, pull the named model |
| 5 | Phase 3 | Gemini and/or Groq free key into `.env` (never in chat). Only public SEC data goes to them |
| 6 | Phase 1+ | Hand-verify golden-set queries in `data/golden/review.csv` |
| 7 | Phase 7 | Hugging Face account for the demo Space |

Human judgment is reserved for: golden-set labels, final ablation analysis, ADR sign-off, and PR review of agents / metrics / golden-set code / router.

## Git
- Feature branch per phase step; PRs once the GitHub remote exists. Do not merge branches touching agents, metrics, golden-set code or the router without the human's review.
- Commit messages read like a lab notebook ("A3: add bge-reranker; nDCG@10 +x, see results/a3.json").

## Frontend (brief §11) — before any UI work
Follow Anthropic's frontend-design skill (read its SKILL.md). Confirm the framing first (research tool for analysts reading SEC filings; job = ask, read cited answer, see agent trace). Two passes: design plan (4–6 named hex colours, typefaces + roles, ASCII wireframes) → self-review against the brief → code. Avoid generic tells (cream+terracotta, near-black+acid accent, rounded-card SaaS kit, all-caps eyebrows, single accented headline word, middle-dot meta strings, arrows on buttons). Boldness in one place: the live numbered agent trace with clickable citations. One deliberate motion moment max. Responsive, visible focus, reduced motion, accessible contrast. Plain copy. Screenshot and self-critique before calling UI done.
