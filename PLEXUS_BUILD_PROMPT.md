# Plexus — Build Prompt for Claude Code

> How to use: put this file in an empty project folder, open Claude Code there, and say:
> "Read PLEXUS_BUILD_PROMPT.md and start Session 1."
> To resume after any stop: "Read CLAUDE.md and PROGRESS.md, then continue."

---

## 1. Your role

You are the senior engineer building **Plexus** with me. I am a final-year B.Tech student. This is my main resume project, aimed at AI engineering roles (for example, Cognizant's Ace Frontier Engineer role: multi-agent systems, RAG pipelines, AI quality gates in CI/CD, observability, guardrails, cost and latency trade-offs).

I must be able to explain and defend every component in an interview. So:
- Explain non-obvious design choices briefly as you go, and record major ones as decision records.
- Leave the judgment-heavy work to me where this prompt says so (golden-set labels, final ablation analysis, decision-record sign-off).
- Never fabricate results. Every number in the README must come from a script in the repo that I can re-run.

---

## 2. What Plexus is and why we are building it

**One line:** an eval-driven, multi-agent hybrid GraphRAG system that answers questions over SEC filings and proves, with its own benchmark, where each retrieval strategy helps.

**The problem:** top-k chunk retrieval fails on two kinds of question:
- **Multi-hop:** "Which suppliers named in Company A's filing also appear in Company B's?"
- **Corpus-level:** "What risk themes recur across the semiconductor sector?"

**The approach:** a vector index plus a knowledge graph, with four small agents that plan, research, write and verify. Every component is justified by a measured gain over a naive-RAG baseline.

**Why this project:** it gives one research question, a benchmark I own, ablations, and systems numbers (latency, cost, cache hit rate). That is a deep interview story, not a feature demo.

**Corpus:** 10-K and 10-Q filings from SEC EDGAR for one sector, about 30 to 50 companies, frozen as a snapshot. Answers are research output, not investment advice.

---

## 3. Non-negotiable rules

1. **Free to build and run.** Use only local tools and free tiers. Never add a paid service. If something would cost money, stop and tell me.
2. **Claude Code runs on my Pro subscription.** At the start of each session, check that `ANTHROPIC_API_KEY` is not set in the environment. If it is set, warn me, because Claude Code would bill the API instead of my plan.
3. **The app's runtime agents do not depend on my Claude login.** Default runtime providers are Ollama (local) and the Gemini or Groq free tiers. Add an optional, off-by-default `claude_sdk` provider for personal local comparison runs only, and note in the README that it must never serve other users.
4. **Cache every LLM call on disk,** keyed by provider, model, prompt and parameters. Reruns and CI must cost nothing.
5. **Reproducible:** pinned dependencies (lockfile), fixed seeds, corpus checksum, one command per experiment.
6. **No secrets in git.** `.env` is gitignored; ship `.env.example`.
7. **Small, reviewable steps.** One task at a time, tests with it, commit after each passing step.
8. **Honest AI attribution.** Keep the `Co-Authored-By` commit trailer. Maintain an "AI-assisted development" README section and a `docs/ai_review_log.md` where we record AI-generated mistakes that tests or evals caught.

---

## 4. Session and usage-limit protocol (keep work continuous within Pro limits)

My plan has a rolling 5-hour usage window and a weekly cap. You cannot see my exact remaining quota, so work so that a stop at any moment loses nothing.

**At the start of every session**
1. Read `CLAUDE.md` and `PROGRESS.md`.
2. Check `git status`; if there are uncommitted changes, review and finish or commit them first.
3. State in two or three lines: where we are, the next task, and its done-condition.

**While working**
- Break work into tasks of about 20 to 40 minutes, each ending in a commit.
- After every commit, update `PROGRESS.md`: done, in progress, next task, open questions, blockers.
- Prefer cheap operations: read only the files you need, avoid re-reading large files, avoid dumping long logs into context (write them to files and summarize).
- Never start a long or risky step (large refactor, big extraction run) without a checkpoint commit first.

**Session length**
- Aim for a natural stopping point roughly every 2 to 3 hours of work, or when I say so. At that point write a handoff entry in `PROGRESS.md` titled "Resume here" with the exact next command or task, and commit.
- If you are cut off by a usage limit, the last commit plus `PROGRESS.md` must be enough to resume. That is why we commit often.
- When I come back after the 5-hour window resets, I will say "continue" and you follow "At the start of every session".

**Long-running jobs**
- Extraction and embedding runs go in scripts I start myself (for example overnight with `make extract`), resumable from their own cache, so they never consume Claude Code usage.

---

## 5. Minimum setup I must do myself

Do everything you can yourself. Only ask me for the steps below, **one at a time, at the moment they are needed** (not all up front). For each: give the exact commands or links, say why it is needed, and wait for me to confirm. Then verify it worked with a check script.

| # | When | What I do | Why |
| --- | --- | --- | --- |
| 1 | Session 1 | Install Python 3.11+, Git, Docker Desktop, Node.js LTS (only those missing; check first) | Core toolchain |
| 2 | Session 1 | Create an empty public GitHub repo named `plexus` and give you its URL | Free CI, portfolio |
| 3 | Phase 1 | Set a descriptive `User-Agent` with my name and email for SEC EDGAR requests (required by SEC fair-access policy) | Data download |
| 4 | Phase 3 | Install Ollama and pull one small model you name | Free local LLM for dev and cheap agents |
| 5 | Phase 3 | Create a free Gemini API key and/or Groq API key and paste them into `.env` myself (never into chat) | Free-tier LLM for Writer and extraction |
| 6 | Phase 1 onward | Hand-verify golden-set queries in a review file you prepare | The benchmark must be human-checked |
| 7 | Phase 7 | Create a free Hugging Face account for the demo Space | Free hosting |

Note for step 5: free-tier providers may use submitted data. Only public SEC data goes to them.

Provide `make doctor`, a script that checks every prerequisite and prints exactly what is missing and how to fix it.

---

## 6. Architecture

### Offline (once per corpus snapshot)
1. **Ingest:** download filings from EDGAR (rate-limited, cached), keep section structure (Item 1A Risk Factors, Item 7 MD&A) as metadata.
2. **Chunk:** structure-aware chunks with parent-child links; index children, return parent sections as context.
3. **Hybrid index:** Qdrant (local or Docker) with dense and sparse vectors per chunk; fuse with reciprocal rank fusion.
4. **Graph:** LLM extraction of typed entities (company, person, product, supplier, risk factor, segment) and typed edges, with a strict JSON schema; entity resolution (alias table + fuzzy match + embedding similarity); store in Kuzu (embedded).
5. **Communities:** Leiden clustering; an LLM summary per community, embedded and indexed.

### Online (per question): four agents on a fixed LangGraph
Each agent is one LLM call with a narrow prompt and typed input and output (Pydantic). Agents never chat freely; they read and write one shared state object. The graph topology is fixed and has a hard step cap.

| Agent | Job | Model tier |
| --- | --- | --- |
| **Planner** | Classify the query (local, multi-hop, global); split multi-hop questions into sub-questions | Small or local |
| **Researcher** | For each sub-question, call deterministic retrieval tools: `hybrid_search`, `graph_traverse`, `community_search`; return evidence with IDs. Sub-questions may run in parallel | Small or local |
| **Writer** | Draft the answer; every claim cites an evidence ID | Strongest free option |
| **Verifier** | Check each claim against its cited evidence; on failure send structured feedback to Planner or Researcher | Small or local |

Flow: question → semantic cache → Planner → Researcher → Writer → Verifier → pass: stream answer; fail: loop with feedback; after the retry cap, return "insufficient evidence" and flag for human review.

Also build the **single-graph baseline** (router → retrieve → grade → rewrite → generate) first, so multi-agent is measured against it.

### Guardrails
- Treat retrieved text as untrusted: strip or neutralize instruction-like content before it reaches agents (prompt-injection defence), and test it with planted malicious chunks.
- Enforce citations; flag unsupported claims.
- Refuse investment-advice and out-of-scope questions with a clear message.
- Measure each guardrail (for example, injection attempts blocked out of N planted).

### Serving and UI
- FastAPI with Server-Sent Events: stream tokens and agent events (agent started, route, evidence, verifier verdict).
- Next.js UI with a live trace panel showing which agent did what, with cited evidence.
- Semantic cache keyed by query embedding and corpus version, with TTL.
- OpenTelemetry spans per agent and stage, viewed in Arize Phoenix (self-hosted).

---

## 7. Stack (all free)

| Layer | Choice |
| --- | --- |
| Language | Python 3.11+ (backend), TypeScript (UI) |
| Orchestration | LangGraph |
| LLM client | One provider-agnostic wrapper (for example LiteLLM) with disk cache; per-agent provider and model set in config |
| Runtime LLMs | Ollama (local), Gemini or Groq free tier; optional `claude_sdk` provider, off by default |
| Parsing | Docling or unstructured |
| Embeddings | bge-small for development, bge-m3 for final index (local) |
| Reranker | bge-reranker (local) |
| Vector store | Qdrant (local mode or Docker) |
| Graph store | Kuzu (embedded) |
| Communities | igraph or graspologic (Leiden) |
| Tracing | Arize Phoenix + OpenTelemetry |
| API / UI | FastAPI (SSE), Next.js |
| Testing | pytest, ruff, pre-commit; Locust for load tests |
| CI | GitHub Actions (public repo) |
| Demo | Hugging Face Spaces (free CPU) with prebuilt indexes |

Before adopting any free tier, check its current terms and note the date checked in `docs/adr/`.

---

## 8. Repository layout

```
plexus/
  CLAUDE.md                 standing rules for Claude Code (create in Session 1 from this prompt)
  PROGRESS.md               live progress and "Resume here" handoff
  README.md                 results table, ablations, failure cases, AI-assisted development section
  Makefile                  doctor, ingest, index, extract, eval, serve, ui, test
  pyproject.toml + lockfile
  .env.example
  configs/                  one YAML per experiment (A0 to A9)
  data/raw, data/processed  gitignored, rebuilt by scripts
  data/golden/              committed golden set and labels
  src/plexus/
    llm/                    provider wrapper + disk cache
    ingest/                 edgar, parse, chunk
    extract/                schema, entities, resolution
    index/                  dense_sparse, graph, communities
    retrieve/               hybrid, rerank, graph_search, global_search
    agents/                 state, planner, researcher, writer, verifier, graph
    baseline/               single-graph router pipeline
    guardrails/
    serve/                  FastAPI app, SSE, semantic cache
    eval/                   golden, metrics, judge, run, bootstrap
  ui/                       Next.js app
  tests/
  docs/adr/                 decision records
  docs/ai_review_log.md
  .github/workflows/        ci.yml, nightly-eval.yml
```

---

## 9. Evaluation (build this before the system)

**Golden set (~150 queries):** 60 local, 50 multi-hop, 40 global. You generate ~200 candidates with an LLM into `data/golden/review.csv` with columns for my verdict and corrections. I verify at least half. Report the label-noise rate found. Never score on unverified queries.

**Metrics**
- Retrieval: recall@k, nDCG@10, MRR, graph path hit rate.
- Generation: faithfulness (claims supported by citations), answer correctness, citation precision.
- Extraction: entity and relation precision and recall on a labelled sample I check.
- Agents: Planner decomposition accuracy (hand-labelled sub-questions); Verifier recall on deliberately corrupted answers; routing accuracy.
- Systems: p50/p95 latency, first-token latency, LLM calls and tokens per query, cache hit rate.
- Guardrails: injection-block rate, refusal accuracy.

**Ablation ladder** (one config each, every metric recorded with bootstrap confidence intervals):
A0 naive RAG → A1 structure-aware chunking → A2 hybrid + fusion → A3 reranker → A4 graph traversal → A5 community search → A6 router → A7 grade-and-rewrite (single graph) → A8 semantic cache → **A9 multi-agent vs A7**.

**Rigor**
- LLM judge validated against ~50 answers I label myself; report Cohen's kappa.
- CI gate on every PR uses retrieval metrics only (deterministic, no LLM calls). Full agent eval runs nightly or manually on a subset, from cache.
- Drift check: re-run the golden set on a newer filings snapshot and report changes.
- Keep `docs/failures.md`. Negative results are part of the deliverable.

---

## 10. Phases and gates

Do not start a phase until the previous gate passes. Record each gate's evidence in `PROGRESS.md`.

| Phase | Work | Gate |
| --- | --- | --- |
| 0. Setup (Session 1) | Toolchain check, repo, CLAUDE.md, PROGRESS.md, Makefile, `make doctor`, CI skeleton, LLM wrapper with cache | `make doctor` and CI pass |
| 1. Harness | EDGAR snapshot, golden-set candidates, my verification, metrics, naive baseline A0 | A0 scored with CIs |
| 2. Retrieval + CI gate | A1 to A3, retrieval eval gate on PRs | A1 to A3 recorded; gate blocks a planted regression |
| 3. Graph | Extraction, resolution, Kuzu, Leiden, summaries, A4 and A5 | Extraction P/R measured; A4, A5 recorded |
| 4. Single-graph agent | Router and grade-and-rewrite loop, A6 and A7 | Router compared with always-local and always-global |
| 5. Multi-agent | Planner, Researcher, Writer, Verifier, guardrails, A9 | A9 vs A7 with latency and cost; guardrail metrics |
| 6. Systems + UI | Semantic cache (A8), SSE, Phoenix, Next.js UI, Locust | p95, cache hit rate, cost per query reported |
| 7. Ship | README, decision records, demo Space, drift check, 3-minute demo script | Definition of done met |

---

## 11. Frontend design

Follow Anthropic's frontend-design skill:
https://github.com/anthropics/claude-code/blob/main/plugins/frontend-design/skills/frontend-design/SKILL.md

If the frontend-design plugin is installed, use it. Otherwise read that SKILL.md before any UI work and follow it fully. Key points:

- **Ground the design in the subject.** Subject: a research tool for financial analysts reading SEC filings. Audience: analysts, and interviewers watching a demo. Primary job: ask a question, read a cited answer, and see how the agents reached it. Propose this framing to me and confirm before designing.
- **Two passes.** First a compact design plan: 4–6 named hex colors, typefaces and roles, a layout concept with ASCII wireframes and alignment guidance, and the principles that make this page specific. Then review the plan against the brief; revise anything that reads like a generic default and tell me what changed and why. Only then write code.
- **Avoid the generic tells the skill lists,** including the cream-background-plus-terracotta look, the near-black-plus-acid-accent look, the identical-rounded-card SaaS kit with soft shadows and gradient washes, all-caps eyebrow labels, single accented words in headlines, middle-dot meta strings, and arrows appended to buttons.
- **Typography carries personality:** one or two deliberate typefaces, a clear type scale, line length under 80 characters.
- **Structure is information:** numbering only for real sequences. Here the agent trace genuinely is a sequence, so it may be numbered; other content should not be.
- **Motion:** one deliberate moment at most, plus motion that answers user actions (expanding a citation, a trace step completing).
- **Spend boldness in one place.** The live agent trace with clickable citations is the natural candidate.
- **Quality floor:** responsive to mobile, visible keyboard focus, reduced motion respected, accessible contrast.
- **Copy:** plain, user-facing words ("Ask a question", "Sources", "Not enough evidence in the filings to answer this"). Errors say what happened and how to fix it; empty states invite action.
- **Self-critique:** take screenshots of the running UI if your environment allows and review them before calling UI work done.

---

## 12. Git and review workflow

- Feature branches and PRs for each phase step; short PR descriptions with what changed, how it was tested, and eval deltas.
- I review PRs that touch agents, metrics, golden-set code or the router before merge.
- Commit messages read like a lab notebook ("A3: add bge-reranker; nDCG@10 +x, see results/a3.json").
- Use Claude Code subagents where they help (for example a reviewer for PRs against the decision records, a test writer, an eval-results summarizer), and log meaningful catches in `docs/ai_review_log.md`.

---

## 13. Definition of done

- [ ] Golden set (~150) committed, at least half hand-verified, noise rate reported
- [ ] A0 to A9 run, results table with confidence intervals in README
- [ ] Extraction P/R, Planner and Verifier accuracy, router accuracy reported
- [ ] LLM-judge agreement with my labels reported
- [ ] Guardrail metrics reported
- [ ] p50/p95 latency, cache hit rate, calls and tokens per query from a load test
- [ ] Retrieval eval gate in CI, with one demonstrated blocked regression
- [ ] Drift check on a second snapshot
- [ ] UI deployed on a free host, built per the frontend-design skill
- [ ] README with architecture diagram, results, failure cases, reproduction steps, AI-assisted development section
- [ ] Decision records for each major choice, signed off by me
- [ ] Resume bullets drafted using measured numbers only

---

## 14. Session 1 — start here

1. Check the environment (tools present, `ANTHROPIC_API_KEY` not set). Ask me only for missing setup items from section 5, one at a time.
2. Create `CLAUDE.md` containing sections 3, 4, 5 and 11 of this prompt in condensed form, so they apply in every future session.
3. Create `PROGRESS.md` with the phase table and a "Resume here" entry.
4. Scaffold the repo layout, Makefile, `pyproject.toml`, `.env.example`, pre-commit, and a CI workflow running ruff and pytest.
5. Build the LLM wrapper with disk cache and one passing test.
6. Implement `make doctor`.
7. Commit, push, confirm CI is green, update `PROGRESS.md`, and tell me the next task (Phase 1: EDGAR ingestion).
