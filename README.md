# Plexus

An eval-driven, multi-agent hybrid GraphRAG system that answers questions over SEC filings and measures, with its own benchmark, where each retrieval strategy helps.

> Status: **Phase 0 (setup)**. No results yet. Every number that appears here will come from a script in this repo that you can re-run.

## Why

Top-k chunk retrieval fails on two kinds of question:

- **Multi-hop:** "Which suppliers named in Company A's filing also appear in Company B's?"
- **Corpus-level:** "What risk themes recur across the semiconductor sector?"

Plexus combines a hybrid vector index with a knowledge graph and four narrow agents (Planner, Researcher, Writer, Verifier). Each component has to earn its place with a measured gain over a naive-RAG baseline (ablations A0–A9).

Answers are research output over public filings, **not investment advice**.

## Quick start

```bash
conda activate torch              # Python 3.11 env (torch 2.5.1 stays pinned)
make install-ml                   # pip install -c constraints.txt -e '.[ml,dev]'
make doctor                       # what is missing and how to fix it
make test
```

## Results

_None yet._ The ablation table (A0–A9 with bootstrap confidence intervals) goes here once Phase 1 produces A0.

## Reproducibility

- Dependencies are locked in `uv.lock`. Local installs use `constraints.txt` so the shared torch env is never modified.
- Every LLM call is cached on disk (`data/cache/llm.sqlite`) and keyed by provider, model, prompt and parameters. With `PLEXUS_LLM_OFFLINE=1` a cache miss raises an error rather than calling a model, so CI and reruns cost nothing.
- Corpus snapshots are frozen and checksummed (Phase 1).

## LLM providers

| Provider | Use | Default |
| --- | --- | --- |
| Ollama (local) | Planner, Researcher, Verifier, development | on |
| Gemini / Groq free tier | Writer, extraction | on (needs key in `.env`) |
| `claude_sdk` | Personal local comparison runs only | **off** |

The `claude_sdk` provider uses the developer's own Claude login. It is off by default (`PLEXUS_ENABLE_CLAUDE_SDK=1` to enable) and **must never serve other users**.

## AI-assisted development

This project is built with Claude Code as a pair programmer. Commits it co-authored carry a `Co-Authored-By` trailer. Design decisions are recorded in `docs/adr/` and signed off by me. Mistakes in AI-generated code that tests or evals caught are logged in `docs/ai_review_log.md`, and negative results are kept in `docs/failures.md`.
