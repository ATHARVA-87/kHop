# ADR 0002: Reuse the `torch` conda env; split core and ML dependencies

- Status: Proposed (awaiting sign-off)
- Date: 2026-10-26

## Context
The dev machine already has a conda env `torch` (Python 3.11, torch 2.5.1 + CUDA 12, RTX 4050 6 GB) that other projects use. It must not change. CI runs on free GitHub runners with no GPU, where a full torch install is slow.

## Decision
- Develop in the `torch` env. Every install uses `-c constraints.txt`, which pins `torch==2.5.1` and `numpy==1.26.4`.
- `pyproject.toml` has light core deps (pydantic, httpx, litellm, bs4, numpy), an `ml` extra (torch, sentence-transformers, qdrant-client) and a `dev` extra.
- `uv.lock` pins every package for every platform. CI runs `uv sync --frozen --extra dev`, so it never installs torch.
- Tests that need models are marked `@pytest.mark.ml` and skipped in CI. The PR retrieval gate (Phase 2) will run on precomputed embeddings to stay CPU-cheap and deterministic.

## Consequences
- The local env has looser pins than `uv.lock`. A drift check (`uv pip compile` vs `pip freeze`) belongs in `make doctor` later.
- During setup on 2026-10-26, installing into `torch` upgraded five small conda-installed utilities (certifi, click, filelock, platformdirs, PyYAML). torch, numpy and all other existing packages are unchanged and `pip check` is clean.
- transformers 5.x requires torch>=2.6, so the `ml` extra pins `transformers<5` and `sentence-transformers<6` (see `docs/ai_review_log.md`).


## Update 2026-10-29
The owner approved torch 2.5.1 -> 2.6.0 (security fixes; `pip check` clean, all three other torch users accept it, retrieval and answers reproduce exactly). numpy stays 1.26.4, transformers stays < 5. Rollback: conda env `torch-backup-2.5.1`. Note for other projects in the env: torch 2.6 defaults `torch.load(weights_only=True)`.
