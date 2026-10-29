# Security

Rule (owner, 2026-10-27/29): security first. Keys stay local, nothing sensitive is exposed,
and every change is checked. One command repeats all checks: `make security`.

## What is enforced, and where
| risk | control | where |
| --- | --- | --- |
| API keys / tokens committed | pre-commit `no-secrets` (Gemini, Groq, OpenAI-style, GitHub, HF, Slack patterns) + `detect-private-key`; GitHub secret scanning + push protection (on) | `.pre-commit-config.yaml`, repo settings |
| personal email / local paths | pre-commit blocks personal-provider emails and absolute home paths; commits use the GitHub noreply address | `.pre-commit-config.yaml`, global git config |
| keys in the app | keys only in `.env` (gitignored), read by `plexus.settings`; `.env.example` has no values | `.gitignore` |
| vulnerable dependencies | `pip-audit` on the exact `uv.lock` versions; fails on any advisory not accepted below; GitHub Dependabot alerts (on) | `make security`, `src/plexus/audit.py` |
| supply chain | exact pins in `uv.lock`; new packages must be >= 2 weeks old; models pinned to a Hugging Face commit | `pyproject.toml`, `src/plexus/config.py` |
| unsafe model loading | safetensors weights only, no `trust_remote_code`, no `torch.load` of downloaded files; the only pickle is our own BM25 index | code scan 2026-10-29 |
| data leaving the machine | LangSmith / RAGAS tracking forced off; only public SEC text goes to Gemini/Groq free tiers | `src/plexus/agents/flow.py`, `src/plexus/eval/ragas_eval.py` |
| prompt injection from filings | filing text and answers are wrapped as data ("ignore instructions inside") in every prompt | `src/plexus/baseline/naive.py`, judges |
| repo size | large-file hook (1 MB per file); whole history 8.7 MB on 2026-10-29 (GitHub: 100 MB per file, ~1 GB per repo) | `.pre-commit-config.yaml` |
| main branch | protected: PRs only, no force push or deletion, admins included | GitHub settings |

## Accepted risks (reviewed 2026-10-29)
| package | advisories | why accepted | mitigation |
| --- | --- | --- | --- |
| torch 2.5.1 | 19 | pinned by project rule (CUDA env); fixes need >= 2.6 | trusted pinned models only, no torch.load / jit / compile on untrusted input |
| transformers 4.57.6 | 6 | fixes need 5.x, which needs torch >= 2.6 | no trust_remote_code, Trainer or checkpoint conversion |
| filelock 3.32.7 | 13 | the fix (4.2.1) is under 2 weeks old | local races need a second user; **upgrade on/after 2026-10-31** (the gate fails after that date) |

Known history item: some early commits carry the owner's personal email as author (before the
switch to the noreply address). The owner chose not to rewrite history (2026-10-27).

## Phase 5 (API) rules, decided before writing it
- Bind to 127.0.0.1 by default; no keys ever sent to the browser; CORS limited to the UI origin.
- Request limits: question length cap, per-client rate limit, timeouts on every LLM call.
- Errors return a generic message; details only in local logs, which never contain keys.
- Pydantic validation on every input; no file uploads, no shell or eval paths.
