# ADR 0013: Where the public demo runs

Status: **Accepted: option A** (owner, 2026-10-30)
Date: 2026-10-29

## Context
The brief planned a free Hugging Face Docker Space. Research (2026-10-29, sources below) found:
- Since July 2026, new Docker and Gradio Spaces on free CPU hardware need the paid PRO plan. Static Spaces are still free.
- The Groq free tier (for a hosted writer) allows about 1,000 requests and 200,000 tokens per day, with 8,000 tokens per minute. That is tight for retrieval prompts, but it costs nothing without a card.
- Free ZeroGPU Spaces exist only for the Gradio SDK, with a small daily GPU quota per visitor. Whether our FastAPI + Next.js app can run there is unverified.

Sources:
- https://huggingface.co/docs/hub/spaces-overview
- https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580
- https://console.groq.com/docs/rate-limits
- https://huggingface.co/docs/hub/spaces-zerogpu

## Options
| Option | Cost | What a visitor gets | Risk |
| --- | --- | --- | --- |
| A. Static page replaying real recorded runs (same UI; trace, citations and fact-check marks), clearly labelled as recorded, plus a short video of the live app | free | the real interface and real answers for a fixed set of questions; cannot type new ones | none to quota or cost |
| B. Gradio app on a free ZeroGPU Space | free | new questions, a different UI | Gradio only; GPU quota per visitor; port of the app unverified |
| C. Docker Space on PRO | paid | the full app | breaks the free-only rule |
| D. Local only, plus the video | free | nothing hosted | the definition of done asks for a hosted UI |

## Recommendation
**A.** It is free, it shows the real interface and real outputs, and nothing can run up a bill or be abused. The full app stays one `make serve` away for anyone who clones the repo.

## Decision
**Option A**, chosen by the owner on 2026-10-30.

**How it is built:**
- `make record-demo` starts the app with fresh, empty caches, so every answer is generated live.
- It asks each demo question the way the page does. For the typo question it records both "Use this version" and "Ask as I typed".
- It saves every event with the time it arrived (`web/public/replay/`).
- `make demo-site` builds the same page in replay mode: its calls read the recordings, and answers stream back at their recorded speed. A note on the page says the answers are recorded, and a question that was not recorded gets a plain message.

**Hosting:** GitHub Pages, from the public repo (`.github/workflows/demo-site.yml`). It deploys only from `main`, so the page changes only when the owner merges. It needs no Hugging Face account, no server, no model and no keys.
