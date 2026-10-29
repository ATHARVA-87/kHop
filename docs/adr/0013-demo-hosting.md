# ADR 0013: Where the public demo runs

Status: **Open: needs the owner's choice** (free-only rule)
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
Pending the owner.
