# ADR 0008: The web app is a static Next.js export served by FastAPI

Status: accepted (2026-10-29; design approved by the owner)

## Context
The brief asks for FastAPI with Server-Sent Events and a Next.js UI. Next.js published critical advisories for its server runtime in 2026 (versions up to 16.4.0: RCE, DoS; earlier middleware bypasses). Plexus runs on one laptop GPU and later on a free Hugging Face Space with one open port.

## Decision
- Next.js 16.4.1 (the first fixed version that is at least two weeks old) with `output: "export"`: the build is plain HTML/JS/CSS in `web/out`, and there is **no Next.js server at runtime**. FastAPI serves it next to the API on one port (`make serve`), so the same container works on a Space.
- The page reads the answer stream with `fetch` (not `EventSource`), so it can show the server's own messages for 429/503 and never reconnects by itself (a reconnect would ask the question again).
- Filing and answer text is rendered as text only (never `dangerouslySetInnerHTML`); the server sends a Content-Security-Policy, `X-Frame-Options: DENY`, `nosniff` and `no-referrer`.
- Fonts are downloaded at build time and served by Plexus (no request from the reader to Google). Next.js telemetry is off.
- One question on the GPU at a time, at most 4 waiting, 10 questions per minute per client.

## Consequences
- No server-side rendering, API routes or image optimisation: not needed for a single page that streams from our API.
- Vite would have worked equally well; Next.js is kept because the brief names it and the static export removes its server-side risk ("Next.js as a static site generator").
- `make ui` must be re-run after UI changes; `make ui-dev` gives live reload against `make serve` (dev only).
