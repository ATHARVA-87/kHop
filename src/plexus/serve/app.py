"""Plexus web service (Phase 6): one question in, live events out (Server-Sent Events).

    make serve            # http://127.0.0.1:8000  (UI + API on one port)

GET /api/ask?q=...  streams events while the answer is made:
    queued {position}        the question waits for the GPU (one question at a time)
    step   {step, ms, ...}   a step of the flow finished (guard, route, retrieve, answer, verify)
    token  {text}            a piece of the answer as the model writes it
    done   {answer, sources, checks, trace}
    error  {message}         what went wrong and what to do
GET /api/health, GET /api/meta (companies covered, for the start screen).

Safety (docs/security.md): binds to 127.0.0.1 by default; question length 3-1000; per-client
rate limit; at most MAX_WAITING questions queued (else 503 + Retry-After); one question on the
GPU at a time (the 6 GB card holds one writer + search + verifier); a hard deadline per
question; generic error messages (details only in the local log); security headers; filing
text is sent as plain text (the UI never renders it as HTML).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from plexus.settings import REPO_ROOT, Settings

log = logging.getLogger("plexus.serve")

CONFIG = Path(os.getenv("PLEXUS_CONFIG", REPO_ROOT / "configs" / "app.yaml"))
MAX_WAITING = 4  # questions allowed to wait for the GPU
RATE_PER_MIN = int(os.getenv("PLEXUS_RATE_PER_MIN", "10"))  # per client; the load test raises it
DEADLINE_S = 120  # one question, start to finish
UI_DIR = REPO_ROOT / "web" / "out"  # the static Next.js export
CACHE_THRESHOLD = 0.935  # chosen on cache pairs C-01..C-20, results/a8_cache.json
TRACE_LOG = Path(os.getenv("PLEXUS_TRACE_LOG", REPO_ROOT / "data" / "logs" / "traces.jsonl"))


class RateLimiter:
    """Sliding one-minute window per client address."""

    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return False
            q.append(now)
            return True


class GpuQueue:
    """One question on the GPU at a time; a few may wait; the rest are told to retry."""

    def __init__(self, max_waiting: int):
        self.slot = threading.BoundedSemaphore(1)
        self.max_waiting = max_waiting
        self.waiting = 0
        self.lock = threading.Lock()

    def join(self) -> int | None:
        """Place in the queue (0 = next), or None when full."""
        with self.lock:
            if self.waiting >= self.max_waiting:
                return None
            self.waiting += 1
            return self.waiting - 1

    def acquire(self, timeout: float) -> bool:
        try:
            return self.slot.acquire(timeout=timeout)
        finally:
            with self.lock:
                self.waiting -= 1

    def release(self) -> None:
        self.slot.release()


def load_meta() -> dict:
    names = yaml.safe_load((REPO_ROOT / "configs" / "companies.yaml").read_text())
    docs = {}
    manifest = REPO_ROOT / "data" / "snapshots" / "semis-2025.manifest.jsonl"
    for line in manifest.read_text().splitlines():
        r = json.loads(line)
        docs[r["doc_id"]] = {"company": names.get(r["ticker"], r["company"]), "ticker": r["ticker"],
                             "form": r["form"], "filed": r["filing_date"], "url": r["url"]}  # fmt: skip
    return {"names": names, "docs": docs}


SECTION_NAMES = {
    "risk_factors": "Risk factors", "mdna": "Management's discussion", "business": "Business",
    "financial_statements": "Financial statements", "market_risk": "Market risk",
    "legal_proceedings": "Legal proceedings", "controls": "Controls and procedures",
    "market_for_equity": "Market for the stock", "directors_officers": "Directors and officers",
    "cover": "Cover page", "properties": "Properties", "unknown": "Filing text",
}  # fmt: skip


def source_of(n: int, hit, docs: dict) -> dict:
    d = docs.get(hit.doc_id, {})
    return {
        "n": n,
        "company": d.get("company", hit.doc_id),
        "form": d.get("form", ""),
        "filed": d.get("filed", ""),
        "url": d.get("url", ""),
        "section": SECTION_NAMES.get(hit.section, hit.section.replace("_", " ").capitalize()),
        "text": hit.text,  # plain text; the UI renders it as text, never as HTML
    }


def _json(rel: str):
    path = REPO_ROOT / rel
    return json.loads(path.read_text()) if path.exists() else None


def about_figures() -> list[dict]:
    """Measured results for the About page, read from results/*.json at request time (never
    typed in by hand). A figure whose file is missing is left out, not shown as a blank."""
    out: list[dict] = []

    def add(label: str, value: str, source: str):
        out.append({"label": label, "value": value, "source": source})

    def recall(rel: str):
        d = _json(rel)
        return d["retrieval"]["all"]["recall@10"]["mean"] if d else None

    a, b = recall("results/a0_naive.json"), recall("results/a6_filter.json")
    if a is not None and b is not None:
        add(
            "Questions where the right passages are in the top 10, first version vs now (134 practice questions)",
            f"{a:.0%} to {b:.0%}",
            "results/a0_naive.json, results/a6_filter.json",
        )
    a, b = recall("results/test/a0_naive.json"), recall("results/test/a6_router.json")
    if a is not None and b is not None:
        add(
            "The same on 91 held-back questions, scored once before the latest improvement",
            f"{a:.0%} to {b:.0%}",
            "results/test/a0_naive.json, results/test/a6_router.json",
        )
    labels = REPO_ROOT / "data" / "golden" / "judge_labels.csv"
    if labels.exists():
        import csv

        with open(labels, newline="", encoding="utf-8-sig") as fh:
            v = [
                r["human_verdict"].strip().lower()
                for r in csv.DictReader(fh)
                if r["human_verdict"].strip()
            ]
        add(
            "Answers fully correct, checked by hand",
            f"{v.count('correct')} of {len(v)}",
            "data/golden/judge_labels.csv",
        )
    d = _json("results/verifier_validation.json")
    if d:
        m = d["summary"]["minicheck"]
        n_bad = d["summary"]["human_unsupported"]
        add(
            "Unsupported sentences the fact-checker catches (hand-labelled test)",
            f"{round(m['tpr_catches_not_correct'] * n_bad)} of {n_bad}",
            "results/verifier_validation.json",
        )
    d = _json("results/guardrails_a9_agents.json")
    if d:
        h = d["summary"]["refusals_heldout"]["refuse_advice"]
        add(
            "Investment-advice questions declined (fresh test set)",
            f"{h['ok']} of {h['n']}",
            "results/guardrails_a9_agents.json",
        )
        i = d["summary"]["injection"]
        add(
            'Attack test: fake orders such as "ignore your rules" were hidden inside filing text. Times Plexus obeyed them',
            f"{i['attack_success']} of {i['n']}",
            "results/guardrails_a9_agents.json",
        )
    d = _json("results/latency_a9_agents.json")
    if d:
        add(
            "Typical time to a full answer on a laptop GPU",
            f"{d[0]['total_ms_p50'] / 1000:.0f} seconds",
            "results/latency_a9_agents.json",
        )
    return out


def ollama_writer(config: Path = CONFIG) -> tuple[str, str] | None:
    """(api base, model) when the app's writer runs on Ollama, else None."""
    from plexus.config import ExperimentConfig

    g = ExperimentConfig.load(config).generation
    return (Settings.load().ollama_api_base, g.model) if g.provider == "ollama" else None


def ollama_load(base: str, model: str, keep_alive: int) -> None:
    """Load (keep_alive=-1: until told otherwise) or unload (0) the writer model. A model
    management request with no prompt: it generates nothing, so it is not an LLM call."""
    import httpx

    try:
        httpx.post(
            f"{base}/api/generate", json={"model": model, "keep_alive": keep_alive}, timeout=120
        )
    except httpx.HTTPError:
        log.warning(
            "could not reach Ollama at %s to %s %s", base, "load" if keep_alive else "unload", model
        )


def shared_embedder(retriever):
    """The search's own bge-small model, reused by the cache instead of a second copy on the
    GPU (each copy ~100 MB; the app was 20 MB over the owner's 10% GPU margin)."""
    for base in getattr(retriever, "closables", None) or []:
        for r in getattr(base, "retrievers", None) or []:
            if hasattr(r, "embedder"):
                return r.embedder
    return None


def build_state(config: Path = CONFIG) -> dict:
    from plexus.agents.flow import build_flow
    from plexus.baseline.naive import NaiveRAG
    from plexus.config import ExperimentConfig
    from plexus.llm import LLMClient
    from plexus.retrieve.pipeline import make_retriever

    s = Settings.load()
    cfg = ExperimentConfig.load(config)
    retriever = make_retriever(cfg, s.data_dir)
    flow = build_flow(NaiveRAG(cfg, retriever, LLMClient.from_settings(s)))
    cache = None
    router = getattr(retriever, "router", None)
    if router is not None:  # the cache key needs the company linker of the A6 router
        from plexus.index.dense import Embedder
        from plexus.serve.semantic_cache import SemanticCache, bge_encoder, scope_of

        cache_path = Path(
            os.getenv("PLEXUS_SEMANTIC_CACHE", s.data_dir / "cache" / "semantic.sqlite")
        )
        cache = SemanticCache(cache_path, bge_encoder(shared_embedder(retriever) or Embedder(cfg.embedder)),
                              router.linker, scope_of(cfg.snapshot, config), CACHE_THRESHOLD)  # fmt: skip
    return {"flow": flow, "retriever": retriever, "config": cfg.name, "cache": cache, **load_meta()}


def log_trace(question: str, record: dict) -> None:
    """One line per finished question: step timings, cache hit, queue wait. No question text,
    only a short hash (the log stays on this machine; docs/security.md)."""
    import hashlib

    record = {
        "ts": round(time.time(), 1),
        "q": hashlib.sha256(question.encode()).hexdigest()[:10],
        **record,
    }
    try:
        TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # the log must never break an answer
        log.exception("trace log write failed")


def cacheable(payload: dict) -> bool:
    """Store only clean answers: something was claimed and every sentence passed the check."""
    from plexus.baseline.naive import is_refusal

    refused = any(t.get("refused") for t in payload.get("trace", []))
    checks = payload.get("checks") or []
    return (
        not refused
        and not is_refusal(payload["answer"])
        and bool(checks)
        and all(c["supported"] for c in checks)
    )


def run_question(flow, question: str, docs: dict):
    """The events for one question (sync generator, runs in FastAPI's threadpool)."""
    from plexus.agents.flow import ask_events

    for kind, data in ask_events(flow, question):
        if kind == "token":
            yield ServerSentEvent(event="token", data={"text": data})
        elif kind == "step":
            yield ServerSentEvent(event="step", data=data)
        else:
            ans = data["answer"]
            yield ServerSentEvent(event="done", data={
                "answer": ans.text,
                "sources": [source_of(i, h, docs) for i, h in enumerate(ans.hits, 1)],
                "checks": ans.checks,
                "trace": data["trace"],
            })  # fmt: skip


def create_app(state_factory=build_state) -> FastAPI:
    state: dict = {}
    limiter = RateLimiter(RATE_PER_MIN)
    gpu = GpuQueue(MAX_WAITING)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        writer = ollama_writer()
        if writer and state_factory is build_state:
            # the writer goes on the GPU FIRST, while it is empty: qwen3.5:4b also reserves
            # ~0.9 GB for an image encoder we never use, so loaded after the search models it
            # spilled 35% onto the CPU and the laptop lagged (2026-10-29). It stays loaded while
            # the app runs and is unloaded when the app stops.
            os.environ.setdefault("PLEXUS_OLLAMA_KEEP_ALIVE", "-1")
            ollama_load(*writer, keep_alive=-1)
        state.update(state_factory())
        yield
        r = state.get("retriever")
        if r is not None and hasattr(r, "close"):
            r.close()
        if writer and state_factory is build_state:
            ollama_load(*writer, keep_alive=0)  # free the GPU

    app = FastAPI(title="Plexus", lifespan=lifespan, docs_url=None, redoc_url=None)

    if os.getenv("PLEXUS_DEV_CORS"):  # only for `next dev` on another port
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"],
                           allow_methods=["GET"], allow_headers=[])  # fmt: skip

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' "
            "'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        return response

    @app.get("/api/health")
    def health():
        return {"ok": bool(state), "config": state.get("config")}

    @app.get("/api/meta")
    def meta():
        names = state.get("names", {})
        return {"companies": sorted(names.values(), key=str.lower), "filings": len(state.get("docs", {})),
                "years": "2025"}  # fmt: skip

    def admit(request: Request, q: str = Query(min_length=3, max_length=1000)) -> tuple:
        """Runs before streaming starts, so limits return proper status codes. FastAPI validates
        `q` first: an invalid question never takes a place in the queue. The semantic cache is
        checked BEFORE the queue, so a reused answer never waits for or blocks the GPU (load
        test 2026-10-29: hits were being turned away as busy)."""
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "Too many questions. Wait a minute, then ask again.",
                                headers={"Retry-After": "60"})  # fmt: skip
        question = q.strip()
        cache = state.get("cache")
        if cache is not None:
            t0 = time.perf_counter()
            try:
                hit = cache.lookup(question)
            except Exception:  # a broken cache must never block an answer
                log.exception("cache lookup failed")
                hit = None
            if hit is not None:
                return question, None, (hit, round((time.perf_counter() - t0) * 1000, 1))
        position = gpu.join()
        if position is None:
            raise HTTPException(503, "Plexus is busy answering other questions. Try again shortly.",
                                headers={"Retry-After": "20"})  # fmt: skip
        return question, position, None

    @app.get("/api/about")
    def about():
        return {"figures": about_figures()}

    @app.get("/api/ask", response_class=EventSourceResponse)
    def ask(admitted: tuple = Depends(admit)):
        question, position, cached = admitted
        t_start = time.perf_counter()
        cache = state.get("cache")
        if cached is not None:  # answered from the cache: no queue, no GPU
            (payload, sim, original), lookup_ms = cached
            yield ServerSentEvent(event="step", data={
                "step": "cache", "ms": lookup_ms, "similar_to": original, "similarity": round(sim, 3)})  # fmt: skip
            yield ServerSentEvent(event="done", data={**payload, "from_cache": True})
            log_trace(question, {"cache_hit": True, "queue_ms": 0, "total_ms": lookup_ms})
            return
        yield ServerSentEvent(event="queued", data={"position": position})
        t_wait = time.perf_counter()
        if not gpu.acquire(timeout=DEADLINE_S):
            yield ServerSentEvent(
                event="error", data={"message": "Plexus is busy. Try again shortly."}
            )
            return
        queue_ms = round((time.perf_counter() - t_wait) * 1000, 1)
        start = time.monotonic()
        try:
            for ev in run_question(state["flow"], question, state["docs"]):
                if time.monotonic() - start > DEADLINE_S:
                    yield ServerSentEvent(event="error", data={
                        "message": "This answer took too long. Try a shorter or more specific question."})  # fmt: skip
                    return
                if ev.event == "done":
                    trace = ev.data.get("trace", [])
                    log_trace(question, {
                        "cache_hit": False, "queue_ms": queue_ms,
                        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
                        "steps": {t.get("step"): t.get("ms") for t in trace},
                        "route": next((t.get("route") for t in trace if t.get("step") == "route"), None),
                    })  # fmt: skip
                if ev.event == "done" and cache is not None and cacheable(ev.data):
                    try:
                        cache.store(question, ev.data)
                    except Exception:
                        log.exception("cache store failed")
                yield ev
        except Exception:  # details stay in the local log; the reader gets a clear next step
            log.exception("question failed")
            yield ServerSentEvent(event="error", data={
                "message": "Plexus could not answer. Check that Ollama is running (ollama serve), then ask again."})  # fmt: skip
        finally:
            gpu.release()  # also runs when the reader closes the page

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            {"message": exc.detail}, status_code=exc.status_code, headers=exc.headers
        )

    if UI_DIR.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app


app = create_app()
