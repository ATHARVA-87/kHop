import json
from typing import ClassVar

from fastapi.testclient import TestClient

from plexus.serve import app as srv


class FakeHit:
    def __init__(self, doc_id="NVDA_10K_2025-02-26"):
        self.doc_id, self.section, self.text = doc_id, "risk_factors", "We depend on TSMC."


class FakeAnswer:
    text, checks = (
        "TSMC makes our chips [1].",
        [{"sentence": "TSMC makes our chips.", "supported": True, "cited": [1]}],
    )
    hits: ClassVar[list] = [FakeHit()]


class FakeFlow:
    def __init__(self, fail=False):
        self.fail = fail


def fake_events(flow, question):
    if flow.fail:
        raise RuntimeError("ollama down")
    yield "step", {"step": "route", "ms": 0.1}
    yield "token", "TSMC makes "
    yield "token", "our chips [1]."
    yield "done", {"answer": FakeAnswer(), "trace": [{"step": "route"}]}


def _client(monkeypatch, fail=False):
    import tempfile
    from pathlib import Path

    import plexus.agents.flow as flow_mod

    monkeypatch.setattr(srv, "TRACE_LOG", Path(tempfile.mkdtemp()) / "t.jsonl")
    monkeypatch.setattr(flow_mod, "ask_events", fake_events)
    state = {"flow": FakeFlow(fail), "config": "test", **srv.load_meta()}
    return TestClient(srv.create_app(lambda: state))


def _events(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_ask_streams_queued_steps_tokens_done(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.get("/api/ask", params={"q": "Who makes NVIDIA chips?"})
        ev = _events(r.text)
        assert [k for k, _ in ev] == ["queued", "step", "token", "token", "done"]
        done = ev[-1][1]
        assert done["sources"][0]["company"] == "NVIDIA" and done["sources"][0]["url"].startswith(
            "https://www.sec.gov/"
        )
        assert r.headers["content-security-policy"].startswith("default-src 'self'")


def test_validation_rate_limit_and_errors(monkeypatch):
    with _client(monkeypatch) as c:
        assert c.get("/api/ask", params={"q": "hi"}).status_code == 422
        assert c.get("/api/ask", params={"q": "x" * 1001}).status_code == 422
    with _client(monkeypatch, fail=True) as c:
        ev = _events(c.get("/api/ask", params={"q": "Who makes chips?"}).text)
        assert ev[-1][0] == "error" and "ollama serve" in ev[-1][1]["message"]
        # the GPU slot was released after the failure: the next question runs
        assert _events(c.get("/api/ask", params={"q": "Who makes chips?"}).text)[0][0] == "queued"


def test_rate_limiter_and_queue():
    rl = srv.RateLimiter(2)
    assert rl.allow("a", 0) and rl.allow("a", 1) and not rl.allow("a", 2) and rl.allow("a", 61)
    q = srv.GpuQueue(max_waiting=1)
    assert q.join() == 0 and q.join() is None


def test_meta_lists_all_companies(monkeypatch):
    with _client(monkeypatch) as c:
        m = c.get("/api/meta").json()
        assert len(m["companies"]) == 39 and m["filings"] == 156


def test_about_figures_come_from_results(monkeypatch):
    with _client(monkeypatch) as c:
        figs = c.get("/api/about").json()["figures"]
        assert figs and all(f["source"] and f["value"] for f in figs)


class FakeCache:
    def __init__(self):
        self.stored = {}

    def lookup(self, q):
        hit = self.stored.get(q.lower().rstrip("?"))
        return (hit, 0.97, "earlier question") if hit else None

    def store(self, q, payload):
        self.stored[q.lower().rstrip("?")] = payload


def test_cache_stores_clean_answers_and_serves_repeats(monkeypatch):
    import plexus.agents.flow as flow_mod

    monkeypatch.setattr(flow_mod, "ask_events", fake_events)
    monkeypatch.setattr(
        srv, "TRACE_LOG", __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "t.jsonl"
    )
    cache = FakeCache()
    state = {"flow": FakeFlow(), "config": "test", "cache": cache, **srv.load_meta()}
    with TestClient(srv.create_app(lambda: state)) as c:
        first = _events(c.get("/api/ask", params={"q": "Who makes NVIDIA chips?"}).text)
        assert first[-1][0] == "done" and cache.stored
        again = _events(c.get("/api/ask", params={"q": "who makes nvidia chips"}).text)
        assert [k for k, _ in again] == ["step", "done"]  # no queue for a cache hit
        assert again[0][1]["step"] == "cache" and again[-1][1]["from_cache"] is True


def test_refusals_are_not_cached():
    assert not srv.cacheable(
        {
            "answer": "I can't help with investment advice.",
            "checks": [],
            "trace": [{"step": "guard", "refused": True}],
        }
    )
    assert not srv.cacheable(
        {"answer": "x [1].", "checks": [{"supported": False, "cited": [1]}], "trace": []}
    )
    assert srv.cacheable(
        {"answer": "x [1].", "checks": [{"supported": True, "cited": [1]}], "trace": []}
    )
    # an uncited summary line does not block caching (it is not a claim to check)
    assert srv.cacheable(
        {
            "answer": "Summary. x [1].",
            "trace": [],
            "checks": [{"supported": False, "cited": []}, {"supported": True, "cited": [1]}],
        }
    )


def test_cache_hit_needs_no_queue_place(monkeypatch):
    import plexus.agents.flow as flow_mod

    monkeypatch.setattr(flow_mod, "ask_events", fake_events)
    monkeypatch.setattr(
        srv, "TRACE_LOG", __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "t.jsonl"
    )
    monkeypatch.setattr(srv, "MAX_WAITING", 0)  # queue full: new questions get 503
    cache = FakeCache()
    cache.store("old question", {"answer": "A [1].", "sources": [], "checks": [], "trace": []})
    state = {"flow": FakeFlow(), "config": "test", "cache": cache, **srv.load_meta()}
    with TestClient(srv.create_app(lambda: state)) as c:
        assert c.get("/api/ask", params={"q": "brand new question"}).status_code == 503
        ev = _events(c.get("/api/ask", params={"q": "old question"}).text)
        assert ev[-1][1]["from_cache"] is True


def slow_events(flow, question):
    import time as _t

    yield "step", {"step": "route", "ms": 0.1}
    yield "token", "Partial "
    _t.sleep(3)  # a slow writer
    yield "token", "answer [1]."
    yield "done", {"answer": FakeAnswer(), "trace": []}


def test_leaving_mid_answer_does_not_block_others_or_leak(monkeypatch):
    import time as _t

    import plexus.agents.flow as flow_mod

    monkeypatch.setattr(flow_mod, "ask_events", slow_events)
    monkeypatch.setattr(
        srv, "TRACE_LOG", __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "t.jsonl"
    )
    state = {"flow": FakeFlow(), "config": "test", **srv.load_meta()}
    app = srv.create_app(lambda: state)
    with TestClient(app) as c:
        with c.stream("GET", "/api/ask", params={"q": "Who makes NVIDIA chips?"}) as r:
            for line in r.iter_lines():
                if "Partial" in line:
                    break  # the reader leaves while the writer is still working
        t0 = _t.perf_counter()
        assert c.get("/api/health").status_code == 200
        assert _t.perf_counter() - t0 < 1.0  # other requests are not blocked
        for _ in range(50):  # the worker finishes on its own and frees the GPU slot
            if app.state.gpu.waiting == 0 and app.state.gpu.slot.acquire(blocking=False):
                app.state.gpu.slot.release()
                break
            _t.sleep(0.1)
        else:
            raise AssertionError("GPU slot or queue place leaked")
