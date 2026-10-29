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
        [{"sentence": "TSMC makes our chips.", "supported": True}],
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
    import plexus.agents.flow as flow_mod

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
