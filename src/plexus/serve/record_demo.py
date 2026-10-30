"""Record real runs of the app for the static demo (ADR 0013, option A).

    make serve                # in another terminal, started with fresh caches (make record-demo does it)
    python -m plexus.serve.record_demo

Asks the running app each demo question exactly as the page would (reframe first; for a changed
question both "Use this version" and "Ask as I typed"), and saves every event with the time it
arrived, so the demo replays real answers at their real speed. Writes web/public/replay/.
Only the demo questions below and public SEC text are saved.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request

from plexus.settings import REPO_ROOT

OUT = REPO_ROOT / "web" / "public" / "replay"
SERVER = "http://127.0.0.1:8000"
DEMO = [  # the page's own examples first, then one of each behaviour the demo script shows
    ("Two companies", "What do Impinj and Allegro MicroSystems each rely on TSMC for?"),
    ("Two companies", "How do Lam Research and KLA describe their exposure to Huawei under export restrictions?"),
    ("One company", "What natural disaster risks does NXP face at its factories in places like Singapore, Taiwan, Malaysia and Thailand?"),
    ("Whole sector", "How do all semiconductor companies describe the risks that tariffs and trade policy pose to their costs and supply chains?"),
    ("A typo in the name", "How did nvdia companies perform in 2024"),
    ("Two companies", "Which suppliers do NVIDIA and AMD both depend on?"),
    ("Declined: investment advice", "Should I buy NVIDIA stock?"),
]  # fmt: skip


def get_json(path: str) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=60) as r:
        return json.load(r)


def record_ask(question: str) -> list[dict]:
    """Every Server-Sent Event of one answer, with ms since the question was sent."""
    url = f"{SERVER}/api/ask?q={urllib.parse.quote(question)}"
    events, event, data = [], "message", ""
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith(":"):
                continue  # keep-alive comment
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
            elif not line and data:
                ms = round((time.perf_counter() - t0) * 1000)
                events.append({"t": ms, "event": event, "data": json.loads(data)})
                event, data = "message", ""
    if not events or events[-1]["event"] not in {"done", "error"}:
        raise RuntimeError(f"answer did not finish: {question!r}")
    if events[-1]["event"] == "error":
        raise RuntimeError(f"the app returned an error for {question!r}: {events[-1]['data']}")
    return events


def main() -> int:
    for sub in ("ask", "reframe"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for old in (OUT / sub).glob("*.json"):
            old.unlink()
    index: dict = {
        "recorded": dt.date.today().isoformat(),
        "examples": [],
        "ask": {},
        "reframe": {},
    }
    (OUT / "meta.json").write_text(json.dumps(get_json("/api/meta"), indent=1) + "\n")
    (OUT / "about.json").write_text(json.dumps(get_json("/api/about"), indent=1) + "\n")
    for n, (kind, q) in enumerate(DEMO, 1):
        index["examples"].append({"kind": kind, "q": q})
        r = get_json(f"/api/reframe?q={urllib.parse.quote(q)}")
        rid = f"r{n}"
        (OUT / "reframe" / f"{rid}.json").write_text(json.dumps(r, indent=1) + "\n")
        index["reframe"][q] = rid
        asks = [r["question"], q] if r.get("changed") else [r["question"]]
        for m, question in enumerate(asks):
            aid = f"a{n}{'bcdef'[m - 1] if m else ''}"
            events = record_ask(question)
            (OUT / "ask" / f"{aid}.json").write_text(json.dumps(events) + "\n")
            index["ask"][question] = aid
            print(f"{aid}: {events[-1]['t'] / 1000:.1f} s, {len(events)} events  {question}")
            time.sleep(2)
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"recorded {len(index['ask'])} answers into {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
