"""Per-step timings of the live app, from its trace log: python -m plexus.eval.stage_times

Reads data/logs/traces.jsonl (one line per finished question, written by the server) and
reports p50 / p95 per step, queue wait, total, and the cache hit rate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from plexus.settings import REPO_ROOT


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def summarize(records: list[dict]) -> dict:
    fresh = [r for r in records if not r.get("cache_hit")]
    steps: dict[str, list] = {}
    for r in fresh:
        for k, v in (r.get("steps") or {}).items():
            steps.setdefault(k, []).append(v)
    return {
        "questions": len(records),
        "cache_hit_rate": round(sum(bool(r.get("cache_hit")) for r in records) / len(records), 3)
        if records
        else None,
        "step_ms": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)} for k, v in steps.items()},
        "queue_ms": {
            "p50": pct([r.get("queue_ms") for r in fresh], 0.5),
            "p95": pct([r.get("queue_ms") for r in fresh], 0.95),
        },
        "total_ms_fresh": {
            "p50": pct([r.get("total_ms") for r in fresh], 0.5),
            "p95": pct([r.get("total_ms") for r in fresh], 0.95),
        },
        "total_ms_cached": {
            "p50": pct([r.get("total_ms") for r in records if r.get("cache_hit")], 0.5)
        },
    }


def main() -> int:
    path = Path(os.getenv("PLEXUS_TRACE_LOG", REPO_ROOT / "data" / "logs" / "traces.jsonl"))
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = summarize(records)
    (REPO_ROOT / "results" / "phase6_stages.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
