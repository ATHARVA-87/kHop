"""Phase 6 load test: several people asking at once against `make serve`.

    PLEXUS_LLM_CACHE=<empty file> PLEXUS_SEMANTIC_CACHE=<empty file> PLEXUS_RATE_PER_MIN=1000 make serve
    python -m plexus.eval.load_test --levels 1,2,4,8

Cold: the server runs with empty answer caches, so every new question is really generated.
Each level sends `level` questions at the same time, `rounds` times; questions are unique dev
questions (no reuse between levels). A final "mix" round repeats rewordings of questions
answered earlier (owner-checked cache pairs) to show the semantic cache under load.
One GPU answers one question at a time, so total time grows with the queue: that is the design
(MAX_WAITING = 4; more get a 503 "busy" with Retry-After).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time

import httpx

from plexus.settings import REPO_ROOT


async def one(client: httpx.AsyncClient, base: str, q: str) -> dict:
    t0 = time.perf_counter()
    first = None
    out = {
        "status": None,
        "first_token_s": None,
        "total_s": None,
        "from_cache": False,
        "error": None,
    }
    try:
        async with client.stream("GET", f"{base}/api/ask", params={"q": q}, timeout=600) as r:
            out["status"] = r.status_code
            if r.status_code != 200:
                out["total_s"] = round(time.perf_counter() - t0, 2)
                return out
            event = None
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    if event in ("token", "done") and first is None:
                        first = time.perf_counter() - t0
                    if event == "done":
                        out["from_cache"] = bool(json.loads(line[6:]).get("from_cache"))
                    if event == "error":
                        out["error"] = json.loads(line[6:]).get("message")
    except httpx.HTTPError as exc:
        out["error"] = type(exc).__name__
    out["first_token_s"] = None if first is None else round(first, 2)
    out["total_s"] = round(time.perf_counter() - t0, 2)
    return out


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == 200 and not r["error"]]
    return {
        "requests": len(rows),
        "answered": len(ok),
        "busy_503": sum(r["status"] == 503 for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "first_token_p50_s": pct([r["first_token_s"] for r in ok], 0.5),
        "first_token_p95_s": pct([r["first_token_s"] for r in ok], 0.95),
        "total_p50_s": pct([r["total_s"] for r in ok], 0.5),
        "total_p95_s": pct([r["total_s"] for r in ok], 0.95),
        "cache_hits": sum(r["from_cache"] for r in ok),
    }


async def run(base: str, levels: list[int], rounds: int) -> dict:
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "data/golden/golden.jsonl").read_text().splitlines()
    ]
    qs = [r["question"] for r in rows]
    random.Random(0).shuffle(qs)
    out: dict = {"levels": {}}
    async with httpx.AsyncClient() as client:
        for level in levels:
            results = []
            for _ in range(rounds):
                batch, qs = qs[:level], qs[level:]
                results += await asyncio.gather(*(one(client, base, q) for q in batch))
            out["levels"][str(level)] = summarize(results)
            print(level, out["levels"][str(level)], file=sys.stderr)
        with open(
            REPO_ROOT / "data/golden/cache_pairs.csv", newline="", encoding="utf-8-sig"
        ) as fh:
            pairs = [p for p in csv.DictReader(fh) if p["kind"] == "reuse"][:6]
        mix = []  # 6 originals one by one (stored), then their rewordings + 4 new questions together
        for p in pairs:
            mix.append(await one(client, base, p["original"]))
        mixed = [p["variant"] for p in pairs] + qs[:4]
        mix_round = await asyncio.gather(*(one(client, base, q) for q in mixed))
        out["mix_30pct_rewordings"] = summarize(list(mix_round))
        out["mix_note"] = (
            "6 rewordings of questions answered just before + 4 new questions, sent together"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args(argv)
    out = asyncio.run(run(a.base, [int(x) for x in a.levels.split(",")], a.rounds))
    out["cost_usd"] = 0.0
    (REPO_ROOT / "results" / "phase6_load.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
