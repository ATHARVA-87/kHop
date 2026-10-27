"""Paired comparison of two result files on the same queries.

    python -m plexus.eval.compare results/a0_naive.json results/a0_bm25.json

For each slice (all / local / multi_hop / global) and metric, reports mean(B - A) with a
paired bootstrap 95% CI over queries. A CI that excludes 0 means the change is unlikely to
be resampling noise on this golden set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plexus.eval.bootstrap import paired_bootstrap

DEFAULT_METRICS = ("recall@10", "complete@10", "ndcg@10", "mrr", "company_coverage@10")


def _per_query(path: Path) -> dict[str, dict]:
    return {r["qid"]: r for r in json.loads(path.read_text())["per_query"]}


def compare(a: Path, b: Path, metrics=DEFAULT_METRICS, seed: int = 0) -> dict:
    qa, qb = _per_query(a), _per_query(b)
    if set(qa) != set(qb):
        raise ValueError(f"different query sets: {len(set(qa) ^ set(qb))} qids differ")
    out: dict[str, dict] = {}
    for slice_ in ("all", "local", "multi_hop", "global"):
        qids = sorted(q for q in qa if slice_ == "all" or qa[q]["qtype"] == slice_)
        for m in metrics:
            kept = [q for q in qids if m in qa[q] and m in qb[q]]
            pairs = [(qa[q][m], qb[q][m]) for q in kept]
            if not pairs:
                continue
            clusters = [qa[q].get("cluster", q) for q in kept]
            d = paired_bootstrap(
                [float(x) for x, _ in pairs],
                [float(y) for _, y in pairs],
                seed=seed,
                clusters=clusters,
            )
            out.setdefault(slice_, {})[m] = {
                "a": round(sum(float(x) for x, _ in pairs) / len(pairs), 4),
                "b": round(sum(float(y) for _, y in pairs) / len(pairs), 4),
                **d.to_dict(),
            }
    return out


def format_table(res: dict, name_a: str, name_b: str) -> str:
    lines = [f"{'slice':<10} {'metric':<20} {name_a:>9} {name_b:>9}   diff [95% CI]"]
    for slice_, ms in res.items():
        for m, r in ms.items():
            d = r["diff"]
            flag = " *" if d["low"] > 0 or d["high"] < 0 else ""
            lines.append(
                f"{slice_:<10} {m:<20} {r['a']:>9.3f} {r['b']:>9.3f}   "
                f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]{flag}"
            )
    lines.append("* = 95% CI excludes 0")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = compare(args.a, args.b)
    print(json.dumps(res, indent=2) if args.json else format_table(res, args.a.stem, args.b.stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
