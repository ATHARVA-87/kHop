"""Semantic cache (A8) evaluation on owner-checked pairs: python -m plexus.eval.cache_eval

Rule fixed before the first run (2026-10-29):
- tune on pairs C-01..C-20: threshold = the LOWEST value in 0.80..0.99 (step 0.005) with zero
  wrong reuses on the tune "new" rows (key + similarity, as in the app);
- report once on C-21..C-40: reuse rate on rewordings, wrong-reuse rate on near-misses with a
  rule-of-three upper bound when it is zero, and the same threshold WITHOUT the
  entity/period key (the ablation that shows why the key is there).
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

from plexus.settings import REPO_ROOT

PAIRS = REPO_ROOT / "data" / "golden" / "cache_pairs.csv"


def load_pairs() -> list[dict]:
    with open(PAIRS, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    unchecked = [r["pid"] for r in rows if r["label_is_right"].strip().lower() not in ("yes", "no")]
    if unchecked:
        raise SystemExit(f"owner check missing (label_is_right yes/no) for: {unchecked[:5]}...")
    return [r for r in rows if r["label_is_right"].strip().lower() == "yes"]


def decide(rows: list[dict], threshold: float, use_key: bool) -> list[bool]:
    return [(r["same_key"] or not use_key) and r["sim"] >= threshold for r in rows]


def rates(rows: list[dict], reused: list[bool]) -> dict:
    pos = [u for r, u in zip(rows, reused, strict=True) if r["kind"] == "reuse"]
    neg = [u for r, u in zip(rows, reused, strict=True) if r["kind"] == "new"]
    fp = sum(neg)
    return {
        "reused_rewordings": f"{sum(pos)}/{len(pos)}",
        "reuse_rate": round(sum(pos) / len(pos), 3) if pos else None,
        "wrong_reuses": f"{fp}/{len(neg)}",
        "wrong_reuse_rate": round(fp / len(neg), 3) if neg else None,
        "wrong_reuse_upper_95": round(3 / len(neg), 3)
        if neg and fp == 0
        else None,  # rule of three
    }


def main() -> int:
    from plexus.config import ExperimentConfig
    from plexus.index.dense import Embedder
    from plexus.retrieve.pipeline import make_linker
    from plexus.serve.semantic_cache import bge_encoder, key_of, normalize
    from plexus.settings import Settings

    rows = load_pairs()
    cfg = ExperimentConfig.load(REPO_ROOT / "configs" / "app.yaml")
    s = Settings.load()
    linker = make_linker(cfg, s.data_dir)
    encode = bge_encoder(Embedder(cfg.embedder))
    a = encode([normalize(r["original"]) for r in rows])
    b = encode([normalize(r["variant"]) for r in rows])
    for r, x, y in zip(rows, a, b, strict=True):
        r["sim"] = float(np.dot(x, y))
        r["same_key"] = key_of(r["original"], linker) == key_of(r["variant"], linker)
    tune = [r for r in rows if int(r["pid"][2:4]) <= 20]
    test = [r for r in rows if int(r["pid"][2:4]) > 20]
    grid = [round(0.80 + 0.005 * i, 3) for i in range(39)]
    ok = [t for t in grid if not any(decide([r for r in tune if r["kind"] == "new"], t, True))]
    threshold = min(ok) if ok else 0.99
    out = {
        "threshold": threshold,
        "tune": rates(tune, decide(tune, threshold, True)),
        "test": rates(test, decide(test, threshold, True)),
        "test_without_key": rates(test, decide(test, threshold, False)),
        "near_misses_blocked_by_key_on_test": sum(
            1 for r in test if r["kind"] == "new" and not r["same_key"]
        ),
        "per_pair": [{k: r[k] for k in ("pid", "kind", "what_changed", "same_key")} | {"sim": round(r["sim"], 4)} for r in rows],
    }  # fmt: skip
    (REPO_ROOT / "results" / "a8_cache.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_pair"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
