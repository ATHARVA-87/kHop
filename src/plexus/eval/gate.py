"""Retrieval regression gate: re-run a config and fail if it scores worse than its baseline.

    python -m plexus.eval.gate --config configs/a1h_bm25.yaml
    python -m plexus.eval.gate --config configs/a1h_bm25.yaml --config configs/a3h_rerank.yaml

The baseline is the committed results/<name>.json. Retrieval is deterministic (fixed seeds,
no LLM calls), so on unchanged code the re-run reproduces the baseline exactly. A PR that
lowers a gated metric by more than `tolerance` fails. A PR that changes scores on purpose
must re-run `make eval` and commit the new results file, so the change shows in review.

The baseline is "stale" (also a failure) when the golden set or corpus differs from the
one it was scored on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from plexus.config import ExperimentConfig
from plexus.eval.compare import compare, format_table
from plexus.settings import REPO_ROOT

# (slice, metric) pairs that must not drop
GATED = [
    ("all", "recall@10"),
    ("all", "ndcg@10"),
    ("all", "mrr"),
    ("all", "recall@3000w"),
    ("local", "recall@10"),
    ("multi_hop", "recall@10"),
]


def check(baseline: dict, current: dict, tolerance: float) -> list[str]:
    problems = []
    bg, cg = baseline.get("golden", {}), current.get("golden", {})
    if bg.get("sha256") != cg.get("sha256"):
        problems.append(
            "stale baseline: golden set changed since the baseline was scored; "
            "re-run `make eval` for this config and commit the results file"
        )
    if baseline.get("corpus_checksum") != current.get("corpus_checksum"):
        problems.append("stale baseline: corpus checksum differs")
    if problems:
        return problems
    for slice_, metric in GATED:
        b = baseline["retrieval"].get(slice_, {}).get(metric)
        c = current["retrieval"].get(slice_, {}).get(metric)
        if b is None or c is None:
            continue
        if c["mean"] < b["mean"] - tolerance:
            problems.append(
                f"{slice_} {metric}: {c['mean']:.4f} < baseline {b['mean']:.4f} - {tolerance}"
            )
    return problems


def run_gate(config: Path, baseline_dir: Path, tolerance: float) -> tuple[bool, str]:
    from plexus.eval.run import run

    cfg = ExperimentConfig.load(config)
    base_path = baseline_dir / f"{cfg.name}.json"
    if not base_path.exists():
        return False, f"{cfg.name}: no baseline at {base_path}"
    baseline = json.loads(base_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        current = run(cfg, results_dir=Path(tmp))
        cur_path = Path(tmp) / f"{cfg.name}.json"
        table = format_table(compare(base_path, cur_path), "baseline", "this_run")
    problems = check(baseline, current, tolerance)
    lines = [f"== gate {cfg.name}: {'FAIL' if problems else 'pass'}", table, *problems]
    return not problems, "\n".join(lines)


def golden_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True, type=Path)
    ap.add_argument("--baseline-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--tolerance", type=float, default=0.005)
    args = ap.parse_args(argv)
    ok_all = True
    for c in args.config:
        ok, report = run_gate(c, args.baseline_dir, args.tolerance)
        print(report, flush=True)
        ok_all &= ok
    print("GATE PASSED" if ok_all else "GATE FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
