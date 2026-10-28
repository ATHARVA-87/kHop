"""Router evaluation (Phase 4 gate): accuracy against the golden question types, and the
oracle router (gold type) as the upper bound for end-to-end retrieval.

    python -m plexus.eval.routing --config configs/a6_router.yaml
    python -m plexus.eval.routing --config configs/a6_router.yaml --golden data/golden/golden_test.jsonl --results-dir results/test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from plexus.config import ExperimentConfig
from plexus.eval.golden import load_golden
from plexus.eval.run import aggregate, score_retrieval
from plexus.ingest.process import iter_docs
from plexus.retrieve.pipeline import make_retriever
from plexus.retrieve.router import ROUTES, OracleRouter
from plexus.settings import REPO_ROOT, Settings


def confusion(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (gold, predicted) -> accuracy and a gold x predicted count table."""
    table = {g: {p: 0 for p in ROUTES} for g in ROUTES}
    for g, p in pairs:
        table[g][p] += 1
    acc = sum(g == p for g, p in pairs) / max(len(pairs), 1)
    return {"accuracy": round(acc, 4), "n": len(pairs), "gold_x_predicted": table}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/a6_router.yaml")
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    a = ap.parse_args(argv)
    s = Settings.load()
    cfg = ExperimentConfig.load(a.config)
    queries = load_golden(a.golden or s.data_dir / "golden" / "golden.jsonl")
    retriever = make_retriever(cfg, s.data_dir)
    router = retriever.router
    pairs = [(q.qtype, router.route(q.question)) for q in queries]
    errors = [{"qid": q.qid, "gold": g, "routed": p, "question": q.question}
              for q, (g, p) in zip(queries, pairs, strict=True) if g != p]  # fmt: skip
    # oracle: same retrievers, gold type decides the route
    gold = {q.question: q.qtype for q in queries}
    retriever.router = OracleRouter(lambda question: gold[question])
    docs = {m["doc_id"]: d.text for m, d in iter_docs(s.data_dir / "processed", cfg.snapshot)}
    from plexus.eval.global_themes import load_labels, load_themes

    labels = load_labels(s.data_dir / "golden" / "global_themes.csv")
    l2k = {label: kw for kw, label in load_themes().items()}
    rows, label_errors = score_retrieval(queries, retriever, docs, cfg.retrieval.top_k, labels, l2k)
    retriever.close()
    out = {
        "config": "a6_oracle (gold question type as route)",
        "router_eval": {**confusion(pairs), "errors": errors},
        "golden": {"scored_queries": len(rows), "label_errors": label_errors},
        "retrieval": aggregate(rows, seed=cfg.seed),
        "per_query": rows,
    }
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir / "a6_oracle.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out["router_eval"].items() if k != "errors"}, indent=2))
    print(
        "routing errors by (gold, routed):", dict(Counter((e["gold"], e["routed"]) for e in errors))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
