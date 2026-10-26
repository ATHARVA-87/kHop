"""Score an experiment config on the verified golden set.

    python -m plexus.eval.run --config configs/a0_naive.yaml

Writes results/<config name>.json: per-metric bootstrap CIs overall and per query type,
retrieval latency, the config, corpus checksum and git commit, so every number in the
README traces back to one file and one command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from plexus.config import ExperimentConfig
from plexus.eval.bootstrap import bootstrap_ci
from plexus.eval.golden import GoldenQuery, load_golden
from plexus.eval.metrics import Span, locate_quote, retrieval_metrics
from plexus.settings import REPO_ROOT, Settings


class SpanLike(Protocol):
    doc_id: str
    start: int
    end: int


class Retriever(Protocol):
    def retrieve(self, question: str, k: int | None = None) -> Sequence[SpanLike]: ...


def gold_spans(q: GoldenQuery, doc_texts: dict[str, str]) -> tuple[list[Span], list[dict]]:
    """Locate each evidence quote. Unlocatable evidence is returned separately (label errors)."""
    spans, missing = [], []
    for ev in q.evidence:
        text = doc_texts.get(ev.doc_id)
        loc = locate_quote(text, ev.quote) if text is not None else None
        if loc is None:
            reason = "unknown doc_id" if text is None else "quote not found"
            missing.append(
                {"qid": q.qid, "doc_id": ev.doc_id, "reason": reason, "quote": ev.quote[:120]}
            )
        else:
            spans.append(Span(ev.doc_id, *loc))
    return spans, missing


def score_retrieval(
    queries: Sequence[GoldenQuery], retriever: Retriever, doc_texts: dict[str, str], k: int
) -> tuple[list[dict], list[dict]]:
    rows, label_errors = [], []
    for q in queries:
        gold, missing = gold_spans(q, doc_texts)
        label_errors.extend(missing)
        if not gold:  # cannot score retrieval without any locatable evidence
            continue
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.question, k)
        ms = (time.perf_counter() - t0) * 1000
        retrieved = [Span(h.doc_id, h.start, h.end) for h in hits]
        rows.append(
            {
                "qid": q.qid,
                "qtype": q.qtype,
                "retrieval_ms": ms,
                **retrieval_metrics(retrieved, gold),
            }
        )
    return rows, label_errors


METRICS = ["recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr"]


def aggregate(rows: list[dict], seed: int = 0) -> dict:
    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r["qtype"], []).append(r)
    out = {}
    for name, rs in groups.items():
        out[name] = {m: bootstrap_ci([r[m] for r in rs], seed=seed).to_dict() for m in METRICS}
        lat = np.array([r["retrieval_ms"] for r in rs])
        out[name]["retrieval_ms"] = {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        }
    return out


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True, text=True
        )
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def run(cfg: ExperimentConfig, golden_path: Path | None = None) -> dict:
    from plexus.baseline.naive import NaiveRAG
    from plexus.index.build import index_dir
    from plexus.index.dense import DenseIndex, Embedder
    from plexus.ingest.process import iter_docs

    settings = Settings.load()
    data = settings.data_dir
    queries = load_golden(golden_path)
    doc_texts = {m["doc_id"]: d.text for m, d in iter_docs(data / "processed", cfg.snapshot)}
    embedder = Embedder(cfg.embedder, cache_path=data / "cache" / "embeddings.sqlite")
    idx_path = index_dir(data, cfg)
    index = DenseIndex(idx_path / "qdrant", "chunks", embedder.dim)
    pipeline = NaiveRAG(cfg, embedder, index)
    rows, label_errors = score_retrieval(queries, pipeline, doc_texts, cfg.retrieval.top_k)
    index.close()

    build_info = json.loads((idx_path / "build_info.json").read_text())
    result = {
        "config": cfg.model_dump(),
        "git_commit": _git_commit(),
        "corpus_checksum": build_info.get("corpus_checksum"),
        "index": build_info,
        "golden": {
            "file": str((golden_path or data / "golden" / "golden.jsonl").relative_to(REPO_ROOT)),
            "verified_queries": len(queries),
            "scored_queries": len(rows),
            "label_errors": label_errors,
        },
        "retrieval": aggregate(rows, seed=cfg.seed) if rows else {},
        "per_query": rows,
    }
    out = REPO_ROOT / "results" / f"{cfg.name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--golden", type=Path, default=None)
    args = ap.parse_args(argv)
    res = run(ExperimentConfig.load(args.config), args.golden)
    print(
        f"{res['config']['name']}: {res['golden']['scored_queries']} queries scored, "
        f"{len(res['golden']['label_errors'])} label errors"
    )
    for group, metrics in res["retrieval"].items():
        line = ", ".join(f"{m}={metrics[m]['mean']:.3f}" for m in ("recall@10", "ndcg@10", "mrr"))
        print(f"  {group:<10} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
