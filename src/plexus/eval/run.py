"""Score an experiment config on the verified golden set.

    python -m plexus.eval.run --config configs/a0_naive.yaml

Writes results/<config name>.json: per-metric bootstrap CIs overall and per query type,
retrieval latency, the config, corpus checksum and git commit, so every number in the
README traces back to one file and one command.
"""

from __future__ import annotations

import argparse
import json
import re
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
from plexus.eval.metrics import (
    GoldItem,
    Span,
    locate_quote_all,
    recall_at_k,
    retrieval_metrics,
    threshold_sensitivity,
)
from plexus.settings import REPO_ROOT, Settings


class SpanLike(Protocol):
    doc_id: str
    start: int
    end: int


class Retriever(Protocol):
    def retrieve(self, question: str, k: int | None = None) -> Sequence[SpanLike]: ...


def _company(doc_id: str) -> str:
    return doc_id.split("_", 1)[0]  # doc ids are TICKER_FORM_DATE


def gold_spans(
    q: GoldenQuery, doc_texts: dict[str, str], same_company: bool = True
) -> tuple[list[GoldItem], list[dict]]:
    """Locate each evidence quote -> one gold item per evidence. Unlocatable evidence is
    returned separately as label errors.

    Alternatives: every occurrence of the quote in its own filing and, when
    `same_company`, in the same company's other filings. 10-Qs repeat the 10-K's risk
    factors word for word (~75% of risk-factor spans in this corpus), so retrieving the
    identical sentence from the 10-Q is a correct retrieval, not a miss.
    """
    items: list[GoldItem] = []
    missing = []
    for ev in q.evidence:
        text = doc_texts.get(ev.doc_id)
        locs = locate_quote_all(text, ev.quote) if text is not None else []
        if not locs:
            reason = "unknown doc_id" if text is None else "quote not found"
            missing.append(
                {"qid": q.qid, "doc_id": ev.doc_id, "reason": reason, "quote": ev.quote[:120]}
            )
            continue
        alts = [Span(ev.doc_id, s, e) for s, e in locs]
        if same_company:
            for other, other_text in doc_texts.items():
                if other != ev.doc_id and _company(other) == _company(ev.doc_id):
                    alts += [Span(other, s, e) for s, e in locate_quote_all(other_text, ev.quote)]
        items.append(tuple(alts))
    return items, missing


def theme_of(q: GoldenQuery, label_to_keyword: dict[str, str]) -> str | None:
    """Theme keyword of a global question, from its source tag 'theme=<label>'."""
    m = re.search(r"theme=(.+)$", q.source)
    return label_to_keyword.get(m.group(1).strip()) if m else None


def _on_theme(keyword: str, text: str) -> bool:
    from plexus.eval.global_themes import theme_regex

    return theme_regex(keyword).search(text) is not None


def company_coverage(hits: Sequence, relevant: set[str], keyword: str, k: int) -> float:
    """Relevant companies with >=1 on-theme chunk in the top k, divided by the most that k
    chunks could cover, min(k, |relevant|). Chunker-independent and needs no quote labels (design review finding #2)."""
    covered = {
        _company(h.doc_id)
        for h in hits[:k]
        if _company(h.doc_id) in relevant and _on_theme(keyword, getattr(h, "text", "") or "")
    }
    # normalised by what top-k can reach: k chunks can cover at most k companies, and most
    # themes apply to 30+ of 39 companies, so plain coverage would be capped near k/|relevant|
    return len(covered) / min(k, len(relevant))


def score_retrieval(
    queries: Sequence[GoldenQuery],
    retriever: Retriever,
    doc_texts: dict[str, str],
    k: int,
    theme_labels: dict[str, set[str]] | None = None,
    label_to_keyword: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Queries with ANY unlocatable evidence are skipped (and reported), never scored on the
    remaining evidence, which would inflate recall."""
    ks = [x for x in RECALL_KS if x <= k]
    rows, label_errors = [], []
    for q in queries:
        gold, missing = gold_spans(q, doc_texts)
        strict, _ = gold_spans(q, doc_texts, same_company=False)
        if missing:
            label_errors.extend(missing)
            continue
        t0 = time.perf_counter()
        hits = retriever.retrieve(q.question, k)
        ms = (time.perf_counter() - t0) * 1000
        retrieved = [Span(h.doc_id, h.start, h.end) for h in hits]
        extra: dict = {}
        kw = theme_of(q, label_to_keyword or {}) if q.qtype == "global" else None
        if kw and theme_labels and theme_labels.get(kw):
            extra["theme"] = kw
            extra["n_relevant_companies"] = len(theme_labels[kw])
            for kk in (10, 20):
                if kk <= k:
                    extra[f"company_coverage@{kk}"] = company_coverage(
                        hits, theme_labels[kw], kw, kk
                    )
        rows.append(
            {
                "qid": q.qid,
                "qtype": q.qtype,
                **extra,
                "retrieval_ms": ms,
                **retrieval_metrics(retrieved, gold, ks),
                **threshold_sensitivity(retrieved, gold),
                # secondary: only the exact filing the question was written from counts
                "strict_recall@10": recall_at_k(retrieved, strict, min(10, k)),
            }
        )
    return rows, label_errors


RECALL_KS = (1, 5, 10, 20)
METRICS = [
    "recall@1", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr",
    "recall@10@cover0.3", "recall@10@cover0.7", "strict_recall@10",
    "company_coverage@10", "company_coverage@20",
]  # fmt: skip


def aggregate(rows: list[dict], seed: int = 0) -> dict:
    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r["qtype"], []).append(r)
    out = {}
    for name, rs in groups.items():
        out[name] = {}
        for m in METRICS:
            sub = [r for r in rs if m in r]
            if not sub:
                continue
            # global questions: resample whole themes (questions on one theme are not independent)
            clusters = [r["theme"] for r in sub] if all("theme" in r for r in sub) else None
            out[name][m] = bootstrap_ci([r[m] for r in sub], seed=seed, clusters=clusters).to_dict()
        lat = np.array([r["retrieval_ms"] for r in rs])
        out[name]["retrieval_ms"] = {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
        }
    return out


def score_generation(
    queries: Sequence[GoldenQuery],
    rows: list[dict],
    pipeline,
    doc_texts: dict[str, str],
    judge=None,
) -> None:
    """Add answer-level fields to each scored row (in place).

    citation_gold_precision is deterministic: the share of cited excerpts that hit gold
    evidence. It is a lower bound on citation quality (a cited excerpt can support a
    claim without being the labelled evidence). faithfulness/correctness need the judge.
    """
    from plexus.baseline.naive import is_refusal
    from plexus.eval.judge import VERDICT_SCORE
    from plexus.eval.metrics import is_hit

    by_qid = {q.qid: q for q in queries}
    for row in rows:
        q = by_qid[row["qid"]]
        gold, _ = gold_spans(q, doc_texts)
        ans = pipeline.answer(q.question)
        cited = [h for h in ans.hits if h.chunk_id in set(ans.cited_chunk_ids)]
        hits_gold = [any(is_hit(Span(h.doc_id, h.start, h.end), g) for g in gold) for h in cited]
        row.update(
            question=q.question,
            reference=q.answer,
            answer=ans.text,
            refused=is_refusal(ans.text),
            n_citations=len(cited),
            citation_gold_precision=(sum(hits_gold) / len(cited)) if cited else None,
            generation_ms=ans.generation_ms,
            generation_cached=ans.cached,
        )
        if judge is not None:
            row["faithfulness"] = judge.faithfulness(
                q.question, ans.text, [h.text for h in ans.hits]
            )
            verdict = judge.correctness(q.question, q.answer, ans.text)
            row["judge_verdict"] = verdict
            row["correctness"] = VERDICT_SCORE[verdict]


GEN_METRICS = ["citation_gold_precision", "faithfulness", "correctness"]


def aggregate_generation(rows: list[dict], seed: int = 0) -> dict:
    groups: dict[str, list[dict]] = {"all": rows}
    for r in rows:
        groups.setdefault(r["qtype"], []).append(r)
    out = {}
    for name, rs in groups.items():
        g: dict = {"refusal_rate": round(sum(r["refused"] for r in rs) / len(rs), 4)}
        for m in GEN_METRICS:
            vals = [r[m] for r in rs if r.get(m) is not None]
            g[m] = bootstrap_ci(vals, seed=seed).to_dict() if vals else None
        uncached = [r["generation_ms"] for r in rs if not r["generation_cached"]]
        g["generation_ms_uncached"] = (
            {
                "p50": round(float(np.percentile(uncached, 50)), 1),
                "p95": round(float(np.percentile(uncached, 95)), 1),
            }
            if uncached
            else None  # all served from cache: latency would be meaningless
        )
        out[name] = g
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


def _display_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def run(
    cfg: ExperimentConfig,
    golden_path: Path | None = None,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    generate: bool = False,
    use_judge: bool = False,
    llm=None,
) -> dict:
    from plexus.baseline.naive import NaiveRAG
    from plexus.eval.judge import Judge
    from plexus.index.build import index_dir
    from plexus.ingest.process import iter_docs
    from plexus.llm import LLMClient

    data = data_dir or Settings.load().data_dir
    golden_path = golden_path or data / "golden" / "golden.jsonl"
    queries = load_golden(golden_path)
    doc_texts = {m["doc_id"]: d.text for m, d in iter_docs(data / "processed", cfg.snapshot)}
    from plexus.retrieve.pipeline import make_retriever

    idx_path = index_dir(data, cfg)
    retriever = make_retriever(cfg, data)
    if (generate or use_judge) and llm is None:
        llm = LLMClient.from_settings()
    pipeline = NaiveRAG(cfg, retriever, llm)
    theme_labels, label_to_keyword = None, None
    themes_csv = data / "golden" / "global_themes.csv"
    if themes_csv.exists():
        from plexus.eval.global_themes import load_labels, load_themes

        theme_labels = load_labels(themes_csv)
        label_to_keyword = {label: kw for kw, label in load_themes().items()}
    rows, label_errors = score_retrieval(
        queries, pipeline, doc_texts, cfg.retrieval.top_k, theme_labels, label_to_keyword
    )
    generation = {}
    if generate or use_judge:
        judge = Judge(llm, cfg.judge.provider, cfg.judge.model, cfg.seed) if use_judge else None
        score_generation(queries, rows, pipeline, doc_texts, judge)
        generation = aggregate_generation(rows, seed=cfg.seed) if rows else {}
        generation["llm_usage"] = {
            "calls": llm.stats.calls,
            "cache_hits": llm.stats.cache_hits,
            "cache_hit_rate": round(llm.stats.cache_hit_rate, 4),
            "prompt_tokens_uncached": llm.stats.prompt_tokens,
            "completion_tokens_uncached": llm.stats.completion_tokens,
            "by_provider": llm.stats.by_provider,
        }
    if hasattr(retriever, "close"):
        retriever.close()

    build_info = json.loads((idx_path / "build_info.json").read_text())
    result = {
        "config": cfg.model_dump(),
        "git_commit": _git_commit(),
        "corpus_checksum": build_info.get("corpus_checksum"),
        "index": build_info,
        "golden": {
            "file": _display_path(golden_path),
            "verified_queries": len(queries),
            "scored_queries": len(rows),
            "label_errors": label_errors,
        },
        "retrieval": aggregate(rows, seed=cfg.seed) if rows else {},
        "generation": generation,
        "per_query": rows,
    }
    out_dir = results_dir or REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument(
        "--generate", action="store_true", help="also generate answers (LLM calls, cached)"
    )
    ap.add_argument("--judge", action="store_true", help="also judge answers (implies --generate)")
    args = ap.parse_args(argv)
    res = run(
        ExperimentConfig.load(args.config),
        args.golden,
        generate=args.generate,
        use_judge=args.judge,
    )
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
