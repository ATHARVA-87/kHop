"""Drift check (Phase 7): does Plexus still find the facts in newer filings, with no retuning?

    make refresh   # (you) download, parse and index the 2026 snapshot (configs/corpus_2026.yaml)
    make drift     # compare the golden set on semis-2025 vs semis-2026 -> results/drift.json

1. Carry-over: each golden question's evidence quotes are searched word for word (same
   normalisation as scoring, ADR 0004) in the SAME company's 2026 filings. A question carries
   over only if every quote is found; otherwise the fact was reworded, updated or removed.
2. Search on the carried-over questions: recall@10 with the same settings on the old index and
   the new one, paired bootstrap. No graph: a graph needs a 4.5 h LLM extraction per snapshot.
3. Corpus: filings and companies in each snapshot, and section-detection quality of the parser.

The golden set is never edited: carried-over questions point at the 2026 filing in memory only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from plexus.eval.golden import Evidence, GoldenQuery, load_golden
from plexus.eval.metrics import locate_quote_all
from plexus.settings import REPO_ROOT, Settings

OLD_CONFIG = REPO_ROOT / "configs" / "a3h_rerank.yaml"
NEW_CONFIG = REPO_ROOT / "configs" / "a3h_rerank_2026.yaml"


def _company(doc_id: str) -> str:
    return doc_id.split("_", 1)[0]  # doc ids are TICKER_FORM_DATE


def _form(doc_id: str) -> str:
    parts = doc_id.split("_")
    return parts[1] if len(parts) > 2 else ""


def carry_over(
    queries: Sequence[GoldenQuery], new_texts: dict[str, str]
) -> tuple[list[GoldenQuery], dict[str, str]]:
    """Each question -> its status and, if it carries over, a copy whose evidence points at the
    newer filing that repeats the quote (same form first, then any form of the same company).

    Status: "carried" (every quote found), "partly" (some found), "gone" (none found),
    "no_filings" (the company has no filing in the new snapshot), "no_evidence".
    """
    by_company: dict[str, list[str]] = {}
    for doc_id in sorted(new_texts):
        by_company.setdefault(_company(doc_id), []).append(doc_id)
    kept: list[GoldenQuery] = []
    status: dict[str, str] = {}
    for q in queries:
        if not q.evidence:
            status[q.qid] = "no_evidence"
            continue
        moved: list[Evidence] = []
        missing_company = False
        for ev in q.evidence:
            docs = by_company.get(_company(ev.doc_id), [])
            if not docs:
                missing_company = True
                continue
            docs = sorted(docs, key=lambda d: (_form(d) != _form(ev.doc_id), d))
            hit = next((d for d in docs if locate_quote_all(new_texts[d], ev.quote)), None)
            if hit:
                moved.append(Evidence(doc_id=hit, quote=ev.quote))
        if len(moved) == len(q.evidence):
            status[q.qid] = "carried"
            kept.append(q.model_copy(update={"evidence": moved}))
        elif moved:
            status[q.qid] = "partly"
        else:
            status[q.qid] = "no_filings" if missing_company else "gone"
    return kept, status


def status_table(queries: Sequence[GoldenQuery], status: dict[str, str]) -> dict:
    """Counts of each status, overall and per question type."""
    out: dict[str, dict[str, int]] = {"all": dict(Counter(status.values()))}
    for qtype in sorted({q.qtype for q in queries}):
        out[qtype] = dict(Counter(status[q.qid] for q in queries if q.qtype == qtype))
    return out


def corpus_summary(data_dir: Path, snapshot: str) -> dict:
    """Filings, companies and parser quality of one snapshot, from files the pipeline wrote."""
    manifest = data_dir / "snapshots" / f"{snapshot}.manifest.jsonl"
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    report_path = data_dir / "snapshots" / f"{snapshot}.parse_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {
        "filings": len(rows),
        "by_form": dict(Counter(r["form"] for r in rows)),
        "companies": len({r["ticker"] for r in rows}),
        "section_coverage": report.get("coverage", {}),  # share of filings with each section found
        "unstructured_docs": report.get("unstructured_docs", []),
    }


def recall_pairs(rows_old: list[dict], rows_new: list[dict], metric: str = "recall@10"):
    """Per-question metric on both snapshots, for the questions scored on both."""
    old = {r["qid"]: r for r in rows_old}
    new = {r["qid"]: r for r in rows_new}
    qids = sorted(set(old) & set(new))
    return qids, [old[q][metric] for q in qids], [new[q][metric] for q in qids]


def main(argv: list[str] | None = None) -> int:
    from plexus.config import ExperimentConfig
    from plexus.eval.bootstrap import bootstrap_ci, paired_bootstrap
    from plexus.eval.run import _git_commit, score_retrieval
    from plexus.ingest.process import iter_docs
    from plexus.retrieve.pipeline import make_retriever

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", type=Path, default=OLD_CONFIG)
    ap.add_argument("--new", type=Path, default=NEW_CONFIG)
    args = ap.parse_args(argv)
    data = Settings.load().data_dir
    old_cfg, new_cfg = ExperimentConfig.load(args.old), ExperimentConfig.load(args.new)
    queries = load_golden(data / "golden" / "golden.jsonl") + load_golden(
        data / "golden" / "golden_test.jsonl"
    )
    texts = {
        name: {m["doc_id"]: d.text for m, d in iter_docs(data / "processed", cfg.snapshot)}
        for name, cfg in (("old", old_cfg), ("new", new_cfg))
    }
    kept_new, status = carry_over(queries, texts["new"])
    carried = {q.qid for q in kept_new}
    kept_old = [q for q in queries if q.qid in carried]

    rows = {}
    for name, cfg, qs in (("old", old_cfg, kept_old), ("new", new_cfg, kept_new)):
        retriever = make_retriever(cfg, data)
        try:
            rows[name], errors = score_retrieval(qs, retriever, texts[name], cfg.retrieval.top_k)
        finally:
            if hasattr(retriever, "close"):
                retriever.close()
        if errors:
            print(f"{name}: {len(errors)} evidence quotes not located", file=sys.stderr)

    search: dict = {}
    for metric in ("recall@10", "ndcg@10"):
        qids, a, b = recall_pairs(rows["old"], rows["new"], metric)
        if not qids:
            continue
        search[metric] = {
            "questions": len(qids),
            "old": bootstrap_ci(a, seed=old_cfg.seed).to_dict(),
            "new": bootstrap_ci(b, seed=old_cfg.seed).to_dict(),
            "new_minus_old": paired_bootstrap(a, b, seed=old_cfg.seed).to_dict(),
        }
    by_type = {}
    for qtype in sorted({q.qtype for q in kept_new}):
        ids = {q.qid for q in kept_new if q.qtype == qtype}
        qids, a, b = recall_pairs(
            [r for r in rows["old"] if r["qid"] in ids], [r for r in rows["new"] if r["qid"] in ids]
        )
        if qids:
            by_type[qtype] = {
                "questions": len(qids),
                "old_recall@10": round(sum(a) / len(a), 4),
                "new_recall@10": round(sum(b) / len(b), 4),
            }
    out = {
        "what": "golden set on semis-2025 vs semis-2026, same settings, no retuning",
        "git_commit": _git_commit(),
        "configs": {"old": old_cfg.name, "new": new_cfg.name},
        "corpus": {
            "old": corpus_summary(data, old_cfg.snapshot),
            "new": corpus_summary(data, new_cfg.snapshot),
        },
        "golden_questions": len(queries),
        "carry_over": status_table(queries, status),
        "search_on_carried_questions": search,
        "search_by_type": by_type,
        "per_question_status": status,
    }
    path = REPO_ROOT / "results" / "drift.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_question_status"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
