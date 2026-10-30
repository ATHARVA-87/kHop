"""RAGAS answer metrics, run in the SEPARATE `ragas` conda env (see constraints-ragas.txt).

    make ragas RESULTS=results/a6_router.json        # uses the ragas env's python

Why a separate env: ragas 0.4.3 pulls ~35 packages (LangChain, LangGraph, datasets) and would
upgrade huggingface_hub past what transformers in the main env allows (research 2026-10-28).
Only two metrics, both claim-level (RAGAS paper, arXiv 2309.15217):
- Faithfulness: share of the answer's claims supported by the retrieved contexts;
- FactualCorrectness (F1): claims in the answer vs claims in the reference answer.
Every LLM call goes through plexus.llm.LLMClient (project rule 4: cached, offline-capable):
`PlexusRagasLLM` adapts it to RAGAS's InstructorBaseRagasLLM interface. RAGAS usage tracking
is switched off before ragas is imported. Scores are compared with the 49 blind human labels
(data/golden/judge_labels.csv) using thresholds fixed BEFORE looking: >= 0.75 correct,
>= 0.25 partially correct, else incorrect.
"""

from __future__ import annotations

import os

# before any ragas / langchain import: no usage tracking, no LangSmith tracing
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse
import asyncio
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from plexus.llm import (
    CacheMissError,
    LLMClient,
    LLMRequest,
    Message,
    ProviderError,
    StructuredOutputError,
)
from plexus.settings import REPO_ROOT, Settings

THRESHOLDS = (0.75, 0.25)  # fixed in advance: correct / partially correct / incorrect


def verdict_of(score: float) -> str:
    hi, lo = THRESHOLDS
    return "correct" if score >= hi else "partially_correct" if score >= lo else "incorrect"


def make_llm(client: LLMClient, provider: str, models: list[str]):
    from ragas.llms.base import InstructorBaseRagasLLM

    class PlexusRagasLLM(InstructorBaseRagasLLM):
        """RAGAS -> our cached client. Main model first, then fallbacks, with a circuit
        breaker (an overloaded model is skipped after 3 failures in a row)."""

        def __init__(self):
            self.failures: Counter = Counter()
            self.used: Counter = Counter()

        def generate(self, prompt: str, response_model):
            last = None
            live = [m for m in models if self.failures[m] < 3] or models[-1:]
            for m in live:
                req = LLMRequest(provider=provider, model=m,
                                 messages=(Message(role="user", content=prompt),),
                                 temperature=0.0, max_tokens=4096, seed=0)  # fmt: skip
                try:
                    out = client.complete_json(req, response_model)
                    self.failures[m] = 0
                    self.used[m] += 1
                    return out
                # a cache miss in offline mode falls through to the next model, like a provider
                # error: the cached run used the fallback model (3.6-flash was overloaded)
                except (ProviderError, StructuredOutputError, CacheMissError) as exc:
                    self.failures[m] += 1
                    last = exc
            raise last  # type: ignore[misc]

        async def agenerate(self, prompt: str, response_model):
            return await asyncio.to_thread(self.generate, prompt, response_model)

    return PlexusRagasLLM()


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_exp = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if p_exp == 1.0 else (p_obs - p_exp) / (1 - p_exp)


def auroc(scores: list[float], positive: list[bool]) -> float | None:
    """How well the score separates human-'correct' answers from the rest (0.5 = chance)."""
    pos = [s for s, p in zip(scores, positive, strict=True) if p]
    neg = [s for s, p in zip(scores, positive, strict=True) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", type=Path, required=True, help="eval results with answers + contexts"
    )
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/golden/judge_labels.csv")
    ap.add_argument(
        "--all", action="store_true", help="score every answer, not only the labelled ones"
    )
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--models", default="gemini-3.6-flash,gemini-3.5-flash-lite")
    a = ap.parse_args(argv)

    from ragas.metrics.collections import FactualCorrectness, Faithfulness

    res = json.loads(a.results.read_text())
    rows = [r for r in res["per_query"] if r.get("answer") and r.get("contexts")]
    with open(a.labels, newline="", encoding="utf-8-sig") as fh:
        human = {r["qid"]: r["human_verdict"].strip().lower()
                 for r in csv.DictReader(fh) if r["human_verdict"].strip()}  # fmt: skip
    if not a.all:
        rows = [r for r in rows if r["qid"] in human]
    client = LLMClient.from_settings(Settings.load())
    llm = make_llm(client, a.provider, a.models.split(","))
    # F1 (RAGAS default, planned first) also penalises EXTRA correct claims; the human rubric
    # says extra correct detail is fine, which matches recall mode. Recall was added after
    # seeing 14 rows, so both are reported, clearly labelled (2026-10-28).
    faith = Faithfulness(llm=llm)
    fact = FactualCorrectness(llm=llm)
    fact_recall = FactualCorrectness(llm=llm, mode="recall")

    per = []
    for i, r in enumerate(rows, 1):
        out = {"qid": r["qid"], "qtype": r["qtype"]}
        for name, fn in (
            ("faithfulness", lambda r=r: faith.score(user_input=r["question"], response=r["answer"],
                                                     retrieved_contexts=r["contexts"])),
            ("factual_correctness", lambda r=r: fact.score(response=r["answer"], reference=r["reference"])),
            ("factual_recall", lambda r=r: fact_recall.score(response=r["answer"], reference=r["reference"])),
        ):  # fmt: skip
            try:
                v = fn().value
                out[name] = (
                    None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
                )
            except Exception as exc:
                out[name], out[f"{name}_error"] = None, f"{type(exc).__name__}: {str(exc)[:160]}"
        per.append(out)
        if i % 10 == 0:
            print(f"{i}/{len(rows)}", file=sys.stderr)

    summary: dict = {"n": len(per), "thresholds": THRESHOLDS, "models_used": dict(llm.used)}
    for name in ("faithfulness", "factual_correctness", "factual_recall"):
        vals = [p[name] for p in per if p[name] is not None]
        summary[name] = {"mean": round(sum(vals) / len(vals), 4) if vals else None,
                         "n": len(vals), "failed": len(per) - len(vals)}  # fmt: skip
    for name in ("factual_correctness", "factual_recall"):
        summary.setdefault(name, None)
        pairs = [(human[p["qid"]], p[name]) for p in per
                 if p["qid"] in human and p.get(name) is not None]  # fmt: skip
        if not pairs:
            continue
        h = [x for x, _ in pairs]
        s_ = [y for _, y in pairs]
        rv = [verdict_of(y) for y in s_]
        au = auroc(s_, [x == "correct" for x in h])
        summary[f"vs_human_{name}"] = {
            "n": len(pairs),
            "kappa_at_fixed_thresholds": round(cohen_kappa(h, rv), 4),
            "raw_agreement": round(sum(x == y for x, y in zip(h, rv, strict=True)) / len(pairs), 4),
            "auroc_correct_vs_not": None if au is None else round(au, 4),
            "confusion": dict(Counter(f"human={x} ragas={y}" for x, y in zip(h, rv, strict=True))),
        }
    out_path = REPO_ROOT / "results" / f"ragas_{a.results.stem}.json"
    out_path.write_text(json.dumps({"summary": summary, "per_query": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
