"""Every number in README.md comes from here (rule 9: never fabricate numbers).

    python -m plexus.readme          # rewrite the tables in README.md from results/*.json
    python -m plexus.readme --check  # fail if README.md is out of date (runs in the tests)

Each table sits between `<!-- table:NAME -->` and `<!-- /table -->` markers; text outside
the markers is hand-written and must not contain measured numbers.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from plexus.settings import REPO_ROOT

RESULTS = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"

LADDER = [  # (results/test file, label, what it adds)
    ("a0_naive", "A0", "Naive RAG: fixed 300-word windows, dense search"),
    ("a1h_hybrid", "A1–A2", "Section-aware chunks with headers, hybrid BM25 + dense (RRF)"),
    ("a2_rr", "A2-RR", "No-graph baseline: alternate between the companies named"),
    ("a3h_rerank", "A3", "Cross-encoder reranker (bge-reranker-base)"),
    ("a4_graph", "A4", "Knowledge-graph hits fused in"),
    ("a5_diverse", "A5", "Company-diverse ranking for sector-wide questions"),
    ("a6_router", "A6", "Rule router picks the route per question (the app's base)"),
]


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _ci(m: dict) -> str:
    return f"{m['mean']:.2f} [{m['low']:.2f}, {m['high']:.2f}]"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _wilson(w: list[float]) -> str:
    return f"[{_pct(max(0.0, w[0]))}, {_pct(w[1])}]"


def ladder() -> str:
    rows = [
        "| Step | What it adds | All recall@10 [95% CI] | Local | Multi-hop | Multi-hop, both facts | Global: companies covered |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    n = None
    for name, label, what in LADDER:
        r = _load(f"test/{name}")["retrieval"]
        n = r["all"]["recall@10"]["n"]
        rows.append(
            f"| {label} | {what} | {_ci(r['all']['recall@10'])} "
            f"| {r['local']['recall@10']['mean']:.2f} | {r['multi_hop']['recall@10']['mean']:.2f} "
            f"| {r['multi_hop']['complete@10']['mean']:.2f} "
            f"| {r['global']['company_coverage@10']['mean']:.2f} |"
        )
    rows.append("")
    rows.append(
        f"Held-out test set, {n} questions, scored once; no design decision used these numbers. "
        "Intervals: bootstrap over questions."
    )
    return "\n".join(rows)


def answers() -> str:
    s = _load("checklist_judge_a6_router")["summary"]
    h = s["human_accuracy"]
    v = _load("verifier_validation_cpu")["summary"]["minicheck"]
    return "\n".join(
        [
            "| Check | Result |",
            "| --- | --- |",
            f"| Answers I judged correct (blind labels, A6 writer) | {h['correct']}/{h['n']} = {_pct(h['rate'])} {_wilson(h['wilson_95'])} |",
            f"| Fact-checker flags sentences I marked as not backed by their passage | {_pct(v['tpr_catches_not_correct'])} {_wilson(v['tpr_ci'])} of {v['human_not_correct']} |",
            f"| Fact-checker passes sentences I marked as backed | {_pct(v['tnr_passes_correct'])} {_wilson(v['tnr_ci'])} of {v['n'] - v['human_not_correct']} |",
            f"| Fact-checker vs me, agreement (kappa) | {v['binary_kappa']:.2f} |",
            f"| LLM judge accepted as a stand-in for me | {'yes' if s['accepted'] else 'no (below its pre-set bar)'} |",
        ]
    )


def guardrails() -> str:
    g = _load("guardrails_a9_agents")["summary"]
    t, h, u, i = g["refusals_tuning_set"], g["refusals_heldout"], g["unanswerable"], g["injection"]
    return "\n".join(
        [
            "| Guardrail | Result [95% CI] |",
            "| --- | --- |",
            f"| Investment-advice requests refused (held-out) | {h['refuse_advice']['ok']}/{h['refuse_advice']['n']} {_wilson(h['refuse_advice']['wilson_95'])} |",
            f"| Fair questions answered, not refused (held-out) | {h['answer']['ok']}/{h['answer']['n']} {_wilson(h['answer']['wilson_95'])} |",
            f"| Off-topic requests refused (tuning set) | {t['refuse_scope']['ok']}/{t['refuse_scope']['n']} {_wilson(t['refuse_scope']['wilson_95'])} |",
            f"| Questions the filings cannot answer: declined | {u['declined']}/{u['n']} {_wilson(u['wilson_95'])} |",
            f"| Hidden instructions planted in passages that changed the answer | {i['attack_success']}/{i['n']} {_wilson(i['wilson_95'])} |",
        ]
    )


def serving() -> str:
    load = _load("phase6_load")
    st = _load("phase6_stages")
    c = _load("a8_cache")["test"]
    rows = [
        "| Users at once | Answered | Told busy | First words p50 | Full answer p50 | Full answer, slowest |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for k, lv in load["levels"].items():
        rows.append(
            f"| {k} | {lv['answered']}/{lv['requests']} | {lv['busy_503']} | {lv['first_token_p50_s']:.1f} s "
            f"| {lv['total_p50_s']:.1f} s | {lv['total_p95_s']:.1f} s |"
        )
    s = st["step_ms"]
    rows += [
        "",
        "| Measure | Result |",
        "| --- | --- |",
        f"| Writer (local qwen3.5:4b) p50 | {s['answer']['p50'] / 1000:.1f} s |",
        f"| Fact-check (CPU) p50 | {s['verify']['p50'] / 1000:.1f} s |",
        f"| Search p50 | {s['retrieve']['p50'] / 1000:.2f} s |",
        f"| Answer from the cache p50 | {st['total_ms_cached']['p50']:.0f} ms |",
        f"| Reworded questions reused from the cache (held-out pairs) | {c['reused_rewordings']} |",
        f"| Wrong answers reused from the cache | {c['wrong_reuses']} |",
        f"| GPU memory, peak | {load['gpu_mib']['peak']:,} of {load['gpu_mib']['total']:,} MiB |",
        f"| Cost per question | ${load['cost_usd']:.2f} |",
        "",
        "One RTX 4050 laptop GPU (6 GB). Small samples: the slowest time is the maximum, not a true p95.",
    ]
    return "\n".join(rows)


def reframe() -> str:
    r = _load("reframe")
    a = r["typos_anywhere_fresh"]
    sp = r["spelling_fresh_set"]
    words = sum(v["total"] for v in a["words_by_length"].values())
    wrong = sum(v["wrong"] for v in a["words_by_length"].values())
    return "\n".join(
        [
            "| Check (fresh typo set, never used for tuning) | Without the layer | With it |",
            "| --- | --- | --- |",
            f"| Same search route as the correctly spelled question | {_pct(a['same_route_without'])} | {_pct(a['same_route_with'])} |",
            f"| Same top-5 passages | {a['top5_identical_without']} | {a['top5_identical_with']} |",
            f"| Typo words changed to a wrong word | | {wrong}/{words} |",
            f"| Correct questions changed (all {sp['golden_questions']}) | | {len(sp['golden_spelling_changes'])} |",
            f"| Company-name typos fixed | | {sp['company_restored']} |",
        ]
    )


def drift() -> str:
    path = RESULTS / "drift.json"
    if not path.exists():
        return "Not run yet. `make refresh` then `make drift` (writes results/drift.json)."
    d = json.loads(path.read_text())
    c = d["carry_over"]["all"]
    rows = [
        "| Snapshot | Filings | Companies |",
        "| --- | --- | --- |",
    ]
    for k in ("old", "new"):
        x = d["corpus"][k]
        rows.append(f"| {d['configs'][k]} | {x['filings']} | {x['companies']} |")
    rows += [
        "",
        f"Of {d['golden_questions']} golden questions, {c.get('carried', 0)} have every evidence quote "
        f"repeated word for word in the same company's newer filings; {c.get('partly', 0)} partly; "
        f"{c.get('gone', 0)} not at all; {c.get('no_filings', 0)} have no newer filing.",
    ]
    s = d["search_on_carried_questions"].get("recall@10")
    if s:
        rows += [
            "",
            "| Carried-over questions | Old index | New index | New minus old [95% CI] |",
            "| --- | --- | --- | --- |",
            f"| recall@10 ({s['questions']} questions) | {_ci(s['old'])} | {_ci(s['new'])} "
            f"| {s['new_minus_old']['diff']['mean']:+.2f} [{s['new_minus_old']['diff']['low']:+.2f}, "
            f"{s['new_minus_old']['diff']['high']:+.2f}] |",
        ]
    return "\n".join(rows)


TABLES: dict[str, Callable[[], str]] = {
    "ladder": ladder,
    "answers": answers,
    "guardrails": guardrails,
    "serving": serving,
    "reframe": reframe,
    "drift": drift,
}


def render(text: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in TABLES:
            raise KeyError(f"README marker for unknown table {name!r}")
        return f"<!-- table:{name} -->\n{TABLES[name]()}\n<!-- /table -->"

    return re.sub(r"<!-- table:(\w+) -->.*?<!-- /table -->", sub, text, flags=re.DOTALL)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    old = README.read_text()
    new = render(old)
    if "--check" in argv:
        if new != old:
            print(
                "README.md tables are out of date: run `python -m plexus.readme`", file=sys.stderr
            )
            return 1
        return 0
    Path(README).write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
