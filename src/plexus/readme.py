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


NAMED = ("local", "multi_hop")  # questions naming companies; global ones are scored by coverage


def named_recall(result: dict, metric: str = "recall@10") -> dict:
    """Mean [95% CI] of `metric` over the questions that name companies (local + multi-hop),
    from a results file's per-question rows; same cluster bootstrap and seed as the scorer.
    Global questions are left out: they are scored by company coverage, not by passages."""
    from plexus.eval.bootstrap import bootstrap_ci

    rows = [r for r in result["per_query"] if r["qtype"] in NAMED]
    ci = bootstrap_ci(
        [r[metric] for r in rows],
        seed=result["config"].get("seed", 0),
        clusters=[r["cluster"] for r in rows],
    )
    return ci.to_dict()


def verdicts(path: Path) -> list[str]:
    """Human verdicts in a judge-labels CSV (blank rows skipped)."""
    import csv

    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    key = "human_verdict" if rows and "human_verdict" in rows[0] else "verdict"
    return [r[key].strip().lower() for r in rows if r[key].strip()]


def ladder() -> str:
    rows = [
        "| Step | What it adds | Named-company questions, recall@10 [95% CI] | Local | Multi-hop | Multi-hop, both facts | Global: companies covered |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    n = n_named = None
    for name, label, what in LADDER:
        d = _load(f"test/{name}")
        r = d["retrieval"]
        n = r["all"]["recall@10"]["n"]
        named = named_recall(d)
        n_named = named["n"]
        rows.append(
            f"| {label} | {what} | {_ci(named)} "
            f"| {r['local']['recall@10']['mean']:.2f} | {r['multi_hop']['recall@10']['mean']:.2f} "
            f"| {r['multi_hop']['complete@10']['mean']:.2f} "
            f"| {r['global']['company_coverage@10']['mean']:.2f} |"
        )
    rows.append("")
    rows.append(
        f"Held-out test set, {n} questions ({n_named} name companies; the rest are sector-wide and "
        "scored by company coverage), scored once at the end. One fix was made after seeing "
        "it (the ticker 'TI' was not linked to Texas Instruments); these numbers are from before "
        "that fix. Intervals: bootstrap over groups of related questions (shared linking company "
        "or theme)."
    )
    return "\n".join(rows)


def answers() -> str:
    s = _load("checklist_judge_a6_router")["summary"]
    h = s["human_accuracy"]
    summary = _load("verifier_validation_cpu")["summary"]
    v, real = summary["minicheck"], summary["real_pairs_only"]
    swaps = summary["swapped_marked_unsupported_by_human"]
    first = verdicts(REPO_ROOT / "data" / "golden" / "judge_labels_first.csv")
    sj, cj, bar = s["single_verdict_judge"], s["checklist_judge"], s["acceptance_bar"]
    return "\n".join(
        [
            "| Check | Result |",
            "| --- | --- |",
            f"| Answers I judged fully correct by hand, on {h['n']} practice (dev) questions. I labelled them twice, both times without knowing which version of kHop wrote them; this is the second check (docs/failures.md) | {h['correct']}/{h['n']} = {_pct(h['rate'])} {_wilson(h['wilson_95'])} |",
            f"| The same answers on my first check ({h['correct'] - first.count('correct')} borderline answers I first called partly correct were fully correct on the second check; both results are shown) | {first.count('correct')}/{len(first)} = {_pct(first.count('correct') / len(first))} |",
            f"| Fact-checker flags real sentences I marked as not backed by their passage | {real['human_not_correct']}/{real['human_not_correct']} caught {_wilson(real['tpr_ci'])} |",
            f"| ...the same, adding {swaps} test sentences paired with a wrong passage on purpose | {_pct(v['tpr_catches_not_correct'])} {_wilson(v['tpr_ci'])} of {v['human_not_correct']} |",
            f"| Fact-checker passes sentences I marked as backed | {_pct(v['tnr_passes_correct'])} {_wilson(v['tnr_ci'])} of {v['n'] - v['human_not_correct']} |",
            f"| Fact-checker vs me, agreement (kappa): real sentences / with the swapped ones | {real['binary_kappa']:.2f} / {v['binary_kappa']:.2f} |",
            f"| LLM judge (one verdict per answer) vs me: Cohen's kappa / catches answers I marked not correct | {sj['binary_kappa']:.2f} / {round(sj['tpr_catches_not_correct'] * sj['human_not_correct'])} of {sj['human_not_correct']} |",
            f"| LLM judge (pre-registered checklist) vs me: Cohen's kappa / catches answers I marked not correct | {cj['binary_kappa']:.2f} / {round(cj['tpr_catches_not_correct'] * cj['human_not_correct'])} of {cj['human_not_correct']} |",
            f"| LLM judge accepted as a stand-in for me (bar: catch {_pct(bar['tpr'])}, pass {_pct(bar['tnr'])}) | {'yes' if s['accepted'] else 'no (below its pre-set bar)'} |",
            *ragas_rows(),
        ]
    )


def ragas_rows() -> list[str]:
    """RAGAS rows, if results/ragas_a6_router.json exists (`make ragas`; resumable daily runs)."""
    path = RESULTS / "ragas_a6_router.json"
    if not path.exists():
        return []
    r = json.loads(path.read_text())["summary"]
    total = r["n"]
    judge = r.get("judge") or ", ".join(r.get("models_used", {})) or "Gemini"
    status = (
        "complete"
        if r.get("complete")
        else f"in progress on the free tier: {r.get('answers_fully_scored', 0)} of {total} answers fully scored"
    )
    rows = [f"| RAGAS ({judge} judge; {status}) | answers scored per metric below |"]
    # The README shows three RAGAS metrics (owner's choice); the results file keeps all six.
    labels = {
        "faithfulness": "faithfulness: claims backed by the retrieved passages",
        "context_recall": "context recall: passages contain what the reference needs",
        "answer_relevancy": "answer relevancy: the answer addresses the question",
    }
    for key, label in labels.items():
        m = r.get(key)
        if m and m.get("n"):
            rows.append(f"| RAGAS {label} | {m['mean']:.2f} ({m['n']} of {total}) |")
    return rows


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
            f"| Questions the filings cannot answer: declined (eval set, not held out) | {u['declined']}/{u['n']} {_wilson(u['wilson_95'])} |",
            f"| Hidden instructions planted in passages that changed the answer (the set that decided to keep datamarking) | {i['attack_success']}/{i['n']} {_wilson(i['wilson_95'])} |",
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
        f"| Wrong answers reused from the cache (upper 95% bound) | {c['wrong_reuses']} ({_pct(c['wrong_reuse_upper_95'])}) |",
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
            "| Check | Set | Without the layer | With it |",
            "| --- | --- | --- | --- |",
            f"| Same search route as the correctly spelled question | fresh typos-anywhere set ({a['sentences']} sentences), not used for tuning | {_pct(a['same_route_without'])} | {_pct(a['same_route_with'])} |",
            f"| Same top-5 passages | same fresh set | {a['top5_identical_without']} | {a['top5_identical_with']} |",
            f"| Typo words changed to a wrong word | same fresh set | | {wrong}/{words} |",
            f"| Correct questions changed | all {sp['golden_questions']} golden questions, a keep bar used while building | | {len(sp['golden_spelling_changes'])} |",
            f"| Company-name typos fixed | one-typo set, a keep bar used while building | | {sp['company_restored']} |",
        ]
    )


def _snapshot(config_name: str) -> str:
    """Snapshot name of a config (configs/<name>.yaml), e.g. semis-2026."""
    import yaml

    path = REPO_ROOT / "configs" / f"{config_name}.yaml"
    return (
        yaml.safe_load(path.read_text()).get("snapshot", config_name)
        if path.exists()
        else config_name
    )


def drift() -> str:
    path = RESULTS / "drift.json"
    if not path.exists():
        return "Not run yet. `make refresh` then `make drift` (writes results/drift.json)."
    d = json.loads(path.read_text())
    c = d["carry_over"]["all"]
    rows = [
        "| Snapshot | Filings | By form | Companies |",
        "| --- | --- | --- | --- |",
    ]
    for k in ("old", "new"):
        x = d["corpus"][k]
        forms = ", ".join(f"{n} {f}" for f, n in sorted(x["by_form"].items()))
        rows.append(
            f"| {_snapshot(d['configs'][k])} | {x['filings']} | {forms} | {x['companies']} |"
        )
    rows += [
        "",
        f"Of {d['golden_questions']} golden questions, {c.get('carried', 0)} have every evidence quote "
        f"repeated word for word in the same company's newer filings; {c.get('partly', 0)} partly; "
        f"{c.get('gone', 0)} not at all; {c.get('no_filings', 0)} have no newer filing. "
        "Only carried-over questions can be scored on the new index, so the comparison below "
        "measures how search copes with a new corpus, not whether changed facts are found. "
        f"Search settings: `{d['configs']['old']}` (hybrid + headers + reranker; no router, "
        "company filter or graph, because a graph needs a long LLM extraction per snapshot). "
        "Sector-wide questions are left out of the comparison: they are scored by company "
        "coverage, not by quotes.",
    ]
    s = d["search_on_carried_questions"].get("recall@10")
    moved = d.get("per_question_search", [])
    up = sum(r["recall@10_new"] > r["recall@10_old"] for r in moved)
    down = sum(r["recall@10_new"] < r["recall@10_old"] for r in moved)
    if s:
        rows += [
            "",
            "| Carried-over questions | Old index | New index | New minus old [95% CI] |",
            "| --- | --- | --- | --- |",
            f"| recall@10 ({s['questions']} questions) | {_ci(s['old'])} | {_ci(s['new'])} "
            f"| {s['new_minus_old']['diff']['mean']:+.2f} [{s['new_minus_old']['diff']['low']:+.2f}, "
            f"{s['new_minus_old']['diff']['high']:+.2f}] |",
            "",
            f"Per question: {up} up, {down} down, the rest unchanged. Intervals resample whole "
            "groups of questions that share a linking company, as in the main scorer; when the "
            "ups and downs fall in the same groups and cancel, the interval has zero width.",
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
