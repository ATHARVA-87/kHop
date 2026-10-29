"""Claim support of saved answers, scored by the validated verifier (no labels needed).

    python -m plexus.eval.support results/a6_router.json results/a9_datamark.json

Per answer: every sentence of 4+ words is checked (verifier.check_answer). Reported per config:
- sentence_support: share of checked sentences that are supported (cited + MiniCheck >= 0.5);
- fully_supported_answers: share of non-refused answers whose sentences are all supported;
- uncited_sentences: share of sentences with no citation;
- refused: answers that decline; caret_leak: answers containing the datamark "^".
The verifier was validated on 60 blind human pairs (kappa 0.88, results/verifier_validation.json).
Paired bootstrap CI on the difference of fully-supported answers vs the FIRST file.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from plexus.settings import REPO_ROOT


def score_file(path: Path, scorer) -> dict:
    from plexus.agents.verifier import check_answer
    from plexus.eval.guardrails import declined

    per = {}
    for r in json.loads(path.read_text())["per_query"]:
        if not r.get("answer") or not r.get("contexts"):
            continue
        refused = declined(r["answer"])
        checks = [] if refused else check_answer(r["answer"], r["contexts"], scorer)
        per[r["qid"]] = {
            "refused": refused,
            "n": len(checks),
            "supported": sum(c.supported for c in checks),
            "uncited": sum("no_citation" in c.issues for c in checks),
            "caret": "^" in r["answer"],
        }
    return per


def summarize(per: dict) -> dict:
    ans = [p for p in per.values() if not p["refused"]]
    n = sum(p["n"] for p in ans)
    full = [p["n"] > 0 and p["supported"] == p["n"] for p in ans]
    return {
        "answers": len(per),
        "refused": sum(p["refused"] for p in per.values()),
        "sentence_support": round(sum(p["supported"] for p in ans) / n, 3) if n else None,
        "fully_supported_answers": round(sum(full) / len(full), 3) if full else None,
        "uncited_sentences": round(sum(p["uncited"] for p in ans) / n, 3) if n else None,
        "caret_leak": sum(p["caret"] for p in per.values()),
    }


def paired_diff(base: dict, other: dict, seed: int = 0, n_boot: int = 2000) -> dict:
    qids = sorted(
        q for q in base if q in other and not base[q]["refused"] and not other[q]["refused"]
    )

    def full(p):
        return float(p["n"] > 0 and p["supported"] == p["n"])

    diffs = [full(other[q]) - full(base[q]) for q in qids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return {"n": len(qids), "diff_fully_supported": round(sum(diffs) / len(diffs), 3),
            "ci95": [round(boots[int(0.025 * n_boot)], 3), round(boots[int(0.975 * n_boot)], 3)]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    from plexus.agents.verifier import MiniCheck

    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    scorer = MiniCheck()
    pers = {p.stem: score_file(p, scorer) for p in paths}
    out = {name: summarize(per) for name, per in pers.items()}
    base = paths[0].stem
    for name, per in pers.items():
        if name != base:
            out[name]["vs_" + base] = paired_diff(pers[base], per)
    dest = REPO_ROOT / "results" / ("support_" + "_vs_".join(p.stem for p in paths) + ".json")
    dest.write_text(json.dumps({"summary": out, "per_query": pers}, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
