"""Validate the verifier against blind human labels before trusting it (Phase 5, step 4).

    python -m plexus.eval.verifier_labels export   # -> data/golden/verifier_labels.csv
    python -m plexus.eval.verifier_labels score    # after the human fills `supported`

60 sentence/source pairs from the saved A6 answers (dev), 20 per question type:
- 50 real pairs: a cited answer sentence and the passage(s) it cites;
- 10 swapped pairs: the same kind of sentence shown with a passage from ANOTHER question, so the
  sheet surely contains unsupported examples (real answers are mostly supported).
Which rows are swapped is stored in a separate key file. The sheet shows no model scores.

Acceptance bar, fixed before scoring (2026-10-29): the verifier catches >= 70% of the pairs the
human marks unsupported AND passes >= 85% of the supported ones (same bar as the answer judge).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from plexus.settings import REPO_ROOT

GOLDEN = REPO_ROOT / "data" / "golden"
SHEET = GOLDEN / "verifier_labels.csv"
KEY = GOLDEN / "verifier_labels_key.json"
ACCEPT_CATCH, ACCEPT_PASS = 0.70, 0.85


def _cited_sentences(row: dict) -> list[tuple[str, list[int]]]:
    from plexus.agents.verifier import _CITE, split_sentences, strip_citations

    out = []
    for sent in split_sentences(row["answer"]):
        cited = sorted({int(n) for n in _CITE.findall(sent)})
        claim = strip_citations(sent)
        if cited and all(1 <= n <= len(row["contexts"]) for n in cited) and len(claim.split()) >= 6:
            out.append((claim, cited))
    return out


def export(results: Path, seed: int = 0, per_type: int = 20, swapped: int = 10) -> int:
    rows = [r for r in json.loads(results.read_text())["per_query"] if r.get("contexts")]
    rng = random.Random(seed)
    pairs = []
    for qtype in ("local", "multi_hop", "global"):
        pool = [(r, s) for r in rows if r["qtype"] == qtype for s in _cited_sentences(r)]
        pairs += [(r, s, False) for r, s in rng.sample(pool, per_type)]
    for i in rng.sample(range(len(pairs)), swapped):  # unsupported by construction
        r, (claim, cited), _ = pairs[i]
        other = rng.choice([o for o in rows if o["qid"] != r["qid"]])
        pairs[i] = ({**r, "contexts": [rng.choice(other["contexts"])]}, (claim, [1]), True)
    rng.shuffle(pairs)
    sheet, key = [], {"results": str(results.relative_to(REPO_ROOT)), "rows": {}}
    for i, (r, (claim, cited), swap) in enumerate(pairs, 1):
        pid = f"V-{i:02d}"
        source = "\n\n".join(f"[{n}] {r['contexts'][n - 1]}" for n in cited)
        sheet.append({"pid": pid, "question": r["question"], "sentence": claim,
                      "source_passages": source, "supported": ""})  # fmt: skip
        key["rows"][pid] = {"qid": r["qid"], "cited": cited, "swapped": swap}
    with open(SHEET, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    KEY.write_text(json.dumps(key, indent=2) + "\n")
    return len(sheet)


def score(model: str = "roberta") -> dict:
    from plexus.agents.verifier import THRESHOLD, MiniCheck, numbers
    from plexus.eval.checklist_judge import binary_stats
    from plexus.eval.ragas_eval import auroc

    with open(SHEET, newline="", encoding="utf-8-sig") as fh:
        sheet = list(csv.DictReader(fh))
    key = json.loads(KEY.read_text())["rows"]
    bad = [r["pid"] for r in sheet if r["supported"].strip().lower() not in ("yes", "no")]
    if bad:
        raise ValueError(f"`supported` must be yes/no; missing or invalid: {bad}")
    m = MiniCheck(REPO_ROOT / "data" / "models" / f"minicheck-{model}-large")
    per = []
    for r in sheet:
        doc = "\n\n".join(p.split("] ", 1)[1] if p.startswith("[") else p
                          for p in r["source_passages"].split("\n\n"))  # fmt: skip
        s = m.score(doc, r["sentence"])
        num_ok = not (numbers(r["sentence"]) - numbers(doc))
        per.append({"pid": r["pid"], "human_supported": r["supported"].strip().lower() == "yes",
                    "support": round(s, 4), "numbers_ok": num_ok,
                    "swapped": key[r["pid"]]["swapped"]})  # fmt: skip
    h = [p["human_supported"] for p in per]
    out = {"n": len(per), "human_unsupported": h.count(False), "threshold": THRESHOLD,
           "acceptance_bar": {"catch_unsupported": ACCEPT_CATCH, "pass_supported": ACCEPT_PASS},
           "auroc_minicheck": round(auroc([p["support"] for p in per], h) or 0, 3),
           "swapped_marked_unsupported_by_human": sum(not p["human_supported"] for p in per if p["swapped"])}  # fmt: skip
    for name, verdict in (
        ("minicheck", [p["support"] >= THRESHOLD for p in per]),
        ("minicheck_plus_numbers", [p["support"] >= THRESHOLD and p["numbers_ok"] for p in per]),
    ):
        st = binary_stats(h, verdict)  # "TPR" = share of human-unsupported pairs flagged
        st["accepted"] = bool(st["tpr_catches_not_correct"] is not None
                              and st["tpr_catches_not_correct"] >= ACCEPT_CATCH
                              and st["tnr_passes_correct"] >= ACCEPT_PASS)  # fmt: skip
        out[name] = st
    # the 10 swapped pairs are easy negatives; report real answer sentences alone too (review)
    real = [p for p in per if not p["swapped"]]
    out["real_pairs_only"] = binary_stats(
        [p["human_supported"] for p in real], [p["support"] >= THRESHOLD for p in real]
    )
    return {"summary": out, "per_pair": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("export", "score"))
    ap.add_argument("--results", type=Path, default=REPO_ROOT / "results/a6_router.json")
    ap.add_argument("--model", choices=("flan-t5", "roberta"), default="roberta")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(f"{SHEET}: {export(a.results.resolve())} pairs to label")
        return 0
    out = score(a.model)
    out["summary"]["model"] = a.model
    suffix = "_flan_t5" if a.model == "flan-t5" else ""
    (REPO_ROOT / "results" / f"verifier_validation{suffix}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
