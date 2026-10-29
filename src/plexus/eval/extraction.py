"""Extraction precision / recall against gold relation labels (brief §9, Phase 3 gate).

    python -m plexus.eval.extraction --pred data/graph/semis-2025/sample_v6.jsonl

Gold: data/golden/extraction/relation_labels.jsonl if it exists (human-confirmed), else the
AI pre-labels (ai_relation_labels.jsonl): which one was used is printed and stored.

Matching: both sides go through the same entity resolver, so "TSMC" and "Taiwan
Semiconductor Manufacturing Company Limited" are one entity. Three levels:
- strict:   same subject, relation and object;
- relaxed:  same subject and object, any relation or direction (did we find the link?);
- company:  strict, restricted to company-company relations (what A4 traverses).
Non-company objects (products, segments, risks) also match when their word sets overlap by
Jaccard >= 0.5 ("export controls" ~ "U.S. export controls"). CIs: bootstrap over passages.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from plexus.extract.resolve import Resolver, concept_key, load_resolver
from plexus.settings import Settings

GOLD_DIR = Path("data/golden/extraction")


# Coarse relation families: the schema has overlapping types (a regulation can fairly be
# `subject_to` or `exposed_to`; a foundry `manufactured_by` or `supplied_by`), so labeller and
# model often pick different-but-valid types. Family-level scoring is reported alongside strict.
FAMILIES = {
    "exposed_to": "exposure", "subject_to": "exposure", "operates_in": "exposure",
    "supplied_by": "supply", "manufactured_by": "supply",
    "licenses_from": "licensing", "licenses_to": "licensing",
    "acquired": "deal", "sold_business_to": "deal",
}  # fmt: skip


def family(rel: str) -> str:
    return FAMILIES.get(rel, rel)


def _ent(res: Resolver, name: str, etype: str) -> str:
    return res.resolve_exact(name, etype)[0]


def _words(name: str) -> set[str]:
    return set(concept_key(name).split())


def _same(a: tuple, b: tuple, relaxed: bool, fam: bool = False) -> bool:
    (sa, ra, oa, ta, na), (sb, rb, ob, tb, nb) = a, b
    if fam:
        ra, rb = family(ra), family(rb)
    if not relaxed and ra != rb:
        return False
    ends_a, ends_b = {sa, oa}, {sb, ob}
    if relaxed and ends_a == ends_b:
        return True
    if sa != sb and not relaxed:
        return False
    if oa == ob:
        return True if relaxed else sa == sb
    if ta == tb and ta not in ("company", "person"):
        wa, wb = _words(na), _words(nb)
        return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5
    return False


def _triples(res: Resolver, rels: list[dict]) -> list[tuple]:
    out = []
    for r in rels:
        s = _ent(res, r["subject"], r.get("subject_type", "company"))
        o = _ent(res, r["object"], r["object_type"])
        out.append((s, r["relation"], o, r["object_type"], r["object"]))
    return out


def _match(
    pred: list[tuple], gold: list[tuple], relaxed: bool, fam: bool = False
) -> tuple[int, int]:
    """Greedy one-to-one matching; returns (matched predictions, matched gold)."""
    used: set[int] = set()
    hits = 0
    for p in pred:
        for j, g in enumerate(gold):
            if j not in used and _same(p, g, relaxed, fam):
                used.add(j)
                hits += 1
                break
    return hits, len(used)


def score(pred_recs: dict[str, list[dict]], gold_recs: dict[str, list[dict]], res: Resolver,
          seed: int = 0) -> dict:  # fmt: skip
    rows = []
    for pid, gold in gold_recs.items():
        pred = pred_recs.get(pid, [])
        P, G = _triples(res, pred), _triples(res, gold)
        Pc = [t for t in P if t[3] == "company"]
        Gc = [t for t in G if t[3] == "company"]
        row = {
            "pid": pid,
            "n_pred": len(P),
            "n_gold": len(G),
            "n_pred_c": len(Pc),
            "n_gold_c": len(Gc),
        }
        row["tp_strict"], _ = _match(P, G, relaxed=False)
        row["tp_relaxed"], _ = _match(P, G, relaxed=True)
        row["tp_family"], _ = _match(P, G, relaxed=False, fam=True)
        row["tp_company_family"], _ = _match(Pc, Gc, relaxed=False, fam=True)
        row["tp_company"], _ = _match(Pc, Gc, relaxed=False)
        rows.append(row)

    def pr(tp: str, n_pred: str, n_gold: str) -> dict:
        tps = np.array([r[tp] for r in rows], float)
        npred = np.array([r[n_pred] for r in rows], float)
        ngold = np.array([r[n_gold] for r in rows], float)
        rng = np.random.default_rng(seed)
        boot_p, boot_r = [], []
        for _ in range(2000):
            i = rng.integers(0, len(rows), len(rows))
            boot_p.append(tps[i].sum() / max(npred[i].sum(), 1))
            boot_r.append(tps[i].sum() / max(ngold[i].sum(), 1))
        p = tps.sum() / max(npred.sum(), 1)
        r = tps.sum() / max(ngold.sum(), 1)
        return {"precision": round(p, 3), "precision_ci": [round(x, 3) for x in np.quantile(boot_p, [0.025, 0.975])],
                "recall": round(r, 3), "recall_ci": [round(x, 3) for x in np.quantile(boot_r, [0.025, 0.975])],
                "f1": round(2 * p * r / max(p + r, 1e-9), 3),
                "n_pred": int(npred.sum()), "n_gold": int(ngold.sum())}  # fmt: skip

    return {
        "passages": len(rows),
        "strict": pr("tp_strict", "n_pred", "n_gold"),
        "family": pr("tp_family", "n_pred", "n_gold"),
        "relaxed": pr("tp_relaxed", "n_pred", "n_gold"),
        "company_strict": pr("tp_company", "n_pred_c", "n_gold_c"),
        "company_family": pr("tp_company_family", "n_pred_c", "n_gold_c"),
    }


def load_gold(gold_dir: Path) -> tuple[dict[str, list[dict]], str]:
    human = gold_dir / "relation_labels.jsonl"
    path = human if human.exists() else gold_dir / "ai_relation_labels.jsonl"
    gold = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        gold[d["passage_id"]] = d["relations"]
    return gold, path.name


def load_pred(path: Path, ids: set[str]) -> dict[str, list[dict]]:
    out = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if d["passage_id"] in ids:
            out[d["passage_id"]] = d["relations"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = Settings.load()
    gold, gold_name = load_gold(a.gold_dir)
    pred = load_pred(a.pred, set(gold))
    missing = set(gold) - set(pred)
    res = load_resolver(s.data_dir, "semis-2025")
    result = {"pred": str(a.pred.name), "gold": gold_name, "missing_pred_passages": len(missing),
              **score(pred, gold, res)}  # fmt: skip
    print(json.dumps(result, indent=2))
    if a.out:
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    rels = Counter(r["relation"] for v in gold.values() for r in v)
    print("gold relations by type:", dict(rels.most_common()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
