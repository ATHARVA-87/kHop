"""Entity resolution: many surface names -> one graph node (Phase 3, step 2).

    python -m plexus.extract.resolve [--extractions PATH]

"Taiwan Semiconductor Manufacturing Company Limited", "TSMC" and "Taiwan Semiconductor" must
be one node, or no bridge question ("which foundry do X and Y both use?") can connect.

Layers, cheapest and safest first; every merge is logged with its method for spot-checks:
1. alias table: configs/entity_aliases.yaml plus every filer's legal name, cleaned name,
   ticker and common name (from the snapshot manifest);
2. normalised key: case, punctuation, articles and trailing legal suffixes removed
   ("Microsoft Corporation" == "Microsoft");
3. fuzzy: near-identical keys of the same type that share their first four letters
   (spelling and punctuation variants), never across types.
Anonymised names ("NVDA Customer A") are never merged with anything.

Outputs in data/graph/<snapshot>/: entities.jsonl, edges.jsonl, resolution_log.json.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from plexus.extract.schema import is_anonymous, strip_article
from plexus.ingest.headers import COMMON_NAMES, clean_company
from plexus.retrieve.bm25 import s_stem
from plexus.settings import REPO_ROOT, Settings

_LEGAL_TAIL = re.compile(
    r"[\s,]+(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|n\.?\s?v|ag|s\.?a|"
    r"llc|gmbh|holdings?|group|technolog(?:y|ies))\.?$",
    re.I,
)
FUZZY_RATIO = 0.92


def norm_key(name: str) -> str:
    """'The Microsoft Corporation' -> 'microsoft'; 'Hon Hai Precision Industry Co., Ltd.' ->
    'hon hai precision industry'. Only TRAILING legal words are removed, so
    'ON Semiconductor' keeps 'semiconductor'."""
    n = strip_article(" ".join(name.split()))
    prev = None
    while prev != n:
        prev = n
        n = _LEGAL_TAIL.sub("", n).strip(" ,.")
    return " ".join(re.findall(r"[a-z0-9]+", n.lower()))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def concept_key(name: str) -> str:
    """Risks, segments, products: lowercase with plurals folded ('export controls' ==
    'export control')."""
    return " ".join(s_stem(t) for t in norm_key(name).split())


class Resolver:
    def __init__(self, filers: dict[str, dict], aliases: dict[str, dict[str, list[str]]]):
        # key -> (entity id, canonical name, method)
        self.index: dict[tuple[str, str], tuple[str, str]] = {}
        self.filer_ids: dict[str, str] = {}
        for ticker, m in filers.items():
            eid, canon = f"company:{ticker}", clean_company(m["company"])
            self.filer_ids[ticker] = eid
            names = {m["company"], canon, ticker, COMMON_NAMES.get(ticker, "")}
            for n in names:
                if n:
                    self.index[("company", norm_key(n))] = (eid, canon)
        for etype, table in aliases.items():
            for canon, alts in table.items():
                key = (
                    ("company", norm_key(canon)) if etype == "company" else (etype, norm_key(canon))
                )
                eid, canon_name = self.index.get(key, (f"{etype}:{_slug(canon)}", canon))
                for n in [canon, *alts]:
                    self.index.setdefault((etype, norm_key(n)), (eid, canon_name))

    def resolve_exact(self, name: str, etype: str) -> tuple[str, str, str]:
        """(entity id, canonical name, method) without fuzzy matching."""
        if etype == "company" and is_anonymous(name.split(" ", 1)[-1]):
            return f"company:anon:{_slug(name)}", name, "anonymous"
        key = norm_key(name)
        if (etype, key) in self.index:
            eid, canon = self.index[(etype, key)]
            return eid, canon, "alias"
        if etype in ("company", "person", "geography", "regulation"):
            return f"{etype}:{_slug(key)}", name, "normalised"
        ck = concept_key(name)
        return f"{etype}:{_slug(ck)}", name, "normalised"


def fuzzy_merges(ids_by_type: dict[str, dict[str, str]]) -> dict[str, tuple[str, float]]:
    """Map near-duplicate ids onto one representative (the most frequent spelling wins later).
    Conservative: same type, same first 4 letters (blocking), SequenceMatcher ratio >=
    FUZZY_RATIO, so typos merge ("wistronn") but "intel" / "intelsat" (0.77) do not."""
    remap: dict[str, tuple[str, float]] = {}
    for _etype, ids in ids_by_type.items():
        by_first: dict[str, list[str]] = defaultdict(list)
        for eid, key in ids.items():
            if key:
                by_first[key[:4]].append(eid)
        for group in by_first.values():
            group.sort()
            for i, a in enumerate(group):
                if a in remap:
                    continue
                for b in group[i + 1 :]:
                    if b in remap:
                        continue
                    r = difflib.SequenceMatcher(None, ids[a], ids[b]).ratio()
                    if r >= FUZZY_RATIO:
                        remap[b] = (a, round(r, 3))
    return remap


def build(extraction_files: list[Path], resolver: Resolver, out_dir: Path) -> dict:
    mentions: Counter = Counter()
    names: dict[str, Counter] = defaultdict(Counter)
    etypes: dict[str, str] = {}
    keys: dict[str, str] = {}
    methods: Counter = Counter()
    raw_edges = []
    for path in extraction_files:
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            for r in rec["relations"]:
                ends = []
                for name, t in ((r["subject"], r["subject_type"]), (r["object"], r["object_type"])):
                    eid, canon, how = resolver.resolve_exact(name, t)
                    methods[how] += 1
                    mentions[eid] += 1
                    names[eid][canon] += 1
                    etypes[eid] = t
                    keys.setdefault(eid, norm_key(canon) if how != "alias" else "")
                    ends.append(eid)
                raw_edges.append((ends[0], r["relation"], ends[1], r, rec))
    # fuzzy pass over ids that did not come from the alias table (those are already canonical)
    by_type: dict[str, dict[str, str]] = defaultdict(dict)
    for eid, k in keys.items():
        if k and not eid.startswith("company:anon:"):
            by_type[etypes[eid]][eid] = k
    remap = fuzzy_merges(by_type)

    def final(eid: str) -> str:
        while eid in remap:
            eid = remap[eid][0]
        return eid

    edges: dict[tuple[str, str, str], dict] = {}
    for s, rel, o, r, rec in raw_edges:
        s, o = final(s), final(o)
        if s == o:
            continue
        e = edges.setdefault((s, rel, o), {"src": s, "relation": rel, "dst": o,
                                           "qualifiers": [], "evidence": []})  # fmt: skip
        if r.get("qualifier") and r["qualifier"] not in e["qualifiers"]:
            e["qualifiers"].append(r["qualifier"])
        e["evidence"].append({"doc_id": rec["doc_id"], "passage_id": rec["passage_id"],
                              "start": r["start"], "end": r["end"], "quote": r["quote"]})  # fmt: skip
    ents: dict[str, dict] = {}
    for eid in mentions:
        f = final(eid)
        ent = ents.setdefault(f, {"id": f, "type": etypes[f], "names": Counter(), "mentions": 0})
        ent["names"].update(names[eid])
        ent["mentions"] += mentions[eid]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "entities.jsonl", "w") as fh:
        for e in sorted(ents.values(), key=lambda x: -x["mentions"]):
            common = e["names"].most_common()
            fh.write(json.dumps({"id": e["id"], "type": e["type"], "name": common[0][0],
                                 "aliases": [n for n, _ in common[1:]], "mentions": e["mentions"],
                                 "ticker": e["id"].split(":")[1] if e["id"] in resolver.filer_ids.values() else None})
                     + "\n")  # fmt: skip
    with open(out_dir / "edges.jsonl", "w") as fh:
        for e in edges.values():
            e["support"] = len(e["evidence"])
            fh.write(json.dumps(e) + "\n")
    log = {
        "entities": len(ents),
        "entities_by_type": dict(Counter(e["type"] for e in ents.values())),
        "edges": len(edges),
        "edges_by_relation": dict(Counter(k[1] for k in edges)),
        "mention_resolution": dict(methods),
        "fuzzy_merges": [
            {"merged": b, "into": a, "ratio": r} for b, (a, r) in sorted(remap.items())
        ],
    }
    (out_dir / "resolution_log.json").write_text(json.dumps(log, indent=2) + "\n")
    return {k: v for k, v in log.items() if k != "fuzzy_merges"} | {"fuzzy_merges": len(remap)}


def load_resolver(data_dir: Path, snapshot: str) -> Resolver:
    filers = {}
    for line in (data_dir / "snapshots" / f"{snapshot}.manifest.jsonl").read_text().splitlines():
        m = json.loads(line)
        filers.setdefault(m["ticker"], m)
    aliases = yaml.safe_load((REPO_ROOT / "configs" / "entity_aliases.yaml").read_text())
    return Resolver(filers, aliases)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="semis-2025")
    ap.add_argument("--extractions", type=Path, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    s = Settings.load()
    gdir = s.data_dir / "graph" / a.snapshot
    files = a.extractions or [gdir / "extractions.jsonl"]
    print(json.dumps(build(files, load_resolver(s.data_dir, a.snapshot), a.out or gdir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
