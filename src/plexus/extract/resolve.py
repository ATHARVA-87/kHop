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


# Words that appear in company names but say nothing about WHICH company, plus places.
GENERIC_NAME_WORDS = {
    "semiconductor", "semiconductors", "solutions", "technology", "technologies",
    "intertechnology", "systems", "devices", "instruments", "materials", "power", "micro",
    "integrations", "international", "innovation", "logic", "laboratories", "labs", "research",
    "electronics", "microsystems", "holding", "holdings", "group", "manufacturing", "precision",
    "industry", "industries", "products", "global", "advanced", "applied", "analog", "digital",
    "data", "computer", "network", "networks", "national", "united", "general", "corp", "inc",
    "co", "company", "corporation", "incorporated", "limited", "ltd", "plc", "llc", "the",
}  # fmt: skip
PLACE_WORDS = {"texas", "taiwan", "china", "japan", "korea", "america", "american", "europe",
               "asia", "pacific", "california", "silicon"}  # fmt: skip


def token_case_stats(texts, tokens: set[str]) -> dict[str, list[int]]:
    """For each token: [occurrences, occurrences written with a capital first letter]."""
    out = {t: [0, 0] for t in tokens}
    for text in texts:
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9]*", text):
            low = m.group().lower()
            if low in out:
                out[low][0] += 1
                out[low][1] += m.group()[0].isupper()
    return out


def distinctive_filer_tokens(
    filers: dict[str, dict], stats: dict[str, list[int]], min_cap: float = 0.9, min_count: int = 5
) -> dict[str, str]:
    """token -> ticker, for name words that identify exactly one filer on their own
    ("skyworks", "nvidia", "cirrus"). A word qualifies if it is in only one filer's name, is
    not generic or a place, and the filings write it capitalised >= min_cap of the time, i.e.
    it is used as a proper noun ("applied", "onto", "diodes" fail: common lowercase words)."""
    owners: dict[str, set[str]] = defaultdict(set)
    for ticker, m in filers.items():
        for t in norm_key(clean_company(m["company"])).split():
            owners[t].add(ticker)
    out = {}
    for t, tick in owners.items():
        if len(tick) != 1 or len(t) < 3 or t in GENERIC_NAME_WORDS or t in PLACE_WORDS:
            continue
        n, cap = stats.get(t, [0, 0])
        if n >= min_count and cap / n >= min_cap:
            out[t] = next(iter(tick))
    return out


class Resolver:
    def __init__(
        self,
        filers: dict[str, dict],
        aliases: dict[str, dict[str, list[str]]],
        filer_tokens: dict[str, str] | None = None,
        token_caps: dict[str, float] | None = None,
    ):
        # key -> (entity id, canonical name, method)
        self.index: dict[tuple[str, str], tuple[str, str]] = {}
        self.filer_ids: dict[str, str] = {}
        self.filer_canon: dict[str, str] = {}
        # a distinctive word of one filer's name links to that filer ("Skyworks")
        self.filer_tokens = filer_tokens or {}
        # share of capitalised uses per name word; < 0.9 means also a common word ("diodes")
        self.token_caps = token_caps or {}
        for ticker, m in filers.items():
            eid, canon = f"company:{ticker}", clean_company(m["company"])
            self.filer_ids[ticker] = eid
            self.filer_canon[ticker] = canon
            names = {m["company"], canon, ticker, COMMON_NAMES.get(ticker, "")}
            keys = {norm_key(n) for n in names if n}
            for k in keys:
                if k:
                    self.index[("company", k)] = (eid, canon)
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
        if etype == "company" and self.filer_tokens:
            # "Skyworks Semiconductor", "Skyworks Inc": one distinctive filer word, the rest
            # generic -> that filer. Anything more specific ("Skyworks Japan") stays separate.
            toks = key.split()
            owners = {self.filer_tokens[t] for t in toks if t in self.filer_tokens}
            if len(owners) == 1 and all(
                t in self.filer_tokens or t in GENERIC_NAME_WORDS for t in toks
            ):
                tk = owners.pop()
                return self.filer_ids[tk], self.filer_canon[tk], "filer_token"
        if etype in ("company", "person", "geography", "regulation"):
            return f"{etype}:{_slug(key)}", name, "normalised"
        ck = concept_key(name)
        return f"{etype}:{_slug(ck)}", name, "normalised"


def _codes(key: str) -> set[str]:
    """Tokens that carry an identity in one character: numbers, and short letter+digit codes."""
    return {t for t in key.split() if any(ch.isdigit() for ch in t) or len(t) == 1}


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
                    if _codes(ids[a]) != _codes(ids[b]):
                        continue  # "country groups d1" vs "d4": codes and numbers must match
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
    stats = _name_word_stats(data_dir, snapshot, filers)
    caps = {t: (c / n if n else 0.0) for t, (n, c) in stats.items()}
    return Resolver(filers, aliases, distinctive_filer_tokens(filers, stats), caps)


def _name_word_stats(
    data_dir: Path, snapshot: str, filers: dict[str, dict]
) -> dict[str, list[int]]:
    """Capitalisation counts of every filer-name word over the corpus, cached on disk."""
    cache = data_dir / "graph" / snapshot / "name_word_case.json"
    words = {t for m in filers.values() for t in norm_key(clean_company(m["company"])).split()}
    words |= {t for m in filers.values() for t in norm_key(m["company"]).split()}
    if cache.exists():
        stats = json.loads(cache.read_text())
        if words <= set(stats):
            return stats
    from plexus.ingest.process import iter_docs

    stats = token_case_stats(
        (d.text for _, d in iter_docs(data_dir / "processed", snapshot)), words
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(stats, sort_keys=True) + "\n")
    return stats


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
