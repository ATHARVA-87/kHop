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
import contextlib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from cleanco import basename
from rapidfuzz import fuzz

from plexus.extract.schema import is_anonymous, strip_article
from plexus.ingest.headers import COMMON_NAMES, clean_company
from plexus.retrieve.bm25 import s_stem
from plexus.settings import REPO_ROOT, Settings

_LEGAL_TAIL = re.compile(
    r"[\s,]+(incorporated|inc|corporation|corp|company|co|limited|ltd|plc|n\.?\s?v|ag|s\.?a|"
    r"llc|gmbh|holdings?|group|technolog(?:y|ies))\.?$",
    re.I,
)
FUZZY_RATIO = 92.0  # rapidfuzz token_sort_ratio, 0-100
_EDGAR_STATE = re.compile(r"\s*/[A-Z]{2,}/?\s*$")  # "QUALCOMM INC/DE", "ARM HOLDINGS PLC /UK"


def norm_key(name: str) -> str:
    """'The Microsoft Corporation' -> 'microsoft'; 'Hon Hai Precision Industry Co., Ltd.' ->
    'hon hai precision industry'. Only TRAILING legal words are removed, so
    'ON Semiconductor' keeps 'semiconductor'."""
    n = _EDGAR_STATE.sub("", strip_article(" ".join(name.split())))
    # cleanco: the standard legal-suffix stripper ("Technologies A.G.", "Co., Ltd.")
    with contextlib.suppress(Exception):  # odd strings: keep the name as it was
        n = basename(basename(n)) or n
    prev = None
    while prev != n:
        prev = n
        n = _LEGAL_TAIL.sub("", n).strip(" ,.")
    return " ".join(re.findall(r"[a-z0-9]+", n.lower()))


_PURE_LEGAL = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
               "plc", "llc", "lp", "l", "p", "sa", "s", "a", "nv", "n", "v", "ag", "se", "pte",
               "gmbh", "bv", "the"}  # fmt: skip


def sec_key(name: str) -> str:
    """Key for the SEC gazetteer: the FULL name minus only pure legal words ("Inc", "Corp",
    "Ltd", EDGAR "/DE"). Unlike norm_key it keeps "Technologies", "Group", "Holdings", so
    "STAr Technologies" never matches "Star Group" and "Merck KGaA" never matches "Merck & Co"
    (review 2026-10-28). Keys shorter than two words are not used ("Target", "Global")."""
    n = _EDGAR_STATE.sub("", strip_article(" ".join(name.split())))
    toks = re.findall(r"[a-z0-9]+", n.lower())
    while toks and toks[-1] in _PURE_LEGAL:
        toks.pop()
    return " ".join(toks) if len(toks) >= 2 else ""


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
        sec: dict[str, tuple[str, str]] | None = None,
    ):
        # key -> (entity id, canonical name, method)
        self.index: dict[tuple[str, str], tuple[str, str]] = {}
        self.filer_ids: dict[str, str] = {}
        self.filer_canon: dict[str, str] = {}
        # a distinctive word of one filer's name links to that filer ("Skyworks")
        self.filer_tokens = filer_tokens or {}
        # share of capitalised uses per name word; < 0.9 means also a common word ("diodes")
        self.token_caps = token_caps or {}
        # SEC registrant list: normalised name -> (ticker, official title)
        self.sec = sec or {}
        for ticker, m in filers.items():
            eid, canon = f"company:{ticker}", clean_company(m["company"])
            self.filer_ids[ticker] = eid
            self.filer_canon[ticker] = canon
            names = {m["company"], canon, ticker, COMMON_NAMES.get(ticker, "")}
            keys = {norm_key(n) for n in names if n}
            for k in keys:
                if k:
                    self.index[("company", k)] = (eid, canon)
        for ticker, alts in (aliases.get("filers") or {}).items():
            if ticker in self.filer_ids:  # short names people use for a filer
                for n in alts:
                    self.index[("company", norm_key(n))] = (
                        self.filer_ids[ticker],
                        self.filer_canon[ticker],
                    )
        for etype, table in aliases.items():
            if etype == "filers":
                continue
            for canon, alts in table.items():
                key = (
                    ("company", norm_key(canon)) if etype == "company" else (etype, norm_key(canon))
                )
                default = (f"{etype}:{_slug(canon)}", canon)
                sk = sec_key(canon) if etype == "company" else ""
                if sk and sk in self.sec:  # e.g. "Applied Materials" -> company:AMAT
                    default = (f"company:{self.sec[sk][0]}", canon)
                eid, canon_name = self.index.get(key, default)
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
        sk = sec_key(name) if etype == "company" else ""
        if sk and sk in self.sec:
            ticker, title = self.sec[sk]
            return f"company:{ticker}", clean_company(title), "sec"
        if etype in ("company", "person", "geography", "regulation"):
            return f"{etype}:{_slug(key)}", name, "normalised"
        ck = concept_key(name)
        return f"{etype}:{_slug(ck)}", name, "normalised"


def _codes(key: str) -> set[str]:
    """Tokens that carry an identity in one character: numbers, and short letter+digit codes."""
    return {t for t in key.split() if any(ch.isdigit() for ch in t) or len(t) == 1}


def _subset_generic(a: str, b: str) -> bool:
    """Same name apart from generic words: 'samsung semiconductor' ~ 'samsung',
    'monolithic power system' ~ 'monolithic power systems' (after plural folding)."""
    ta = {s_stem(t) for t in a.split()}
    tb = {s_stem(t) for t in b.split()}
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    generic = {s_stem(t) for t in GENERIC_NAME_WORDS}
    return small <= big and (big - small) <= generic and bool(small - generic)


def fuzzy_merges(
    ids_by_type: dict[str, dict[str, str]],
    anchors_by_type: dict[str, list[tuple[str, str]]] | None = None,
) -> dict[str, tuple[str, float]]:
    """Map near-duplicate ids onto one representative.

    Candidates share a type and their first 4 letters (blocking). They merge when
    (a) rapidfuzz token_sort_ratio >= FUZZY_RATIO, or (b) they are the same name apart from
    generic words (`_subset_generic`); in both cases codes and numbers must match
    ("country groups d1" vs "d4" never merge). Anchors (filers, alias-table and SEC entities)
    are merge TARGETS only, so "Monolithic Power System" joins the MPS filer node."""
    remap: dict[str, tuple[str, float]] = {}
    anchors_by_type = anchors_by_type or {}

    def similar(x: str, y: str) -> float:
        if _codes(x) != _codes(y):
            return 0.0
        r = fuzz.token_sort_ratio(x, y)
        if r >= FUZZY_RATIO:
            return r
        return 100.0 if _subset_generic(x, y) else 0.0

    for etype, ids in ids_by_type.items():
        block_anchor: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for aid, key in anchors_by_type.get(etype, []):  # one entry per known name variant
            if key:
                block_anchor[key[:4]].append((aid, key))
        by_block: dict[str, list[str]] = defaultdict(list)
        for eid, key in ids.items():
            if key:
                by_block[key[:4]].append(eid)
        for block, group in by_block.items():
            group.sort()
            for eid in group:  # first: join a known entity if one matches
                best = max(((similar(ids[eid], k), a) for a, k in block_anchor.get(block, [])),
                           default=(0.0, ""))  # fmt: skip
                if best[0] > 0 and best[1] != eid:
                    remap[eid] = (best[1], round(best[0], 1))
            rest = [e for e in group if e not in remap]
            for i, a in enumerate(rest):  # then: merge unresolved variants with each other
                if a in remap:
                    continue
                for b in rest[i + 1 :]:
                    if b in remap:
                        continue
                    r = similar(ids[a], ids[b])
                    if r > 0:
                        # the shorter (more general) name is the representative
                        _keep, drop = (a, b) if len(ids[a]) <= len(ids[b]) else (b, a)
                        if drop == a:
                            remap[a] = (b, round(r, 1))
                            break
                        remap[b] = (a, round(r, 1))
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
                    # resolved names are canonical already; only "normalised" ones get merged
                    keys.setdefault(eid, norm_key(canon) if how == "normalised" else "")
                    ends.append(eid)
                raw_edges.append((ends[0], r["relation"], ends[1], r, rec))
    # fuzzy pass over ids that did not come from the alias table (those are already canonical)
    by_type: dict[str, dict[str, str]] = defaultdict(dict)
    for eid, k in keys.items():
        if k and not eid.startswith("company:anon:"):
            by_type[etypes[eid]][eid] = k
    anchors: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (etype, key), (aid, _canon) in resolver.index.items():
        anchors[etype].append((aid, key))
    for eid, how_key in keys.items():  # SEC / filer-word entities seen in this corpus
        if not how_key and not eid.startswith("company:anon:"):
            anchors[etypes[eid]].append((eid, norm_key(names[eid].most_common(1)[0][0])))
    remap = fuzzy_merges(by_type, anchors)

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
    return Resolver(filers, aliases, distinctive_filer_tokens(filers, stats), caps,
                    load_sec_gazetteer(data_dir))  # fmt: skip


def load_sec_gazetteer(data_dir: Path) -> dict[str, tuple[str, str]]:
    """data/reference/sec_company_tickers.json (fetched with checksum, see .meta.json):
    normalised registrant name -> (ticker, title). First entry wins for share classes."""
    path = data_dir / "reference" / "sec_company_tickers.json"
    meta = data_dir / "reference" / "sec_company_tickers.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m plexus.ingest.sec_reference`")
    raw = path.read_bytes()
    want = json.loads(meta.read_text())["sha256"]
    if hashlib.sha256(raw).hexdigest() != want:
        raise ValueError(f"{path} does not match the sha256 in {meta.name}")
    out: dict[str, tuple[str, str]] = {}
    for row in json.loads(raw).values():
        k = sec_key(row["title"])
        if k:
            out.setdefault(k, (row["ticker"].upper(), row["title"]))
    return out


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
    # validated output (current rules, `extract --revalidate`) if present, else the raw run
    validated = gdir / "extractions_validated.jsonl"
    files = a.extractions or [validated if validated.exists() else gdir / "extractions.jsonl"]
    print(json.dumps(build(files, load_resolver(s.data_dir, a.snapshot), a.out or gdir), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
