"""Look things up in the parsed filings while reviewing golden-set candidates.

python -m plexus.eval.inspect_filing quote <doc_id> "<quote>"     # is it verbatim? where? context
python -m plexus.eval.inspect_filing search <TICKER|doc_id> "<regex>" [--max 5]
python -m plexus.eval.inspect_filing docs [TICKER]                # list filings
python -m plexus.eval.inspect_filing section <doc_id> <section_name> [--chars 3000]
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache

from plexus.eval.metrics import locate_quote_all
from plexus.ingest.process import iter_docs
from plexus.settings import Settings


@lru_cache(maxsize=1)
def _docs(snapshot: str = "semis-2025"):
    out = {}
    for meta, doc in iter_docs(Settings.load().data_dir / "processed", snapshot):
        out[doc.doc_id] = (meta, doc)
    return out


def _ctx(text: str, s: int, e: int, width: int) -> str:
    return text[max(0, s - width) : s] + " >>>" + text[s:e] + "<<< " + text[e : e + width]


def cmd_quote(doc_id: str, quote: str, width: int) -> int:
    docs = _docs()
    if doc_id not in docs:
        print(f"unknown doc_id {doc_id}")
        return 1
    meta, doc = docs[doc_id]
    locs = locate_quote_all(doc.text, quote)
    print(f"{doc_id}: {len(locs)} verbatim occurrence(s)")
    for s, e in locs[:3]:
        sec = doc.section_at(s)
        print(
            f"--- section={sec.name if sec else '?'} chars {s}-{e}\n{_ctx(doc.text, s, e, width)}\n"
        )
    others = [d for d, (m, dd) in docs.items() if d != doc_id and m["ticker"] == meta["ticker"]
              and locate_quote_all(dd.text, quote)]  # fmt: skip
    if others:
        print(f"also verbatim in same-company filings: {others}")
    return 0


def cmd_search(target: str, pattern: str, max_hits: int, width: int) -> int:
    rx = re.compile(pattern, re.IGNORECASE)
    n = 0
    for doc_id, (meta, doc) in _docs().items():
        if target not in (doc_id, meta["ticker"]):
            continue
        for m in rx.finditer(doc.text):
            sec = doc.section_at(m.start())
            print(
                f"--- {doc_id} section={sec.name if sec else '?'}\n{_ctx(doc.text, m.start(), m.end(), width)}\n"
            )
            n += 1
            if n >= max_hits:
                return 0
    if n == 0:
        print("no matches")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quote")
    q.add_argument("doc_id")
    q.add_argument("quote")
    s = sub.add_parser("search")
    s.add_argument("target")
    s.add_argument("pattern")
    s.add_argument("--max", type=int, default=5)
    d = sub.add_parser("docs")
    d.add_argument("ticker", nargs="?")
    sec = sub.add_parser("section")
    sec.add_argument("doc_id")
    sec.add_argument("name")
    sec.add_argument("--chars", type=int, default=3000)
    for p in (q, s):
        p.add_argument("--width", type=int, default=500)
    a = ap.parse_args(argv)
    if a.cmd == "quote":
        return cmd_quote(a.doc_id, a.quote, a.width)
    if a.cmd == "search":
        return cmd_search(a.target, a.pattern, a.max, a.width)
    if a.cmd == "docs":
        for doc_id, (meta, doc) in _docs().items():
            if a.ticker in (None, meta["ticker"]):
                print(doc_id, meta["company"], [x.name for x in doc.sections])
        return 0
    meta, doc = _docs()[a.doc_id]
    for x in doc.sections:
        if x.name == a.name:
            print(doc.text[x.start : min(x.end, x.start + a.chars)])
            return 0
    print("no such section")
    return 1


if __name__ == "__main__":
    sys.exit(main())
