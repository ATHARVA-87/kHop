"""Filing HTML -> clean text with Item-level sections.

Two hard parts:
1. EDGAR filings are inline-XBRL HTML: hidden XBRL headers, spans everywhere, tables
   for layout. We keep block structure (one block per line) and drop hidden content.
2. The table of contents repeats every "Item 1A. Risk Factors" heading. For each Item
   we keep the occurrence whose section body is longest, which is the real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment

BLOCK_TAGS = [
    "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table",
    "section", "article", "ul", "ol", "dt", "dd", "blockquote", "pre",
]  # fmt: skip

# "Item 1A." / "ITEM 7 —" / "Items 1 and 2." / glued spans "Item 1Business".
# The item number must not run on into digits or lowercase letters, so "Item 105 of
# Regulation S-K" and "Item 1business" are not headings, but "Item 1Business" is Item 1.
_ITEM_RE = re.compile(
    r"^\s*(?i:items?)\s*(\d{1,2}(?:[A-C]|[a-c](?![a-z]))?)(?![0-9a-z])"
    r"(?:\s*(?i:and|&)\s*\d{1,2}[A-Ca-c]?)?\s*[\.\:\-–—]?\s*(.{0,150})$"
)
# A TOC row ends with a page number cell ("Risk Factors | 12") or a bare trailing number.
_TOC_ROW = re.compile(r"(\|\s*\d{1,3}|\s\d{1,3})\s*$")
VERSION = "2"  # bump when parsing changes: it is part of the index key
_PART_RE = re.compile(r"^\s*part\s+(iv|i{1,3})\b", re.IGNORECASE)
_NOISE_LINE = re.compile(r"^(table of contents|index|\d{1,3}|page \d+)$", re.IGNORECASE)

# Section keys -> canonical label. 10-K items are bare ("1A"); 10-Q items carry the part.
TENK_NAMES = {
    "1": "business", "1A": "risk_factors", "1B": "unresolved_staff_comments",
    "1C": "cybersecurity", "2": "properties", "3": "legal_proceedings", "4": "mine_safety",
    "5": "market_for_equity", "6": "reserved", "7": "mdna", "7A": "market_risk",
    "8": "financial_statements", "9": "accountant_changes", "9A": "controls",
    "9B": "other_information", "9C": "foreign_jurisdiction_disclosure",
    "10": "directors_officers", "11": "executive_compensation", "12": "security_ownership",
    "13": "relationships", "14": "accountant_fees", "15": "exhibits", "16": "summary",
}  # fmt: skip
TENQ_NAMES = {
    "I-1": "financial_statements", "I-2": "mdna", "I-3": "market_risk", "I-4": "controls",
    "II-1": "legal_proceedings", "II-1A": "risk_factors", "II-2": "equity_sales",
    "II-3": "defaults", "II-4": "mine_safety", "II-5": "other_information", "II-6": "exhibits",
}  # fmt: skip


@dataclass
class Section:
    key: str  # "1A" (10-K) or "II-1A" (10-Q) or "cover"
    name: str  # canonical, e.g. "risk_factors"
    title: str  # heading text as written
    start: int  # char offsets into ParsedDoc.text
    end: int


@dataclass
class ParsedDoc:
    doc_id: str
    text: str
    sections: list[Section] = field(default_factory=list)

    def section_at(self, offset: int) -> Section | None:
        for s in self.sections:
            if s.start <= offset < s.end:
                return s
        return None


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "ix:header"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()
    for td in soup.find_all(["td", "th"]):
        td.append(" | ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    raw = soup.get_text("")
    lines = []
    for line in raw.replace("\xa0", " ").replace("​", "").splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip(" |")
        line = re.sub(r"(\s*\|\s*)+", " | ", line).strip()
        if line and not _NOISE_LINE.match(line):
            lines.append(line)
    return "\n".join(lines)


# 10-Q item numbers repeat across Part I and Part II; the title tells them apart even when
# the body has no standalone "PART II" line.
_TENQ_PART_BY_TITLE = [
    ("II", re.compile(r"legal proceedings|risk factors|unregistered|defaults|mine safety|other information|exhibits", re.I)),
    ("I", re.compile(r"financial statements|management.s discussion|quantitative|controls and procedures", re.I)),
]  # fmt: skip


def _heading_candidates(lines: list[str], form: str) -> list[tuple[int, str, str, bool]]:
    """(line_index, key, title, looks_like_toc_row) for every Item-heading-like line."""
    out = []
    part = "I"
    for i, line in enumerate(lines):
        pm = _PART_RE.match(line)
        if pm and len(line) < 120:
            part = pm.group(1).upper()
        # "PART I — Item 1. Financial Statements" on one line
        m = _ITEM_RE.match(line) or (
            _ITEM_RE.match(line[pm.end() :].strip(" .:-–—|")) if pm else None
        )
        if not m:
            continue
        item = m.group(1).upper()
        title = m.group(2).strip(" .:-|")
        if not title and i + 1 < len(lines):
            title = lines[i + 1][:150]
        toc_row = bool(_TOC_ROW.search(line))
        if form.startswith("10-Q"):
            p = next((pt for pt, rx in _TENQ_PART_BY_TITLE if rx.search(title)), part)
            key = f"{p}-{item}"
        else:
            key = item
        out.append((i, key, title, toc_row))
    return out


def _drop_toc(
    cands: list, offsets: list[int], order: list[str], max_gap: int = 120, min_run: int = 4
) -> list:
    """Remove table-of-contents entries.

    1. Rows ending in a page number ("Risk Factors | 12").
    2. Dense blocks: >= min_run headings, each within max_gap chars of the next, in
       increasing item order. The block's last entry (whose gap to the first body heading is
       long) is included only if it continues the increasing order; the body's first heading
       restarts at a lower item, so it is never swallowed.
    """
    cands = [c for c in cands if not c[3]]
    rank = {k: n for n, k in enumerate(order)}
    drop = [False] * len(cands)
    j = 0
    while j < len(cands) - 1:
        a = j
        while (
            j < len(cands) - 1
            and offsets[cands[j + 1][0]] - offsets[cands[j][0]] <= max_gap
            and rank[cands[j + 1][1]] > rank[cands[j][1]]
        ):
            j += 1
        # cands[a..j] are linked by short gaps; cands[j] is the block's last entry.
        if j - a + 1 >= min_run:
            for t in range(a, j + 1):
                drop[t] = True
        j = max(j, a + 1) if j == a else j + 1
    return [c for c, d in zip(cands, drop, strict=True) if not d]


def parse_filing(doc_id: str, html: str | bytes, form: str) -> ParsedDoc:
    text = html_to_text(html)
    lines = text.split("\n")
    offsets = [0]
    for line in lines[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)

    names = TENQ_NAMES if form.startswith("10-Q") else TENK_NAMES
    cands = [c for c in _heading_candidates(lines, form) if c[1] in names]
    cands = _drop_toc(cands, offsets, list(names))

    # If a key still occurs more than once (page headers, a line-initial cross-reference),
    # keep the occurrence with the longest span up to the next candidate.
    best: dict[str, tuple[int, int, str]] = {}
    for n, (i, key, title, _) in enumerate(cands):
        start = offsets[i]
        end = offsets[cands[n + 1][0]] if n + 1 < len(cands) else len(text)
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (start, end, title)

    # Sections tile the text: each runs from its heading to the next CHOSEN heading.
    chosen = sorted(((start, key, title) for key, (start, _, title) in best.items()))
    sections: list[Section] = []
    first = chosen[0][0] if chosen else len(text)
    if first > 0:
        sections.append(Section("cover", "cover" if chosen else "unknown", "Cover", 0, first))
    for n, (start, key, title) in enumerate(chosen):
        end = chosen[n + 1][0] if n + 1 < len(chosen) else len(text)
        sections.append(Section(key, names[key], title, start, end))
    if not sections:
        sections.append(Section("cover", "unknown", "", 0, len(text)))
    return ParsedDoc(doc_id=doc_id, text=text, sections=sections)
