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

_ITEM_RE = re.compile(r"^\s*item\s*(\d{1,2}[a-c]?)\s*[\.\:\-–—]?\s*(.{0,150})$", re.IGNORECASE)
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


def _heading_candidates(lines: list[str], form: str) -> list[tuple[int, str, str]]:
    """(line_index, key, title) for every line that looks like an Item heading."""
    out = []
    part = "I"
    for i, line in enumerate(lines):
        pm = _PART_RE.match(line)
        if pm and len(line) < 120:
            part = pm.group(1).upper()
            # "PART I — Item 1. Financial Statements" on one line is handled below.
        m = _ITEM_RE.match(line) or (
            _ITEM_RE.match(line[pm.end() :].strip(" .:-–—")) if pm else None
        )
        if not m:
            continue
        item = m.group(1).upper()
        title = m.group(2).strip(" .:-|")
        if not title and i + 1 < len(lines):
            title = lines[i + 1][:150]
        key = f"{part}-{item}" if form.startswith("10-Q") else item
        out.append((i, key, title))
    return out


def parse_filing(doc_id: str, html: str | bytes, form: str) -> ParsedDoc:
    text = html_to_text(html)
    lines = text.split("\n")
    offsets = [0]
    for line in lines[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)

    names = TENQ_NAMES if form.startswith("10-Q") else TENK_NAMES
    cands = [c for c in _heading_candidates(lines, form) if c[1] in names]

    # Span of each candidate = until the next candidate heading. Keep the longest per key
    # (the TOC entry is a few characters long; the real section is thousands).
    best: dict[str, tuple[int, int, str]] = {}
    for n, (i, key, title) in enumerate(cands):
        start = offsets[i]
        end = offsets[cands[n + 1][0]] if n + 1 < len(cands) else len(text)
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (start, end, title)

    # The chosen spans can overlap when a TOC entry was the longest for some tiny item;
    # sort by start and clip each section at the next one's start.
    chosen = sorted(best.items(), key=lambda kv: kv[1][0])
    sections: list[Section] = []
    if chosen and chosen[0][1][0] > 0:
        sections.append(Section("cover", "cover", "Cover", 0, chosen[0][1][0]))
    for n, (key, (start, end, title)) in enumerate(chosen):
        if n + 1 < len(chosen):
            end = min(end, chosen[n + 1][1][0])
        if end > start:
            sections.append(Section(key, names[key], title, start, end))
    if sections:
        sections[-1].end = len(text)
    else:
        sections.append(Section("cover", "unknown", "", 0, len(text)))
    return ParsedDoc(doc_id=doc_id, text=text, sections=sections)
