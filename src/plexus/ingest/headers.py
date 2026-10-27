"""Contextual chunk headers: who / what filing / which section, prepended at index time.

A chunk from Lam Research's risk factors usually says "we", never "Lam Research", so a
question naming the company cannot match it on the name, while a generic overview passage
that does name the company scores high (seen with the A3 reranker). Prepending a short header
to the text that is embedded, keyword-indexed and reranked puts every candidate on equal
footing for the company name. The chunk's own text and offsets are unchanged.
"""

from __future__ import annotations

import re

# Names people actually use where they differ from the SEC legal name. General knowledge,
# not tuned to the golden set; extend when the corpus grows.
COMMON_NAMES = {
    "AMD": "AMD",
    "ADI": "ADI",
    "AMAT": "Applied Materials",
    "MPWR": "MPS",
    "ON": "onsemi",
    "SLAB": "Silicon Labs",
    "TXN": "TI",
    "TSM": "TSMC",
}

SECTION_TITLES = {
    "mdna": "Management's Discussion and Analysis",
    "risk_factors": "Risk Factors",
    "unknown": "",
    "cover": "Cover page",
}

_LEGAL_SUFFIX = re.compile(r"\s*/[A-Z]+/?\s*$")  # "QUALCOMM INC/DE", "DIODES INC /DEL/"


def clean_company(name: str) -> str:
    name = _LEGAL_SUFFIX.sub("", name).strip().rstrip(",")
    if name.isupper():  # "LAM RESEARCH CORP" -> "Lam Research Corp"; keep real acronyms
        name = " ".join(w if w in _ACRONYMS else w.title() for w in name.split())
    return name


_ACRONYMS = {"KLA", "ON", "AMD", "NXP"}


def section_title(name: str) -> str:
    return SECTION_TITLES.get(name, name.replace("_", " ").title())


def chunk_header(meta: dict, section: str) -> str:
    ticker = meta.get("ticker", "")
    alias = COMMON_NAMES.get(ticker)
    who = clean_company(meta.get("company", ticker))
    tag = f"{ticker}, {alias}" if alias and alias.lower() not in who.lower() else ticker
    sec = section_title(section)
    period = meta.get("report_date", "")
    head = f"{who} ({tag}) {meta.get('form', '')} for the period ended {period}"
    return f"{head}. Section: {sec}." if sec else f"{head}."


def index_text(header: str, text: str) -> str:
    """The text that is embedded, BM25-indexed and reranked."""
    return f"{header}\n{text}" if header else text
