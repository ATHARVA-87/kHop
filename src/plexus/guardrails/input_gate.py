"""Input gate: refuse before any search (Phase 5, step 2). Plain rules, no LLM.

Order: rule-breaking attempts -> investment advice -> off-scope. Kept only if it refuses none of
the 134 dev questions and none of the 23 "should answer" questions (rule fixed 2026-10-29).
The advice rule is narrow on purpose: the owner wants "price target" / "overvalued" /
"recommend stocks" questions answered with filing facts (no opinion), not refused.
"""

from __future__ import annotations

import re

JAILBREAK_REPLY = "nah, I'd win."  # the owner's joke for "ignore your rules" (2026-10-29)
ADVICE_REPLY = (
    "I can't help with investment advice. Plexus answers factual questions about what "
    "semiconductor companies say in their SEC filings."
)
SCOPE_REPLY = (
    "I can't help with that: Plexus only answers questions about the SEC filings of the "
    "semiconductor companies in its corpus."
)

# the object must be the assistant's own rules ("ignore your rules", "ignore all previous
# instructions"), so filing questions like "bypass export restrictions" pass (review 2026-10-29)
JAILBREAK = re.compile(
    r"\b(ignore|disregard|forget|bypass|override)\s+(all\s+)?(of\s+)?(your|the|my|any|these|those)?"
    r"\s*(previous\s+|prior\s+|above\s+|earlier\s+)?(rules|instructions|guidelines|system prompt|"
    r"prompt|restrictions)\b|\bpretend (you are|to be)\b|\bdeveloper mode\b|\bjailbreak\b"
    r"|(?-i:\bDAN\b)",  # DAN only in capitals: "Dan Durn" is a CFO, not a jailbreak
    re.IGNORECASE,
)
ADVICE = re.compile(
    r"\bshould (i|we|my \w+) (buy|sell|invest|short|hold|put)\b"
    r"|\b(would|will) you (buy|sell|invest|short)\b"
    r"|\b(good|safe|smart|great|bad) (investment|bet|buy|stock to)\b|\bsafe bet\b"
    r"|\bgood time to (buy|sell|short|invest)\b|\bsmart to (buy|sell|short|invest)\b"
    r"|\bbuy/hold/sell\b|\b(buy|hold|sell) rating\b"
    r"|\b(my|your) (portfolio|savings|retirement)\b|\bhow much of my\b"
    r"|\b(which|what) (\w+ ){0,4}(stocks?|shares?) should (i|we)\b"
    r"|\bstock('s)? will go up\b|\b(will|could) .{0,40}\b(shares|stock) (double|go up|rise|soar)\b"
    r"|\bmoney i would make\b|\bbetter buy\b|\bbest dividend stock\b"
    r"|\b(buy|sell|trade) .{0,20}\boptions\b|\bwhich chip stock\b"
    r"|\binvest in .{0,40}\bthis year\b|\bi have \$",
    re.IGNORECASE,
)
# words that place a question inside the corpus even when it names no company
IN_DOMAIN = re.compile(
    r"\b(semiconductor|chip|chips|chipmaker|wafer|fab|fabs|foundry|foundries|lithography|"
    r"memory|dram|nand|gpu|cpu|analog|silicon|export control|10-k|10-q|sec filing|filings?|"
    r"risk factors?|supplier|customers?|revenue|companies)\b",
    re.IGNORECASE,
)


MAX_CHARS = 1000  # docs/security.md: question length cap
TOO_LONG_REPLY = f"Please ask a shorter question (at most {MAX_CHARS} characters)."


def gate(question: str, linker=None) -> tuple[str, str] | None:
    """(kind, reply) to refuse with, or None to let the question through."""
    if len(question) > MAX_CHARS:
        return "too_long", TOO_LONG_REPLY
    if JAILBREAK.search(question):
        return "jailbreak", JAILBREAK_REPLY
    if ADVICE.search(question):
        return "advice", ADVICE_REPLY
    if linker is not None and not linker.link(question) and not IN_DOMAIN.search(question):
        return "scope", SCOPE_REPLY
    return None
