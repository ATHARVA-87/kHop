"""Quote grounding: does an LLM's "verbatim" quote really come from the passage?

Exact matching dropped 588 of 3,513 extracted relations, mostly for trivial reasons: lists
shortened with "...", `"TSMC"` vs `("TSMC")`, a changed comma. Following standard fuzzy span
alignment (rapidfuzz `partial_ratio_alignment`, the research scout's top recommendation):

1. normalise both texts (NFKC, curly -> straight quotes, punctuation -> space, casefold), keeping
   a map back to the original characters;
2. split the quote on "..." / "…" and require every piece (3+ words) to align, IN ORDER, with
   score >= cutoff, each within `max_gap` characters of the previous piece;
3. return the passage's OWN span, so the stored citation is verbatim filing text even when the
   model's copy was slightly off.
The cutoff is calibrated on a labelled sample of the previously dropped quotes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

_ELLIPSIS = re.compile(r"\.{3,}|…")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-"})
DEFAULT_CUTOFF = 95.0


@dataclass(frozen=True)
class Alignment:
    start: int  # offsets into the ORIGINAL passage text
    end: int
    score: float  # lowest piece score, 0-100
    pieces: int


def normalise(text: str) -> tuple[str, list[int]]:
    """Lowercased alphanumerics separated by single spaces, plus, for every output character,
    its index in `text`."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch.isalnum():
            out.append(ch.casefold())
            idx.append(i)
        elif out and out[-1] != " ":
            out.append(" ")
            idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


_FUNCTION_WORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
                   "as", "at", "our", "we", "its", "is", "are", "be", "that", "which", "such"}  # fmt: skip


def ordered_match(quote: str, passage: str, slack: float = 3.0) -> Alignment | None:
    """Every content word of the quote appears in the passage IN ORDER, within a window of
    `slack` x the quote's length (+30 words). This accepts quotes that silently skip list
    items ("our competitors include, among others, Marvell" when the filing lists eight
    names before Marvell): calibration showed those are real edges that score low on
    character alignment."""
    q = [t for t in normalise(quote)[0].split() if t not in _FUNCTION_WORDS]
    if len(q) < 3:
        return None
    norm_p, idx = normalise(passage)
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", norm_p)]
    window = int(len(q) * slack) + 30
    for i, (t, _, _) in enumerate(toks):
        if t != q[0]:
            continue
        j, k = i, 0
        while j < len(toks) and j - i <= window and k < len(q):
            if toks[j][0] == q[k]:
                k += 1
            j += 1
        if k == len(q):
            s, e = toks[i][1], toks[j - 1][2]
            return Alignment(idx[s], idx[e - 1] + 1, 100.0 * len(q) / (j - i), 1)
    return None


def align_quote(
    quote: str, passage: str, cutoff: float = DEFAULT_CUTOFF, max_gap: int = 300
) -> Alignment | None:
    """Fuzzy span alignment of the quote's '...'-separated pieces; if that fails, an
    in-order word match (`ordered_match`)."""
    al = _align_pieces(quote, passage, cutoff, max_gap)
    return al if al is not None else ordered_match(quote, passage)


def _align_pieces(quote: str, passage: str, cutoff: float, max_gap: int) -> Alignment | None:
    norm_p, idx = normalise(passage)
    pieces = [normalise(p)[0] for p in _ELLIPSIS.split(quote)]
    pieces = [p for p in pieces if len(p.split()) >= 3]
    if not pieces or not norm_p:
        return None
    pos = 0
    first = last = None
    worst = 100.0
    for p in pieces:
        if p in norm_p[pos:]:  # exact after normalisation: fast path
            s = norm_p.index(p, pos)
            e, score = s + len(p), 100.0
        else:
            a = fuzz.partial_ratio_alignment(p, norm_p[pos:], score_cutoff=cutoff)
            if a is None:
                return None
            s, e, score = pos + a.dest_start, pos + a.dest_end, a.score
        if last is not None and s - last > max_gap:
            return None  # pieces too far apart to be one quoted statement
        first = s if first is None else first
        last, pos = e, e
        worst = min(worst, score)
    return Alignment(idx[first], idx[last - 1] + 1, worst, len(pieces))


def best_score(quote: str, passage: str) -> float:
    """Diagnostic: the alignment score with no cutoff (for calibrating the cutoff)."""
    a = _align_pieces(quote, passage, cutoff=0.0, max_gap=10**9)
    return a.score if a else 0.0
