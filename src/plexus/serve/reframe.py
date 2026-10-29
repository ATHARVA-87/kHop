"""Reframe layer: one canonical question in the filings' own wording (owner request 2026-10-29).

Different phrasings of the same question ("nvidia company", "nvidia companies", "nvdia companies")
gave different answers: a typo hid the company (wrong search route) and small wording changes
moved the top passages. Two plain-code steps, no LLM call (research 2026-10-29):

1. Spelling: a word is corrected only if it is unknown to the filings, has 4+ letters, no
   capitals after the first letter and no digits, and exactly one clearly best corpus word or
   company name is within 1 edit (4-7 letters) or 2 edits (8+ letters; a swap counts as one).
2. Company names: every mention of a covered company becomes its canonical name
   ("nvidia company" -> "NVIDIA"); a trailing "company / companies / inc / corp" is dropped.
   Words are never reordered.

The user sees the rewritten question first and chooses "Use this version" or "Ask as I typed".
Kept only if it passes data/golden/reframe/PROTOCOL.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rapidfuzz import process
from rapidfuzz.distance import OSA

from plexus.settings import REPO_ROOT

MIN_FREQ_KNOWN = 2  # a corpus word seen at least this often is never "corrected"
MIN_FREQ_TARGET = 20  # a correction must be a word the filings use at least this often
MIN_FREQ_SHORT = 500  # short words (2-3 letters) are targets only if very common: for, the, how
# the first word of a question is usually one of these, so there they win ties ("Ihat" -> What)
# never changed, with or without a system word list (CI and the demo image have none)
PROTECTED = frozenset(
    {"inc", "corp", "ltd", "llc", "plc", "co", "nv", "sa", "ag", "gmbh", "bv", "lp"}
)
QUESTION_WORDS = frozenset({"what", "how", "which", "why", "who", "when", "where", "does", "do", "did",
                            "is", "are", "was", "were", "can", "will", "has", "have", "list", "describe", "compare"})  # fmt: skip
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
TRAILING = r"(?:\s+(?:company|companies|corp\.?|corporation|inc\.?)(?![\w-]))?"


@dataclass
class Reframed:
    original: str
    question: str
    changes: list[dict] = field(default_factory=list)  # {"from", "to", "kind"}

    @property
    def changed(self) -> bool:
        """True when a word changed, not only letter case or spacing (then the user is asked)."""
        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
        return norm(self.original) != norm(self.question)


def build_vocab(chunks_path: Path, cache_path: Path) -> Counter:
    """Word counts of the parsed filings (lowercase, not stemmed), cached next to the index."""
    stamp = f"{chunks_path.stat().st_size}:{int(chunks_path.stat().st_mtime)}"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("stamp") == stamp:
            return Counter(cached["counts"])
    counts: Counter = Counter()
    with open(chunks_path) as fh:
        for line in fh:
            counts.update(re.findall(r"[a-z][a-z'-]+", json.loads(line)["text"].lower()))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"stamp": stamp, "counts": counts}))
    return counts


def english_words() -> frozenset[str]:
    """The system's English word list (e.g. /usr/share/dict/words), used ONLY to recognise valid
    words so they are never "corrected" ("falling" is not a typo of "failing"). Empty if absent."""
    for path in (Path("/usr/share/dict/words"), Path("/usr/share/dict/american-english")):
        if path.exists():
            return frozenset(
                w.strip().lower() for w in path.read_text(errors="ignore").splitlines() if w.strip()
            )
    return frozenset()


def company_forms() -> dict[str, str]:
    """Every way a covered company is written -> its canonical display name."""
    names = yaml.safe_load((REPO_ROOT / "configs" / "companies.yaml").read_text())
    aliases = (
        yaml.safe_load((REPO_ROOT / "configs" / "entity_aliases.yaml").read_text()) or {}
    ).get("filers", {})
    forms: dict[str, str] = {}
    for ticker, display in names.items():
        forms[display] = display
        for a in aliases.get(ticker, []) or []:
            forms[a] = display
    return forms


class Reframer:
    def __init__(
        self, vocab: Counter, forms: dict[str, str], english: frozenset[str] | None = None
    ):
        self.vocab = vocab
        self.forms = forms
        # correction targets: frequent corpus words + every word of every company form
        name_words = {w.lower(): w for f in forms for w in f.split() if len(w) >= 4}
        self.proper = name_words  # lowercase -> how the name is written
        self.name_targets = sorted(name_words)
        # recognise valid words only, never a correction source; tests pass frozenset() so they
        # behave the same on machines without a system word list
        self.english = english_words() if english is None else english
        targets = (
            {
                w
                for w, n in vocab.items()
                if (len(w) >= 4 and n >= MIN_FREQ_TARGET)
                or (2 <= len(w) <= 3 and n >= MIN_FREQ_SHORT)
            }
            | QUESTION_WORDS
            | set(name_words)
        )
        self.targets = sorted(targets)
        self.patterns = []
        # only the official names are rewritten; short forms the linker already knows (TI, ON
        # Semi, MPS) stay as typed, so a question is not "changed" just for using one
        for display in sorted(set(forms.values()), key=len, reverse=True):
            flags = 0 if len(display) <= 3 else re.IGNORECASE
            self.patterns.append(
                (re.compile(rf"(?<![\w-]){re.escape(display)}(?![\w-]){TRAILING}", flags), display)
            )

    @classmethod
    def from_index(cls, chunks_path: Path, cache_dir: Path) -> Reframer:
        return cls(build_vocab(chunks_path, cache_dir / "reframe_vocab.json"), company_forms())

    def _fix_word(self, word: str, first: bool = False) -> str | None:
        core = re.sub(r"(['’]s?)$", "", word)  # possessives: "Micron's", "Texas Instruments'"
        if len(core) < 3 or any(c.isdigit() for c in core):
            return None
        if len(core) == 3 and core.isupper():
            return None  # tickers and short codes (TSM, KLA, AMD) are never changed
        low = core.lower()
        if low in PROTECTED:
            return None
        if self.vocab.get(low, 0) >= MIN_FREQ_KNOWN or low in self.proper or low in self.english:
            return None  # a known word is never changed
        # words with capitals inside (CoWoS, EBITDA, "NVIDbA") may only become a company name
        names_only = any(c.isupper() for c in core[1:])
        pool = self.name_targets if names_only else self.targets
        max_d = 1 if len(low) <= 7 else 2
        found = process.extract(low, pool, scorer=OSA.distance, score_cutoff=max_d, limit=None)
        if not found:
            return None

        def freq(w: str) -> int:
            boost = 10**6 if w in self.proper else 0
            if first and w in QUESTION_WORDS:
                boost += 10**7  # a question usually starts with a question word
            return self.vocab.get(w, 0) + boost

        found.sort(key=lambda x: (x[1], -freq(x[0])))
        best, dist = found[0][0], found[0][1]
        ties = [f for f in found[1:] if f[1] == dist]
        if ties and freq(best) < 10 * freq(ties[0][0]):
            return None  # no clear winner: leave the word alone
        fixed = self.proper.get(best) or (best.capitalize() if core[0].isupper() else best)
        return fixed + word[len(core) :]

    def __call__(self, question: str) -> Reframed:
        if not self.english:
            # without a word list the layer changed 10/225 correct questions and every trap word
            # (measured 2026-10-29), so it switches itself off: questions pass exactly as typed.
            # The demo image installs one (Debian package `wamerican`).
            return Reframed(question, question)
        changes: list[dict] = []
        out, last = [], 0
        quoted = [m.span() for m in re.finditer(r"\"[^\"]*\"|“[^”]*”", question)]
        for m in WORD.finditer(question):
            if any(a <= m.start() < b for a, b in quoted):
                continue
            fixed = self._fix_word(m.group(), first=not question[: m.start()].strip())
            if fixed and fixed != m.group():
                out.append(question[last : m.start()] + fixed)
                last = m.end()
                changes.append({"from": m.group(), "to": fixed, "kind": "spelling"})
        text = "".join(out) + question[last:]
        for pattern, display in self.patterns:

            def sub(m, display=display):
                if m.group(0) != display:
                    changes.append({"from": m.group(0), "to": display, "kind": "company"})
                return display

            text = pattern.sub(sub, text)
        text = re.sub(r"\bthe\s+(?=(" + "|".join(re.escape(d) for d in set(self.forms.values())) + r")\b(?!\s+(?:company|companies)))",
                      "", text, flags=re.IGNORECASE) if changes else text  # fmt: skip
        return Reframed(question, re.sub(r"\s{2,}", " ", text).strip(), changes)
