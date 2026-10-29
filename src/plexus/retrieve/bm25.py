"""BM25 keyword retrieval over chunks (pure Python + numpy, no extra dependencies).

Why: dense embeddings blur exact terms ("PIM business", "Mohawk Valley", "CodeFusion Studio").
On A0's misses, 33 of 38 local questions retrieved the right company but the wrong passage,
which is where exact-term matching helps.

Scoring is Okapi BM25 with Lucene's IDF: idf = ln(1 + (N - df + 0.5) / (df + 0.5)).
Tokens: lowercase alphanumerics (decimals kept, e.g. "4.5"), English function words dropped,
plurals folded with Harman's S-stemmer ("foundries" -> "foundry").
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

TOKENIZER_VERSION = "1"


def _fingerprint() -> str:
    """Saved indexes are tied to this file's exact source, so any edit to the tokenizer,
    stemmer or stopwords rebuilds them (a stale cache would hide a regression)."""
    return TOKENIZER_VERSION + ":" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = frozenset(
    (
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "during",
        "each",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    )
)


def s_stem(word: str) -> str:
    """Harman (1991) S-stemmer: folds common English plurals only, so it rarely over-stems."""
    if len(word) > 3 and word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if len(word) > 2 and word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in STOPWORDS or (len(t) == 1 and not t.isdigit()):
            continue
        out.append(t if t[0].isdigit() else s_stem(t))
    return out


class BM25Index:
    def __init__(
        self,
        postings: dict[str, tuple[np.ndarray, np.ndarray]],
        doc_len: np.ndarray,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.postings = postings  # term -> (doc indices int32, term frequencies float32)
        self.doc_len = doc_len.astype(np.float32)
        self.n_docs = len(doc_len)
        self.set_params(k1, b)

    def set_params(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        avgdl = float(self.doc_len.mean()) if self.n_docs else 1.0
        # per-document length normaliser, reused for every query
        self._norm = k1 * (1.0 - b + b * self.doc_len / max(avgdl, 1e-9))

    @classmethod
    def build(cls, texts: Iterable[str], k1: float = 1.2, b: float = 0.75) -> BM25Index:
        acc: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lens = []
        for i, text in enumerate(texts):
            toks = tokenize(text)
            lens.append(len(toks))
            for term, tf in Counter(toks).items():
                acc[term].append((i, tf))
        postings = {
            t: (np.array([d for d, _ in p], np.int32), np.array([f for _, f in p], np.float32))
            for t, p in acc.items()
        }
        return cls(postings, np.array(lens, np.float32), k1, b)

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0]) if term in self.postings else 0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n_docs, np.float32)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            s[docs] += self.idf(term) * tf * (self.k1 + 1.0) / (tf + self._norm[docs])
        return s

    def search(
        self, query: str, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """allowed: boolean mask over documents (chunks); others score 0."""
        s = self.scores(query)
        if allowed is not None:
            s = np.where(allowed, s, 0.0).astype(np.float32)
        k = min(k, self.n_docs)
        if k <= 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.lexsort((top, -s[top]))]  # score desc, then index asc: deterministic
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"version": _fingerprint(), "postings": self.postings, "doc_len": self.doc_len},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path, k1: float = 1.2, b: float = 0.75) -> BM25Index | None:
        """Our own cache file; returns None if it was written by another tokenizer version."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)  # our own file from save(), never downloaded
        if d.get("version") != _fingerprint():
            return None
        return cls(d["postings"], d["doc_len"], k1, b)


def corpus_token_count(texts: Sequence[str]) -> int:
    return sum(len(tokenize(t)) for t in texts)
