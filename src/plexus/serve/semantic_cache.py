"""Semantic cache (A8): reuse a finished answer when the same question is asked in other words.

A stored answer is reused only if ALL of these hold:
1. same data snapshot and same app setup (config hash), and the entry is younger than the TTL;
2. the same companies / partners and the same time periods are named (exact match of the
   linker's entities and of period words: years, months, quarters). Meaning-similarity
   alone cannot be trusted here: "AMD's risks" and "Intel's risks" read as near-identical
   to an embedding model (research 2026-10-29);
3. cosine similarity of the two questions' bge-small embeddings >= threshold.
Only answers whose sentences all passed the verifier are stored.

The threshold was chosen on cache pairs C-01..C-20 and reported on C-21..C-40
(data/golden/cache_pairs.csv, owner-checked): rule fixed before the first run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

PERIOD = re.compile(
    r"\b((?:19|20)\d{2}|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|q[1-4]|"
    r"first|second|third|fourth|half|nine months|six months|fiscal)\b",
    re.IGNORECASE,
)


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?. ")


def key_of(question: str, linker) -> str:
    """The exact part of the key: who and when the question is about."""
    ents = sorted(set(linker.link(question))) if linker is not None else []
    periods = sorted({p.lower()[:3] for p in PERIOD.findall(question)})
    return json.dumps({"e": ents, "p": periods})


class SemanticCache:
    def __init__(
        self, path: Path, encode, linker, scope: str, threshold: float, ttl_s: float = 7 * 86400
    ):
        """encode(list[str]) -> unit-length vectors; scope = snapshot + config hash."""
        self.encode, self.linker, self.scope = encode, linker, scope
        self.threshold, self.ttl_s = threshold, ttl_s
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS entries (scope TEXT, k TEXT, question TEXT, vec BLOB, "
            "payload TEXT, created REAL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS by_key ON entries (scope, k)")

    def lookup(self, question: str, now: float | None = None) -> tuple[dict, float, str] | None:
        """(payload, similarity, original question) of the best match, or None."""
        now = time.time() if now is None else now
        rows = self.db.execute(
            "SELECT question, vec, payload, created FROM entries WHERE scope=? AND k=?",
            (self.scope, key_of(question, self.linker)),
        ).fetchall()
        rows = [r for r in rows if now - r[3] <= self.ttl_s]
        if not rows:
            return None
        q = self.encode([normalize(question)])[0]
        sims = [float(np.frombuffer(r[1], dtype=np.float32) @ q) for r in rows]
        best = int(np.argmax(sims))
        if sims[best] < self.threshold:
            return None
        return json.loads(rows[best][2]), sims[best], rows[best][0]

    def store(self, question: str, payload: dict, now: float | None = None) -> None:
        vec = self.encode([normalize(question)])[0].astype(np.float32).tobytes()
        self.db.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?)",
            (self.scope, key_of(question, self.linker), question, vec, json.dumps(payload),
             time.time() if now is None else now),
        )  # fmt: skip
        self.db.commit()


def scope_of(snapshot: str, config_path: Path) -> str:
    """Snapshot name + hash of the app config: a new corpus or setup never reuses old answers."""
    return f"{snapshot}:{hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]}"


def bge_encoder(embedder):
    """Plain bge-small sentence vectors (no retrieval query prefix), unit length."""

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(embedder.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    return encode
