"""SQLite-backed disk cache for LLM calls.

Why SQLite rather than one JSON file per call: a single file is easy to ship as a
CI artifact, writes are atomic, and WAL mode lets parallel Researcher calls write
without corrupting anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from plexus.llm.types import LLMRequest, LLMResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key        TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    request    TEXT NOT NULL,
    response   TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class LLMCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, request: LLMRequest) -> LLMResponse | None:
        key = request.cache_key()
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        resp = LLMResponse.model_validate_json(row[0])
        return resp.model_copy(update={"cached": True, "cache_key": key})

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        key = request.cache_key()
        stored = response.model_copy(update={"cached": False, "cache_key": key})
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    request.provider,
                    request.model,
                    json.dumps(request.model_dump(mode="json"), sort_keys=True),
                    stored.model_dump_json(),
                    time.time(),
                ),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
