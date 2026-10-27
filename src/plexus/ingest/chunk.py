"""Chunkers. A0 uses `fixed_chunks` (structure-blind); A1 adds structure-aware parent-child chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from plexus.ingest.parse import ParsedDoc

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int  # char offsets into ParsedDoc.text
    end: int
    section: str  # canonical section name at the chunk start (metadata only for A0)

    def to_dict(self) -> dict:
        return asdict(self)


def fixed_chunks(doc: ParsedDoc, size_words: int = 300, overlap_words: int = 50) -> list[Chunk]:
    """Sliding window of `size_words` words with `overlap_words` overlap, ignoring sections.

    ~300 English words is ~400 BPE tokens, inside bge-small's 512-token window.
    """
    if not 0 <= overlap_words < size_words:
        raise ValueError("need 0 <= overlap_words < size_words")
    words = [(m.start(), m.end()) for m in _WORD.finditer(doc.text)]
    chunks: list[Chunk] = []
    step = size_words - overlap_words
    for n, i in enumerate(range(0, max(len(words) - overlap_words, 1), step)):
        window = words[i : i + size_words]
        if not window:
            break
        start, end = window[0][0], window[-1][1]
        sec = doc.section_at(start)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#c{n:04d}",
                doc_id=doc.doc_id,
                text=doc.text[start:end],
                start=start,
                end=end,
                section=sec.name if sec else "unknown",
            )
        )
    return chunks
