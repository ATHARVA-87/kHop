"""Chunkers. A0 uses `fixed_chunks` (structure-blind); A1 adds structure-aware parent-child chunks."""

from __future__ import annotations

import itertools
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
    # A1 only: the larger passage this chunk belongs to (given to the writer as context)
    parent_id: str = ""
    parent_start: int = -1
    parent_end: int = -1

    def to_dict(self) -> dict:
        return asdict(self)


_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"“•\-])")


def _units(text: str, start: int, end: int, max_words: int, max_chars: int):
    """Spans no larger than the limits, cut at the largest structure that fits:
    paragraph (newline) > sentence > word window."""
    pos = start
    for line in text[start:end].split("\n"):
        ls, le = pos, pos + len(line)
        pos = le + 1
        stripped = line.strip()
        if not stripped:
            continue
        ls += len(line) - len(line.lstrip())
        le = ls + len(stripped)
        if len(stripped) <= max_chars and len(stripped.split()) <= max_words:
            yield ls, le
            continue
        # long paragraph: sentence pieces
        cuts = [ls] + [ls + m.end() for m in _SENTENCE_END.finditer(stripped)] + [le]
        for a, b in itertools.pairwise(cuts):
            sent = text[a:b].strip()
            if not sent:
                continue
            a = a + text[a:b].index(sent[0])
            b = a + len(sent)
            if len(sent) <= max_chars and len(sent.split()) <= max_words:
                yield a, b
                continue
            # very long sentence (tables flattened to one line): word windows
            words = [(a + m.start(), a + m.end()) for m in _WORD.finditer(sent)]
            i = 0
            while i < len(words):
                j = i
                while (
                    j < len(words) and j - i < max_words and words[j][1] - words[i][0] <= max_chars
                ):
                    j += 1
                j = max(j, i + 1)  # a single over-long token still makes progress
                yield words[i][0], words[j - 1][1]
                i = j


def _pack(text: str, units, max_words: int, max_chars: int) -> list[tuple[int, int]]:
    """Greedily merge consecutive units while the merged span stays within the limits."""
    out: list[tuple[int, int]] = []
    cur_s = cur_e = None
    cur_w = 0
    for s, e in units:
        w = len(text[s:e].split())
        if cur_s is not None and cur_w + w <= max_words and e - cur_s <= max_chars:
            cur_e, cur_w = e, cur_w + w
            continue
        if cur_s is not None:
            out.append((cur_s, cur_e))
        cur_s, cur_e, cur_w = s, e, w
    if cur_s is not None:
        out.append((cur_s, cur_e))
    return out


def structured_chunks(
    doc: ParsedDoc, child_words: int = 200, child_chars: int = 1400, parent_words: int = 800
) -> list[Chunk]:
    """A1: section-bounded, paragraph-aligned child chunks, grouped into parent passages.

    Children never cross a section boundary and are cut at paragraph, then sentence, then
    word boundaries. `child_chars` keeps table-heavy text (about 3 chars per token) inside
    bge's 512-token window, which 5.1% of A0's 300-word chunks exceeded. Parents are runs of
    consecutive children in the same section, up to `parent_words`; children are indexed,
    parents are what a writer would read.
    """
    text = doc.text
    sections = doc.sections or []
    chunks: list[Chunk] = []
    n_child = n_parent = 0
    for sec in sections:
        units = _units(text, sec.start, sec.end, child_words, child_chars)
        children = _pack(text, units, child_words, child_chars)
        # parents: consecutive children of this section, up to parent_words
        groups: list[list[tuple[int, int]]] = []
        group_words = 0
        for s, e in children:
            w = len(text[s:e].split())
            if not groups or group_words + w > parent_words:
                groups.append([])
                group_words = 0
            groups[-1].append((s, e))
            group_words += w
        for group in groups:
            pid = f"{doc.doc_id}#p{n_parent:04d}"
            ps, pe = group[0][0], group[-1][1]
            for s, e in group:
                cid = f"{doc.doc_id}#s{n_child:05d}"
                chunks.append(Chunk(cid, doc.doc_id, text[s:e], s, e, sec.name, pid, ps, pe))
                n_child += 1
            n_parent += 1
    return chunks


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
