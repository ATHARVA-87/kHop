import pytest

from plexus.ingest.chunk import fixed_chunks
from plexus.ingest.parse import ParsedDoc, Section


def _doc(n_words: int) -> ParsedDoc:
    text = " ".join(f"w{i}" for i in range(n_words))
    half = text.index(f"w{n_words // 2} ") if n_words > 1 else 0
    return ParsedDoc(
        "D",
        text,
        [Section("1", "business", "", 0, half), Section("1A", "risk_factors", "", half, len(text))],
    )


def test_windows_cover_every_word_with_overlap():
    doc = _doc(1000)
    chunks = fixed_chunks(doc, size_words=300, overlap_words=50)
    covered = set()
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
        covered.update(c.text.split())
    assert covered == set(doc.text.split())
    assert chunks[1].text.split()[:50] == chunks[0].text.split()[-50:]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_short_doc_gives_one_chunk_and_section_metadata():
    doc = _doc(10)
    [only] = fixed_chunks(doc)
    assert only.text == doc.text and only.section == "business"
    big = fixed_chunks(_doc(1000), 300, 50)
    assert big[0].section == "business" and big[-1].section == "risk_factors"


def test_no_tiny_trailing_chunk_made_only_of_overlap():
    chunks = fixed_chunks(_doc(550), 300, 50)  # 0-299, 250-549 -> exactly 2
    assert len(chunks) == 2


def test_bad_params():
    with pytest.raises(ValueError):
        fixed_chunks(_doc(10), 100, 100)
