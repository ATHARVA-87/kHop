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


def _structured_doc():
    from plexus.ingest.parse import ParsedDoc, Section

    business = "Item 1. Business\nWe design chips.\nWe sell to OEMs. TSMC makes our wafers.\n"
    long_par = " ".join(f"Sentence number {i} is here." for i in range(60))
    table = " ".join(f"{i},{i * 7}" for i in range(400))  # flattened table: one huge "word" run
    risks = f"Item 1A. Risk Factors\n{long_par}\n{table}\nShort closing paragraph.\n"
    text = business + risks
    cut = len(business)
    return ParsedDoc("D", text, [Section("1", "business", "", 0, cut),
                                 Section("1A", "risk_factors", "", cut, len(text))])  # fmt: skip


def test_structured_chunks_respect_sections_and_limits():
    from plexus.ingest.chunk import structured_chunks

    doc = _structured_doc()
    chunks = structured_chunks(doc, child_words=40, child_chars=300, parent_words=120)
    cut = doc.sections[1].start
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
        assert len(c.text.split()) <= 40 and len(c.text) <= 300
        assert (c.end <= cut) or (c.start >= cut)  # never crosses a section boundary
        assert c.parent_start <= c.start and c.end <= c.parent_end
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    # the small business section packs its paragraphs into one child
    assert [c.section for c in chunks].count("business") == 1
    # every non-space character of the document is in exactly one child
    covered = sum(len("".join(c.text.split())) for c in chunks)
    assert covered == len("".join(doc.text.split()))


def test_structured_parents_group_children_within_a_section():
    from plexus.ingest.chunk import structured_chunks

    chunks = structured_chunks(_structured_doc(), child_words=40, child_chars=300, parent_words=120)
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c.parent_id, []).append(c)
    assert len(by_parent) < len(chunks)
    for kids in by_parent.values():
        assert len({k.section for k in kids}) == 1
        assert sum(len(k.text.split()) for k in kids) <= 120 or len(kids) == 1


def test_index_key_unchanged_for_fixed_and_distinct_for_structured():
    from plexus.config import ExperimentConfig

    a0 = ExperimentConfig(name="a0")
    assert "__fixed-300-50__" in a0.index_key
    a1 = ExperimentConfig.model_validate({"name": "a1", "chunker": {"type": "structured"}})
    assert "__structured-200-1400-800__" in a1.index_key
