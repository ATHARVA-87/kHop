from plexus.extract.quotes import align_quote, best_score, normalise

PASSAGE = (
    "We rely on Taiwan Semiconductor Manufacturing Company Limited (“TSMC”) to produce our "
    "wafers. Our customers include Apple, Dell, HP, Lenovo and Samsung, among others. "
    "Revenue from Apple was 12% of total revenue in fiscal 2024."
)


def test_normalise_keeps_a_map_to_original_offsets():
    norm, idx = normalise("A (“TSMC”) b")
    assert norm == "a tsmc b"
    assert "A (“TSMC”) b"[idx[norm.index("tsmc")]] == "T"


def test_exact_and_punctuation_variants_align_to_the_passage_span():
    a = align_quote(
        'We rely on Taiwan Semiconductor Manufacturing Company Limited ("TSMC")', PASSAGE
    )
    assert a is not None and a.score == 100.0
    assert (
        PASSAGE[a.start : a.end].startswith("We rely on Taiwan")
        and PASSAGE[a.end - 4 : a.end] == "TSMC"
    )


def test_ellipsis_pieces_must_appear_in_order():
    ok = align_quote("Our customers include Apple ... Lenovo and Samsung, among others", PASSAGE)
    assert ok is not None and ok.pieces == 2
    assert (
        PASSAGE[ok.start : ok.end]
        == "Our customers include Apple, Dell, HP, Lenovo and Samsung, among others"
    )
    swapped = align_quote(
        "Lenovo and Samsung, among others ... Our customers include Apple", PASSAGE
    )
    assert swapped is None


def test_small_wording_slips_pass_but_invented_text_fails():
    assert (
        align_quote("We relies on Taiwan Semiconductor Manufacturing Company Limited", PASSAGE)
        is not None
    )
    assert (
        align_quote("We compete aggressively with Intel and AMD in data center GPUs", PASSAGE)
        is None
    )
    assert (
        best_score("We compete aggressively with Intel and AMD in data center GPUs", PASSAGE) < 80
    )
