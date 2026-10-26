import pytest

from plexus.ingest.parse import html_to_text, parse_filing

FILLER = " ".join(["The company designs semiconductors."] * 40)

TENK = f"""
<html><head><title>x</title><style>.a{{}}</style></head><body>
<div style="display:none"><ix:header>hidden xbrl junk</ix:header></div>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>Table of Contents</p>
<table>
 <tr><td>Item 1.</td><td>Business</td><td>4</td></tr>
 <tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
 <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>40</td></tr>
</table>
<p><b>PART I</b></p>
<p><b>Item 1. Business</b></p>
<p>Our <span>GPUs</span> are made by <span>TSMC</span>. {FILLER}</p>
<p>12</p>
<p><b>Item 1A.</b></p><p><b>Risk Factors</b></p>
<p>We depend on a single foundry. {FILLER}</p>
<p><b>PART II</b></p>
<p>Item 7. Management&#8217;s Discussion and Analysis</p>
<p>Revenue grew 114%. {FILLER}</p>
<p>Item 8. Financial Statements</p>
<table><tr><td>Revenue</td><td>$</td><td>130,497</td></tr></table>
</body></html>
"""


def test_html_to_text_keeps_blocks_and_drops_hidden_content():
    text = html_to_text(TENK)
    assert "hidden xbrl junk" not in text
    assert "Our GPUs are made by TSMC." in text  # inline spans do not break the sentence
    assert "Revenue | $ | 130,497" in text  # table cells are delimited
    assert "\n12\n" not in text  # bare page numbers dropped


def test_sections_skip_table_of_contents():
    doc = parse_filing("NVDA_10K_x", TENK, "10-K")
    by_name = {s.name: s for s in doc.sections}
    assert [s.name for s in doc.sections] == [
        "cover",
        "business",
        "risk_factors",
        "mdna",
        "financial_statements",
    ]
    risk = doc.text[by_name["risk_factors"].start : by_name["risk_factors"].end]
    assert "single foundry" in risk and "TSMC" not in risk
    assert by_name["risk_factors"].title == "Risk Factors"  # title taken from next line
    assert "Revenue grew" in doc.text[by_name["mdna"].start : by_name["mdna"].end]
    # sections tile the text with no gaps or overlaps
    for a, b in zip(doc.sections, doc.sections[1:], strict=False):
        assert a.end == b.start
    assert doc.sections[-1].end == len(doc.text)


def test_section_at_offset():
    doc = parse_filing("d", TENK, "10-K")
    assert doc.section_at(doc.text.index("single foundry")).name == "risk_factors"


TENQ = f"""
<html><body>
<p>PART I. FINANCIAL INFORMATION</p>
<p>Item 1. Financial Statements</p><p>{FILLER}</p>
<p>Item 2. Management's Discussion and Analysis</p><p>Data center demand. {FILLER}</p>
<p>PART II. OTHER INFORMATION</p>
<p>Item 1. Legal Proceedings</p><p>None. {FILLER}</p>
<p>Item 1A. Risk Factors</p><p>Export controls on China. {FILLER}</p>
</body></html>
"""


def test_10q_items_are_keyed_by_part():
    doc = parse_filing("q", TENQ, "10-Q")
    keys = [s.key for s in doc.sections if s.key != "cover"]
    assert keys == ["I-1", "I-2", "II-1", "II-1A"]
    assert doc.section_at(doc.text.index("Export controls")).name == "risk_factors"
    assert doc.section_at(doc.text.index("Data center")).name == "mdna"


def test_unstructured_document_falls_back_to_one_section():
    doc = parse_filing("u", "<p>just text</p>", "10-K")
    assert len(doc.sections) == 1 and doc.sections[0].end == len(doc.text)


# ---- regressions from the Phase 1 review --------------------------------------------

BODY = " ".join(["Body text about the business and its markets."] * 30)
TOC_ROWS = [
    ("1", "Business", 4), ("1A", "Risk Factors", 12), ("1B", "Unresolved Staff Comments", 30),
    ("7", "Management's Discussion and Analysis", 40), ("16", "Form 10-K Summary", 90),
]  # fmt: skip


def _toc(with_pages=True):
    rows = "".join(
        f"<tr><td>Item {k}.</td><td>{t}</td>{f'<td>{p}</td>' if with_pages else ''}</tr>"
        for k, t, p in TOC_ROWS
    )
    return f"<table>{rows}</table>"


def _tenk(toc):
    return f"""<html><body>{toc}
<p>FORWARD-LOOKING STATEMENTS</p><p>{BODY}</p>
<p>Item 1. Business</p><p>{BODY}</p>
<p>Item 1A. Risk Factors</p><p>Supply risk. {BODY}</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>Item 7. Management's Discussion and Analysis</p><p>{BODY}</p>
<p>Item 10. Directors</p><p>Item 105 of Regulation S-K requires disclosure; see the proxy. {BODY}</p>
<p>Item 16. Form 10-K Summary</p><p>None.</p></body></html>"""


@pytest.mark.parametrize("with_pages", [True, False])
def test_toc_never_wins_and_preamble_stays_cover(with_pages):
    doc = parse_filing("d", _tenk(_toc(with_pages)), "10-K")
    names = [s.name for s in doc.sections]
    assert names == ["cover", "business", "risk_factors", "unresolved_staff_comments", "mdna",
                     "directors_officers", "summary"]  # fmt: skip
    cover = doc.sections[0]
    assert "FORWARD-LOOKING" in doc.text[cover.start : cover.end]
    summary = doc.sections[-1]
    assert doc.text[summary.start : summary.end].startswith("Item 16")
    one_b = next(s for s in doc.sections if s.key == "1B")
    assert "None." in doc.text[one_b.start : one_b.end]
    # tiling: no unlabelled gaps
    assert doc.sections[0].start == 0 and doc.sections[-1].end == len(doc.text)
    assert all(a.end == b.start for a, b in zip(doc.sections, doc.sections[1:], strict=False))


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ("Item 105 of Regulation S-K requires", None),
        ("Item 1Business", "1"),
        ("Item 9Changes in and Disagreements", "9"),
        ("ITEM 1A. RISK FACTORS", "1A"),
        ("Item 1A—Risk Factors", "1A"),
        ("Items 1 and 2. Business and Properties", "1"),
        ("Item 7A. Quantitative and Qualitative Disclosures", "7A"),
        ("item 1business", None),
    ],
)
def test_heading_regex(line, key):
    from plexus.ingest.parse import _heading_candidates

    cands = _heading_candidates([line], "10-K")
    assert (cands[0][1] if cands else None) == key


def test_10q_parts_from_titles_when_body_has_no_part_lines():
    html = f"""<html><body>
<table><tr><td>PART I</td></tr><tr><td>Item 1.</td><td>Financial Statements</td><td>3</td></tr>
<tr><td>PART II</td></tr><tr><td>Item 1.</td><td>Legal Proceedings</td><td>30</td></tr>
<tr><td>Item 6.</td><td>Exhibits</td><td>40</td></tr></table>
<p>Item 1. Financial Statements</p><p>{BODY}</p>
<p>Item 2. Management's Discussion and Analysis</p><p>{BODY}</p>
<p>Item 1. Legal Proceedings</p><p>{BODY}</p>
<p>Item 1A. Risk Factors</p><p>No material changes.</p>
<p>Item 6. Exhibits</p><p>31.1 Certification</p></body></html>"""
    doc = parse_filing("q", html, "10-Q")
    assert [s.key for s in doc.sections if s.key != "cover"] == [
        "I-1",
        "I-2",
        "II-1",
        "II-1A",
        "II-6",
    ]
    assert doc.sections[-1].name == "exhibits"
