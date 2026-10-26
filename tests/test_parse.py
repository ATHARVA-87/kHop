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
