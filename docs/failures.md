# Failures and negative results

Things that did not work, or did not help as much as expected. Negative results are part of the deliverable.

_None recorded yet._

## Parser: Intel filings are unstructured (2026-10-27)
Intel's 10-K and 10-Qs use their own section headings ("Risk Factors and Other Key Information", repeated as page headers) and put Item numbers only in a cross-reference index at the end. The Item-heading parser cannot map them. A title-based fallback first labelled 98% of each Intel filing as `risk_factors`, which is wrong. The parser now marks a filing `unknown` when one section would cover more than 80% of it.
- Impact: 4 of 156 filings (2.6%). Their text is fully indexed and retrievable (A0 ignores sections), but they are excluded from section-based golden-candidate sampling and get no section metadata (relevant for A1).
- Possible fix later: parse Intel's cross-reference index to map its own headings to Items.
