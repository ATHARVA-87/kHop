# How to review golden-set candidates

You are the ground truth. The benchmark is only as good as this review. Aim to review **at least 100 of the ~200** candidates, and spread them across all three types.

Open `data/golden/review.csv` in LibreOffice Calc or a spreadsheet tool (keep the CSV format when saving). For each row, open the filing with the `evidence_doc_ids` value under `data/raw/semis-2025/<TICKER>/<doc_id>.htm`, or search `data/processed/semis-2025/docs.jsonl`.

## Verdict column (one of three)

| verdict | when |
| --- | --- |
| `accept` | The question is clear, answerable from the quoted evidence, the answer is correct and complete, and every quote supports it. |
| `edit` | Fixable. Put the fixed text in the `corrected_*` columns, leaving the original columns as they are. Only fill the columns you change. |
| `reject` | Not worth fixing: ambiguous, trivial, unanswerable, about stock price or investment merit, or the wrong type (e.g. a "multi-hop" question that one passage answers). |

Leave `verdict` blank for rows you did not review. Blank rows are **never** scored.

## Checklist per row

1. **Question:** names the company and filing? Could a reasonable analyst ask it? Not yes/no?
2. **Type:** `local` = one passage answers it. `multi_hop` = needs both companies' passages; check that the two `sub_questions` really split it. `global` = a sector-level question whose answer spans 3+ companies.
3. **Answer:** correct according to the filing? For `global`, does it cover every company quoted?
4. **Evidence:** each quote is copied verbatim and actually supports the answer. If a better sentence exists, put it in `corrected_evidence_quotes`.

## Formatting corrections

- Several evidence items go in one cell, separated by ` ||| ` (space, three pipes, space), with doc IDs and quotes in the same order:
  - `corrected_evidence_doc_ids`: `NVDA_10K_2025-02-26 ||| AMD_10K_2025-02-05`
  - `corrected_evidence_quotes`: `We rely on TSMC ... ||| We also depend on TSMC ...`
- Quotes must be copied **exactly** from the filing text, though case and whitespace don't matter. A quote the scorer cannot find is reported as a label error.
- `corrected_qtype`: `local`, `multi_hop` or `global`.
- Put anything worth remembering in `reviewer_notes`, e.g. "answer was outdated in the later 10-Q".

## After reviewing

```bash
make golden      # builds data/golden/golden.jsonl and review_stats.json (label-noise rate)
```

`review_stats.json` reports how many candidates the LLM got wrong (edit + reject over reviewed). That noise rate goes in the README.
