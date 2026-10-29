# ADR 0012: A plain-code spelling layer that asks before it changes anything

Status: Proposed (awaiting owner sign-off)
Date: 2026-10-29

## Context
The owner found that "nvidia company", "nvidia companies" and "nvdia companies" gave different answers. A typo hid the company, so the router chose the wrong route. Small wording changes also moved the top passages. The owner asked for the question to be rewritten into the filings' own wording, with the user choosing whether to accept it.

## Decision
`src/plexus/serve/reframe.py`, with no LLM call:
1. **Spelling.** A word is a typo only if the filings, the company names and the system English word list all fail to recognise it. Candidates are corpus words within 1 edit (2 for long words) by OSA distance; the most frequent wins. Company names and, for the first word, question words are preferred. Close calls are left alone. Tickers, codes and company suffixes are never changed.
2. **Company names** become the official display name ("nvidia company" becomes "NVIDIA"). Words are never reordered.
3. **The user chooses:** "Use this version" or "Ask as I typed".
4. **No system word list, no changes:** without one the layer was measured to change correct questions, so it switches itself off.

Keep rules were written before any code (`data/golden/reframe/PROTOCOL.md`). Each version was judged on a fresh typo set it was not tuned on (`results/reframe.json`).

## Alternatives considered
- **An LLM rewrite.** It costs a call and can change meaning silently.
- **A general spell-checker.** It "fixes" finance terms and tickers it doesn't know.
- **A phrase map from everyday words to filing words.** Not built: hand-written, easy to overfit to the question set, and the expected gain was small. Recorded as a known limit.

## Consequences
- Ambiguous typos stay as typed.
- The demo image must install a word list (Debian `wamerican`).
