# ADR 0011: Semantic cache keyed by meaning AND by the companies and periods asked about

Status: Accepted (owner sign-off 2026-10-30)
Date: 2026-10-29

## Context
On one laptop GPU a fresh answer takes seconds, and a busy queue makes it slower. Reworded repeats ("What did NVIDIA say about export controls?" / "NVIDIA on export controls") should be answered from memory. But "NVIDIA 2024" and "AMD 2024", or "2024" and "2025", look almost identical to an embedding and must never share an answer.

## Decision
- **Lookup:** SQLite plus bge-small embeddings, with cosine similarity of at least 0.935 (tuned on half of the owner-checked pairs, tested on the other half: `results/a8_cache.json`).
- **The key must also match exactly:** the companies the linker finds plus the period words. The scope is the snapshot, config and prompt hash, so any change to these empties the cache.
- **Only fully fact-checked answers are stored.** Questions are stored as hashes, never as text, and entries expire.
- **The cache is checked before the GPU queue,** so a repeat never waits behind new questions.

## Alternatives considered
- **Exact-text cache only.** Reworded repeats miss it.
- **Similarity alone, with no key.** The threshold would have to be so high that almost no reworded question is reused (the post-hoc analysis in `results/a8_cache.json`).

## Consequences
- Hit rate on real traffic depends on how often people repeat questions. The measured rate is for owner-made rewordings.
- A wrong reuse would show a confident wrong answer. The test half had none, but the upper bound is reported because the sample is small.
