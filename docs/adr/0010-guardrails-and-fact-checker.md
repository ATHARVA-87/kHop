# ADR 0010: Rules for refusals, datamarking for passages, a small fact-checker for answers

Status: Proposed (awaiting owner sign-off)
Date: 2026-10-29

## Context
The app must:
- refuse investment advice and off-topic requests;
- resist instructions hidden inside filing text;
- show when a sentence is not backed by the passage it cites.

The brief also asks for every guardrail to be measured. Results: `results/guardrails_a9_*.json` and `results/verifier_validation*.json`; labels in `data/golden/guardrails/` and `data/golden/verifier_labels.csv`, all checked by the owner.

## Decision
1. **Input gate:** regular-expression rules run before search. They cover advice, off-topic requests, rule-breaking attempts, and a 1000-character cap. A refusal is a plain message. The owner's joke reply sits behind a switch that is off.
2. **Datamarking:** retrieved text reaches the writer with a marker between words, plus one prompt rule that marked text is data, never instructions.
3. **Fact-checker:** MiniCheck-RoBERTa-Large checks each sentence that cites a passage against that passage, and the page marks sentences it does not back. It marks; it never deletes or rewrites an answer. It runs on CPU so the writer keeps the whole GPU (owner's choice).

## Alternatives considered
- **An LLM guard or LLM judge.** It costs a call per question. An LLM judge was also measured against the owner's blind labels and did not reach its pre-set bar (docs/failures.md).
- **Replacing unbacked answers with "not enough evidence".** Rejected: it would have blanked answers the owner labelled correct.
- **Flan-T5 MiniCheck.** Ran out of GPU memory. The RoBERTa version passed its pre-set bar on 60 blind pairs.

## Consequences
- Rules can be reworded around. The held-out refusal set measures this honestly, and its misses are listed in docs/failures.md.
- The fact-checker adds about two seconds per answer on CPU (`results/phase6_stages.json`).
