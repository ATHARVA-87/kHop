# AI review log

Mistakes in AI-generated code or data that tests, evals or human review caught. One entry per catch.

| Date | Area | What the AI got wrong | How it was caught | Fix |
| --- | --- | --- | --- | --- |
| 2026-10-26 | env setup | Installed `virtualenv` and then force-pinned `packaging==25.0` over a conda-installed `packaging`, which left a half-removed `packaging-25.0.dist-info` beside a complete 26.3 one. `importlib.metadata.version("packaging")` returned `None`. | `tests/test_dense.py` (transformers import raised "Unable to compare versions for packaging") | Removed the broken dist-info, reinstalled `packaging==25.0` cleanly, `pip check` clean |
| 2026-10-26 | dependencies | The dependency dry run kept torch at 2.5.1 but pulled transformers 5.x, which needs torch>=2.6 (declared only under an optional extra, so pip did not flag it). sentence-transformers failed at import with `NameError: nn`. | `tests/test_dense.py` (ml-marked tests) | Pinned `transformers>=4.41,<5` and `sentence-transformers>=3,<6` in the `ml` extra and re-locked |
| 2026-10-26 | golden candidates | Candidate passages were fixed windows labelled by the section at the window's *start*, so a passage straddling Business → Risk Factors was labelled "business" and the global (risk-theme) sampler found no risk passages. | `tests/test_candidates.py` (global sampler returned nothing) | Passages are now cut inside each section (`section_windows`). The same weakness applies to A0's chunk metadata, which motivates A1 structure-aware chunking |
