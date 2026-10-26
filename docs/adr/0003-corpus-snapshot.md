# ADR 0003: Corpus = 40 US semiconductor filers, 10-K + 10-Q filed in calendar 2025

- Status: Proposed (awaiting sign-off: the ticker list is yours to change)
- Date: 2026-10-26

## Context
The brief asks for one sector, 30–50 companies, frozen as a snapshot. Multi-hop questions need entities that cross company boundaries (suppliers, customers, competitors). Corpus-level questions need recurring themes across many filers.

## Decision
- **Sector:** semiconductors plus semiconductor equipment and materials (`configs/corpus.yaml`, 40 tickers). Designers, IDMs and equipment makers name each other, and the same foundries and suppliers (TSMC, Samsung, ASML, etc.), as customers, suppliers and competitors. That is good multi-hop material. Export controls, China exposure, cyclicality and supply concentration recur across risk factors, which is good corpus-level material.
- **Only US domestic filers (10-K/10-Q).** Foreign private issuers (TSMC, ASML) file 20-F, which has a different structure. They appear as graph entities, not as documents.
- **Window:** forms 10-K and 10-Q *filed* 2025-01-01 to 2025-12-31, with amendments excluded. Expect about one 10-K and three 10-Qs per company, roughly 160 documents.
- **Snapshot identity:** the committed manifest `data/snapshots/semis-2025.manifest.jsonl` with per-document sha256, and a corpus checksum over it. The Phase 7 drift check uses a second snapshot (e.g. `semis-2026`).

## Consequences
- Fiscal years differ (NVDA's ends in January, MU's in August), so "2025" means filing date, not fiscal year. Golden-set questions must name the filing ("NVIDIA's 10-K filed Feb 2025").
- Rate limit 5 req/s against SEC's 10. About 200 requests in total, all cached on disk.
