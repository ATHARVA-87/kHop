# ADR 0006: Graph store (LadybugDB instead of Kuzu)
- Status: Accepted by the user on 2026-10-27 (chosen over pinned Kuzu and over SQLite + igraph).
- Date: 2026-10-27

## Context
The brief (section 7) names Kuzu as the embedded graph store. The Kuzu GitHub repository was archived on 2025-10-10. v0.11.3 is the last release, so there will be no bug or security fixes. Building a 2026 project on an archived dependency is a maintenance and interview liability.

## Options
1. **Kuzu 0.11.3, pinned.** It is what the brief says and works today, but it is archived.
2. **LadybugDB**, the community fork of Kuzu (same engine, Cypher, embedded). It is actively released: 0.22.0 came out on 2026-10-25.
3. **No graph DB.** Entities and edges in SQLite, loaded into igraph in memory. The graph is small (tens of thousands of nodes), so this would work, but it gives no graph-database or Cypher experience.

## Decision
Option 2, `ladybug==0.21.2` (released 2026-10-01), the newest stable release at least two weeks old. Graph access goes through a small `GraphStore` interface, so the engine can be swapped (back to option 3 in an afternoon) if the young fork misbehaves. Communities use `igraph` + `leidenalg` (Leiden).

## Supply-chain check (user rule: security first)
- The PyPI name `ladybug` was first used in 2014 by an unrelated project (versions 0.0.x). LadybugDB took it over in 2026. The current releases link to `github.com/LadybugDB/ladybug-python` and `ladybugdb.com`, which matches the project.
- The older wheel names `real_ladybug` / `real-ladybug` stop at 0.15.3. `lbug` does not exist on PyPI.
- Pinned to an exact version and recorded in `uv.lock` with hashes.

## Consequences
- The extra is `graph` (`pip install -e .[graph]`). CI unit tests that need it skip when it is not installed.
- Re-check LadybugDB's release health before Phase 7 (demo Space).
