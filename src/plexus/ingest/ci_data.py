"""Fetch the processed filings for the CI retrieval gate.

    python -m plexus.ingest.ci_data            # -> data/processed/semis-2025/docs.jsonl

GitHub's CI has no copy of the corpus (data/ is gitignored and rebuilding it from EDGAR needs
a User-Agent and several minutes). The parser output is published once as a GitHub Release
asset; this downloads it and checks both checksums recorded in
data/snapshots/<snapshot>.ci_data.json, so CI scores exactly the corpus the baselines used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

from plexus.settings import Settings


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(snapshot: str, data_dir: Path, force: bool = False, what: str = "data") -> Path:
    """what="data": processed filings; what="graph": raw graph extractions (for A4/A5)."""
    import httpx

    spec = json.loads((data_dir / "snapshots" / f"{snapshot}.ci_{what}.json").read_text())
    out = (
        (data_dir / spec["target"])
        if "target" in spec
        else data_dir / "processed" / snapshot / "docs.jsonl"
    )
    if out.exists() and not force and _sha256(out.read_bytes()) == spec["sha256_jsonl"]:
        print(f"{out} already present and verified")
        return out
    resp = httpx.get(spec["url"], follow_redirects=True, timeout=120)
    resp.raise_for_status()
    gz = resp.content
    if _sha256(gz) != spec["sha256_gz"]:
        raise RuntimeError(f"checksum mismatch for {spec['url']}")
    raw = gzip.decompress(gz)
    if _sha256(raw) != spec["sha256_jsonl"]:
        raise RuntimeError(f"decompressed {out.name} checksum mismatch")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"wrote {out} ({len(raw):,} bytes, verified)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="semis-2025")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--graph", action="store_true", help="fetch the graph extractions instead")
    args = ap.parse_args(argv)
    fetch(args.snapshot, Settings.load().data_dir, args.force, "graph" if args.graph else "data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
