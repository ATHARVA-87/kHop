"""Fetch the SEC company list used as the entity-resolution gazetteer.

    python -m plexus.ingest.sec_reference

Writes data/reference/sec_company_tickers.json byte-for-byte as served, plus a .meta.json with
the fetch date and sha256 (checked on every load by plexus.extract.resolve). Needs
SEC_USER_AGENT in .env (SEC requires a User-Agent; it is sent, never stored).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys

import httpx

from plexus.settings import Settings

URL = "https://www.sec.gov/files/company_tickers.json"


def main() -> int:
    s = Settings.load()
    if not s.sec_user_agent:
        print("SEC_USER_AGENT is not set in .env", file=sys.stderr)
        return 1
    r = httpx.get(URL, headers={"User-Agent": s.sec_user_agent}, timeout=60)
    r.raise_for_status()
    out = s.data_dir / "reference"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sec_company_tickers.json").write_bytes(r.content)
    meta = {"source": URL, "fetched": datetime.date.today().isoformat(),
            "sha256": hashlib.sha256(r.content).hexdigest(), "entries": len(r.json())}  # fmt: skip
    (out / "sec_company_tickers.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
