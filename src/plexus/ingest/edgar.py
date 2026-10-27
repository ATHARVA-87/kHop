"""Download a frozen, checksummed EDGAR snapshot.

    python -m plexus.ingest.edgar --config configs/corpus.yaml

- Rate-limited (default 5 req/s; SEC's limit is 10) and every response is cached on
  disk, so the run is resumable and a rerun makes zero requests.
- Writes the primary HTML document of each filing to data/raw/<snapshot>/<ticker>/ and
  a manifest (committed) to data/snapshots/<snapshot>.manifest.jsonl.
- The corpus checksum is sha256 over the sorted (doc_id, sha256) lines of the manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import yaml
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from plexus.settings import REPO_ROOT, Settings

log = logging.getLogger("plexus.ingest.edgar")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


class EdgarError(RuntimeError):
    pass


@dataclass(frozen=True)
class SnapshotConfig:
    name: str
    tickers: list[str]
    forms: list[str]
    filed_from: str
    filed_to: str
    max_requests_per_second: float = 5.0
    sector: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> SnapshotConfig:
        raw = yaml.safe_load(path.read_text())
        # YAML 1.1 reads bare ON/OFF/YES/NO as booleans: ticker ON (onsemi) became True.
        bad = [t for t in raw.get("tickers", []) if not isinstance(t, str)]
        if bad:
            raise ValueError(f"{path}: tickers must be strings; quote these: {bad}")
        return cls(**raw)


@dataclass(frozen=True)
class Filing:
    doc_id: str
    ticker: str
    cik: int
    company: str
    form: str
    filing_date: str
    report_date: str
    accession: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document
        )


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    # SEC answers rate-limit violations with 403 (and sometimes 429)
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
        403,
        429,
        500,
        502,
        503,
        504,
    }


class EdgarClient:
    """HTTP client for EDGAR with a disk cache and a simple rate limiter."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        max_rps: float = 5.0,
        http: httpx.Client | None = None,
    ):
        if not user_agent or "@" not in user_agent:
            raise EdgarError(
                "SEC_USER_AGENT must contain your name and email (SEC fair-access policy). "
                'Set it in .env, e.g. SEC_USER_AGENT="Jane Doe jane@example.com"'
            )
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / max_rps
        self._last = 0.0
        self._lock = threading.Lock()
        self.http = http or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60,
            follow_redirects=True,
        )
        self.requests_made = 0

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / hashlib.sha1(url.encode()).hexdigest()

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _fetch(self, url: str) -> bytes:
        self._throttle()
        self.requests_made += 1
        resp = self.http.get(url)
        resp.raise_for_status()
        return resp.content

    def get(self, url: str) -> bytes:
        path = self._cache_path(url)
        if path.exists():
            return path.read_bytes()
        body = self._fetch(url)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache file
        return body

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))


def resolve_ciks(client: EdgarClient, tickers: list[str]) -> dict[str, tuple[int, str]]:
    """ticker -> (cik, company title). Unknown tickers are left out (and logged)."""
    data = client.get_json(TICKERS_URL)
    table = {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}
    out = {}
    for t in tickers:
        if t.upper() in table:
            out[t.upper()] = table[t.upper()]
        else:
            log.warning("ticker %s not found in SEC ticker table; skipping", t)
    return out


def list_filings(
    client: EdgarClient, ticker: str, cik: int, company: str, cfg: SnapshotConfig
) -> list[Filing]:
    sub = client.get_json(SUBMISSIONS_URL.format(cik=cik))
    recent = sub["filings"]["recent"]
    filings = []
    for i, form in enumerate(recent["form"]):
        filed = recent["filingDate"][i]
        if form not in cfg.forms or not (cfg.filed_from <= filed <= cfg.filed_to):
            continue
        doc = recent["primaryDocument"][i]
        if not doc:
            continue
        filings.append(
            Filing(
                doc_id=f"{ticker}_{form.replace('-', '')}_{filed}",
                ticker=ticker,
                cik=cik,
                company=company,
                form=form,
                filing_date=filed,
                report_date=recent["reportDate"][i] or "",
                accession=recent["accessionNumber"][i],
                primary_document=doc,
            )
        )
    # The "recent" block holds the last ~1000 filings; older pages are not needed for
    # a single-year snapshot. Warn if the window might reach past it.
    if recent["filingDate"] and min(recent["filingDate"]) > cfg.filed_from:
        log.warning("%s: 'recent' filings may not cover %s", ticker, cfg.filed_from)
    return sorted(filings, key=lambda f: f.filing_date)


def corpus_checksum(manifest_rows: list[dict]) -> str:
    lines = sorted(f"{r['doc_id']}\t{r['sha256']}" for r in manifest_rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def build_snapshot(
    cfg: SnapshotConfig, client: EdgarClient, raw_root: Path, manifest_dir: Path
) -> dict:
    out_dir = raw_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ciks = resolve_ciks(client, cfg.tickers)
    rows: list[dict] = []
    seen_ids: set[str] = set()
    failed_docs: list[dict] = []
    missing = [t for t in cfg.tickers if t.upper() not in ciks]
    for ticker, (cik, company) in ciks.items():
        try:
            filings = list_filings(client, ticker, cik, company, cfg)
        except httpx.HTTPError as exc:
            log.error("%s: submissions failed: %s", ticker, exc)
            missing.append(ticker)
            continue
        if not filings:
            log.warning("%s: no %s filings in window", ticker, cfg.forms)
        for f in filings:
            if f.doc_id in seen_ids:  # two filings of one form on the same day
                f = dataclasses.replace(f, doc_id=f"{f.doc_id}_{f.accession[-6:]}")
            seen_ids.add(f.doc_id)
            try:
                body = client.get(f.url)
            except httpx.HTTPError as exc:  # one bad document must not abort the snapshot
                log.error("%s: download failed: %s", f.doc_id, exc)
                failed_docs.append({"doc_id": f.doc_id, "url": f.url, "error": str(exc)[:200]})
                continue
            dest = out_dir / ticker / f"{f.doc_id}.htm"
            dest.parent.mkdir(parents=True, exist_ok=True)
            # always write: the file on disk must match the sha256 in the manifest
            dest.write_bytes(body)
            rows.append(
                {
                    **asdict(f),
                    "url": f.url,
                    "path": str(dest.relative_to(raw_root)),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "bytes": len(body),
                }
            )
        log.info("%s: %d filings", ticker, len(filings))

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{cfg.name}.manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    summary = {
        "snapshot": cfg.name,
        "documents": len(rows),
        "companies": len({r["ticker"] for r in rows}),
        "missing_tickers": sorted(set(missing)),
        "failed_documents": failed_docs,
        "by_form": {f: sum(r["form"] == f for r in rows) for f in cfg.forms},
        "total_bytes": sum(r["bytes"] for r in rows),
        "corpus_checksum": corpus_checksum(rows),
        "requests_made": client.requests_made,
    }
    (manifest_dir / f"{cfg.name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a frozen EDGAR snapshot")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/corpus.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = Settings.load()
    cfg = SnapshotConfig.from_yaml(args.config)
    try:
        client = EdgarClient(
            settings.sec_user_agent,
            cache_dir=settings.data_dir / "cache" / "edgar_http",
            max_rps=cfg.max_requests_per_second,
        )
    except EdgarError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = build_snapshot(
        cfg,
        client,
        raw_root=settings.data_dir / "raw",
        manifest_dir=settings.data_dir / "snapshots",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
