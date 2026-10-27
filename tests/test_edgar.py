import json

import httpx
import pytest

from plexus.ingest import edgar

TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 2488, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
}


def _submissions(forms_dates):
    return {
        "filings": {
            "recent": {
                "form": [f for f, _ in forms_dates],
                "filingDate": [d for _, d in forms_dates],
                "reportDate": ["2025-01-26"] * len(forms_dates),
                "accessionNumber": [f"0001045810-25-00{i:04d}" for i in range(len(forms_dates))],
                "primaryDocument": [f"doc{i}.htm" for i in range(len(forms_dates))],
            }
        }
    }


SUBS = {
    1045810: _submissions(
        [
            ("10-K", "2025-02-26"),
            ("10-Q", "2025-05-28"),
            ("10-K/A", "2025-03-01"),
            ("8-K", "2025-04-01"),
            ("10-Q", "2024-11-20"),
        ]
    ),
    2488: _submissions([("10-Q", "2025-08-05")]),
}


@pytest.fixture
def fake_sec():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url == edgar.TICKERS_URL:
            return httpx.Response(200, json=TICKERS)
        for cik, body in SUBS.items():
            if url == edgar.SUBMISSIONS_URL.format(cik=cik):
                return httpx.Response(200, json=body)
        if "/Archives/" in url:
            return httpx.Response(200, content=f"<html>{url}</html>".encode())
        return httpx.Response(404)

    return seen, handler


def _client(tmp_path, handler, ua="Jane Doe jane@example.com"):
    http = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": ua})
    return edgar.EdgarClient(ua, tmp_path / "http", max_rps=1000, http=http)


CFG = edgar.SnapshotConfig(
    name="test-snap",
    tickers=["NVDA", "AMD", "NOPE"],
    forms=["10-K", "10-Q"],
    filed_from="2025-01-01",
    filed_to="2025-12-31",
)


def test_user_agent_is_required(tmp_path):
    with pytest.raises(edgar.EdgarError, match="SEC_USER_AGENT"):
        edgar.EdgarClient("", tmp_path)
    with pytest.raises(edgar.EdgarError):
        edgar.EdgarClient("no email here", tmp_path)


def test_snapshot_filters_forms_and_dates_and_writes_manifest(tmp_path, fake_sec):
    seen, handler = fake_sec
    summary = edgar.build_snapshot(
        CFG, _client(tmp_path, handler), tmp_path / "raw", tmp_path / "snaps"
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "snaps/test-snap.manifest.jsonl").read_text().splitlines()
    ]
    ids = sorted(r["doc_id"] for r in rows)
    # 10-K/A, 8-K and the 2024 10-Q are excluded
    assert ids == ["AMD_10Q_2025-08-05", "NVDA_10K_2025-02-26", "NVDA_10Q_2025-05-28"]
    assert summary["missing_tickers"] == ["NOPE"]
    assert summary["by_form"] == {"10-K": 1, "10-Q": 2}
    assert all((tmp_path / "raw" / r["path"]).exists() for r in rows)
    nvda_k = next(r for r in rows if r["doc_id"] == "NVDA_10K_2025-02-26")
    assert nvda_k["url"].endswith("/1045810/000104581025000000/doc0.htm")
    assert all(r.headers["User-Agent"] == "Jane Doe jane@example.com" for r in seen)


def test_rerun_is_served_from_cache_with_same_checksum(tmp_path, fake_sec):
    seen, handler = fake_sec
    first = edgar.build_snapshot(
        CFG, _client(tmp_path, handler), tmp_path / "raw", tmp_path / "snaps"
    )
    n = len(seen)
    second_client = _client(tmp_path, handler)
    second = edgar.build_snapshot(CFG, second_client, tmp_path / "raw", tmp_path / "snaps")
    assert len(seen) == n and second_client.requests_made == 0
    assert first["corpus_checksum"] == second["corpus_checksum"]


def test_checksum_is_order_independent():
    a = [{"doc_id": "x", "sha256": "1"}, {"doc_id": "y", "sha256": "2"}]
    assert edgar.corpus_checksum(a) == edgar.corpus_checksum(a[::-1])
    assert edgar.corpus_checksum(a) != edgar.corpus_checksum([a[0]])


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, content=b"ok")

    monkeypatch.setattr(edgar.EdgarClient._fetch.retry, "wait", lambda *a, **k: 0)
    assert _client(tmp_path, flaky).get("https://www.sec.gov/x") == b"ok"
    assert calls["n"] == 3


def test_corpus_config_loads():
    from plexus.settings import REPO_ROOT

    cfg = edgar.SnapshotConfig.from_yaml(REPO_ROOT / "configs/corpus.yaml")
    assert 30 <= len(cfg.tickers) <= 50
    assert len(set(cfg.tickers)) == len(cfg.tickers)
    assert cfg.max_requests_per_second <= 10


def test_missing_document_is_recorded_not_fatal_and_duplicate_ids_are_split(tmp_path):
    subs = _submissions([("10-Q", "2025-05-28"), ("10-Q", "2025-05-28"), ("10-K", "2025-02-26")])

    def handler(request):
        url = str(request.url)
        if url == edgar.TICKERS_URL:
            return httpx.Response(200, json={"0": TICKERS["0"]})
        if url == edgar.SUBMISSIONS_URL.format(cik=1045810):
            return httpx.Response(200, json=subs)
        if url.endswith("doc2.htm"):
            return httpx.Response(404)
        return httpx.Response(200, content=url.encode())

    cfg = edgar.SnapshotConfig("s", ["NVDA"], ["10-K", "10-Q"], "2025-01-01", "2025-12-31")
    summary = edgar.build_snapshot(
        cfg, _client(tmp_path, handler), tmp_path / "raw", tmp_path / "m"
    )
    rows = [json.loads(x) for x in (tmp_path / "m/s.manifest.jsonl").read_text().splitlines()]
    ids = sorted(r["doc_id"] for r in rows)
    assert len(ids) == 2 and len(set(ids)) == 2 and ids[0].startswith("NVDA_10Q_2025-05-28")
    assert [d["doc_id"] for d in summary["failed_documents"]] == ["NVDA_10K_2025-02-26"]


def test_unquoted_yaml_boolean_ticker_is_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "name: x\nforms: [10-K]\nfiled_from: '2025-01-01'\nfiled_to: '2025-12-31'\ntickers: [NVDA, ON]\n"
    )
    with pytest.raises(ValueError, match="quote these"):
        edgar.SnapshotConfig.from_yaml(cfg)
