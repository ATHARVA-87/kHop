import json
from datetime import date

from plexus import audit


def _run(tmp_path, deps):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"dependencies": deps}))
    return audit.main(str(p))


def test_new_vulnerability_fails_and_accepted_passes(tmp_path, monkeypatch):
    v = [{"id": "X-1"}]
    assert _run(tmp_path, [{"name": "torch", "version": "2.5.1", "vulns": v}]) == 0
    assert _run(tmp_path, [{"name": "requests", "version": "1", "vulns": v}]) == 1
    assert _run(tmp_path, [{"name": "numpy", "version": "1", "vulns": []}]) == 0


def test_accepted_risk_expires(tmp_path, monkeypatch):
    monkeypatch.setitem(audit.ACCEPTED, "filelock", ("r", "m", date(2000, 1, 1)))
    deps = [{"name": "filelock", "version": "3", "vulns": [{"id": "X"}]}]
    assert _run(tmp_path, deps) == 1
