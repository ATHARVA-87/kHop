"""Dependency vulnerability gate: `make security` (docs/security.md).

Reads pip-audit's JSON for the locked dependencies (uv.lock) and fails on any known
vulnerability that is not an accepted, documented risk. Accepted risks are listed here with
the reason and the mitigation, so a new advisory can never pass silently.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# package -> (reason, mitigation, review-by date or None)
ACCEPTED = {
    "torch": (
        "pinned 2.5.1 by project rule (CUDA setup, ADR 0002); fixes need >= 2.6",
        "only pinned BAAI models (HF commit revision), safetensors weights, no torch.load / "
        "torch.jit / torch.compile on untrusted input",
        None,
    ),
    "transformers": (
        "fixes need 5.x, which needs torch >= 2.6",
        "no trust_remote_code, no Trainer, no checkpoint conversion; pinned safetensors models",
        None,
    ),
    "filelock": (
        "12 of 13 fixes need 4.2.1 (released 2026-10-17, under the 2-week supply-chain rule)",
        "local lock-file races need another user on the same machine (single-user laptop)",
        date(2026, 10, 31),  # upgrade to filelock 4.2.1 on or after this date
    ),
}


def main(path: str) -> int:
    deps = json.loads(Path(path).read_text())["dependencies"]
    bad, accepted, overdue = [], [], []
    for d in deps:
        if not d.get("vulns"):
            continue
        ids = [v["id"] for v in d["vulns"]]
        acc = ACCEPTED.get(d["name"].lower())
        if acc is None:
            bad.append(f"{d['name']} {d['version']}: {', '.join(ids)}")
        elif acc[2] and date.today() >= acc[2]:
            overdue.append(f"{d['name']} {d['version']}: review date {acc[2]} passed")
        else:
            accepted.append(f"{d['name']} {d['version']}: {len(ids)} advisories (accepted)")
    for line in accepted:
        print("ACCEPTED", line)
    for line in overdue + bad:
        print("FAIL", line)
    print(f"{len(deps)} packages audited, {len(bad)} new vulnerable, {len(overdue)} overdue")
    return 1 if bad or overdue else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
