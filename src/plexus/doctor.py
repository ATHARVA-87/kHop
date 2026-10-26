"""`make doctor`: check every prerequisite and say exactly how to fix what is missing.

Each check belongs to the phase that first needs it. `--phase N` makes checks for
phases <= N required (exit code 1 if they fail); later ones are shown as notes.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

from plexus.settings import REPO_ROOT, Settings


@dataclass
class Result:
    ok: bool
    detail: str
    fix: str = ""


@dataclass
class Check:
    name: str
    phase: int
    run: Callable[[], Result]


def _run(cmd: list[str], timeout: float = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def _version(text: str) -> tuple[int, ...]:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return tuple(int(g) for g in m.groups() if g is not None) if m else (0,)


def check_python() -> Result:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Result(
        ok,
        f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
        "Use Python 3.11+: `conda activate torch` (or install from python.org)",
    )


def check_anthropic_key_unset() -> Result:
    if os.getenv("ANTHROPIC_API_KEY"):
        return Result(
            False,
            "ANTHROPIC_API_KEY is set",
            "Unset it (`unset ANTHROPIC_API_KEY`, remove from ~/.bashrc): "
            "Claude Code would bill the API instead of your Pro plan",
        )
    return Result(True, "not set (Claude Code uses your Pro plan)")


def check_tool(binary: str, fix: str, min_version: tuple[int, ...] = ()) -> Callable[[], Result]:
    def _check() -> Result:
        path = shutil.which(binary)
        if not path:
            return Result(False, "not found", fix)
        code, out = _run([binary, "--version"])
        ver = _version(out)
        if min_version and ver < min_version:
            want = ".".join(map(str, min_version))
            return Result(False, f"{out.splitlines()[0] if out else '?'} (need >= {want})", fix)
        return Result(code == 0, out.splitlines()[0] if out else path, fix)

    return _check


def check_docker_daemon() -> Result:
    if not shutil.which("docker"):
        return Result(
            False, "docker not found", "Install Docker: https://docs.docker.com/engine/install/"
        )
    code, out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code != 0:
        return Result(
            False,
            "daemon not reachable",
            "Start Docker (`sudo systemctl start docker`) and make sure your user is "
            "in the docker group (`sudo usermod -aG docker $USER`, then log out/in)",
        )
    return Result(True, f"server {out}")


def check_imports(modules: list[str], fix: str) -> Callable[[], Result]:
    def _check() -> Result:
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        if missing:
            return Result(False, "missing: " + ", ".join(missing), fix)
        return Result(True, "all importable")

    return _check


def check_torch_cuda() -> Result:
    if importlib.util.find_spec("torch") is None:
        return Result(False, "torch not installed", "pip install -e '.[ml]' (in the torch env)")
    import torch

    if torch.cuda.is_available():
        return Result(True, f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
    return Result(True, f"torch {torch.__version__}, CPU only (embedding will be slower)")


def check_env_file() -> Result:
    if (REPO_ROOT / ".env").exists():
        return Result(True, ".env present")
    return Result(False, ".env missing", "cp .env.example .env  (then fill values in an editor)")


def check_sec_user_agent() -> Result:
    ua = Settings.load().sec_user_agent
    if not ua:
        return Result(
            False,
            "SEC_USER_AGENT empty",
            'Set in .env: SEC_USER_AGENT="Your Name your.email@example.com" '
            "(SEC fair-access policy, https://www.sec.gov/os/accessing-edgar-data)",
        )
    if "@" not in ua or len(ua.split()) < 2:
        return Result(False, "SEC_USER_AGENT should contain a name and an email", "Fix it in .env")
    return Result(True, "set")


def check_git_remote() -> Result:
    code, out = _run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"])
    if code != 0:
        return Result(
            False,
            "no `origin` remote",
            "Create an empty public GitHub repo named `plexus`, then "
            "`git remote add origin <url> && git push -u origin main`",
        )
    return Result(True, out)


def check_ollama() -> Result:
    if not shutil.which("ollama"):
        return Result(
            False,
            "not installed",
            "curl -fsSL https://ollama.com/install.sh | sh  (needs sudo; then `ollama pull qwen3.5:4b`)",
        )
    import httpx

    try:
        httpx.get(Settings.load().ollama_api_base + "/api/tags", timeout=3).raise_for_status()
    except httpx.HTTPError:
        return Result(
            False, "installed but server not running", "Run `ollama serve` (or start the service)"
        )
    return Result(True, "server reachable")


def check_free_tier_keys() -> Result:
    Settings.load()
    have = [k for k in ("GEMINI_API_KEY", "GROQ_API_KEY") if os.getenv(k)]
    if not have:
        return Result(
            False,
            "neither GEMINI_API_KEY nor GROQ_API_KEY set",
            "Create a free key (https://aistudio.google.com/apikey or https://console.groq.com/keys) "
            "and paste it into .env yourself",
        )
    return Result(True, "set: " + ", ".join(have))


def check_snapshot(name: str = "semis-2025") -> Callable[[], Result]:
    def _check() -> Result:
        import json

        summary = REPO_ROOT / "data" / "snapshots" / f"{name}.summary.json"
        if not summary.exists():
            return Result(False, f"{name} not downloaded", "make ingest  (needs SEC_USER_AGENT)")
        s = json.loads(summary.read_text())
        miss = f", missing {s['missing_tickers']}" if s.get("missing_tickers") else ""
        return Result(True, f"{s['documents']} docs from {s['companies']} companies{miss}")

    return _check


def check_processed(name: str = "semis-2025") -> Result:
    if not (REPO_ROOT / "data" / "processed" / name / "docs.jsonl").exists():
        return Result(
            False, "not parsed", "make process  (then read data/snapshots/*.parse_report.json)"
        )
    return Result(True, "parsed")


def check_golden(min_verified: int = 100) -> Result:
    import json

    stats = REPO_ROOT / "data" / "golden" / "review_stats.json"
    if not (REPO_ROOT / "data" / "golden" / "review.csv").exists():
        return Result(False, "no candidates yet", "make candidates  (needs an LLM provider)")
    if not stats.exists():
        return Result(
            False,
            "candidates not reviewed",
            "Review data/golden/review.csv (docs/golden_review_guide.md), then make golden",
        )
    s = json.loads(stats.read_text())
    n = sum(s.get("verified_by_type", {}).values())
    if n < min_verified:
        return Result(
            False,
            f"{n} verified (< {min_verified})",
            "Review more rows in data/golden/review.csv, then make golden",
        )
    return Result(True, f"{n} verified, label noise {s.get('label_noise_rate')}")


CHECKS: list[Check] = [
    Check("Python >= 3.11", 0, check_python),
    Check("ANTHROPIC_API_KEY unset", 0, check_anthropic_key_unset),
    Check("git", 0, check_tool("git", "sudo apt install git")),
    Check("Node.js >= 20", 0, check_tool("node", "Install Node LTS: https://nodejs.org", (20,))),
    Check("Docker daemon", 0, check_docker_daemon),
    Check(
        "core Python deps",
        0,
        check_imports(
            ["pydantic", "yaml", "dotenv", "httpx", "tenacity", "litellm", "bs4", "lxml", "numpy"],
            "pip install -e '.[dev]'",
        ),
    ),
    Check(
        "dev tools (pytest, ruff, pre-commit)",
        0,
        check_imports(["pytest", "ruff", "pre_commit"], "pip install -e '.[dev]'"),
    ),
    Check("GitHub remote", 0, check_git_remote),
    Check(".env file", 1, check_env_file),
    Check("SEC_USER_AGENT", 1, check_sec_user_agent),
    Check(
        "ML deps (torch, sentence-transformers, qdrant)",
        1,
        check_imports(
            ["torch", "sentence_transformers", "qdrant_client"],
            "pip install -c constraints.txt -e '.[ml]' (in the torch env; keeps torch 2.5.1)",
        ),
    ),
    Check("torch / GPU", 1, check_torch_cuda),
    Check("EDGAR snapshot", 1, check_snapshot()),
    Check("Parsed snapshot", 1, check_processed),
    Check("Golden set (>=100 verified)", 1, check_golden),
    Check("Ollama", 3, check_ollama),
    # Needed in Phase 1 already: golden-set candidates and the judge use Gemini (ADR 0005).
    Check("Gemini or Groq key", 1, check_free_tier_keys),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase", type=int, default=0, help="treat checks up to this phase as required"
    )
    args = ap.parse_args(argv)

    failed_required = 0
    print(f"Plexus doctor (required up to phase {args.phase})\n")
    for c in CHECKS:
        try:
            r = c.run()
        except Exception as exc:  # a broken check must not hide the others
            r = Result(False, f"check crashed: {exc}")
        required = c.phase <= args.phase
        tag = "OK  " if r.ok else ("FAIL" if required else "TODO")
        print(f"[{tag}] {c.name:<48} {r.detail}")
        if not r.ok:
            print(f"       phase {c.phase}: {r.fix}")
            failed_required += required
    print()
    if failed_required:
        print(f"{failed_required} required check(s) failed.")
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
