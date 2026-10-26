"""Process-wide settings read from the environment (and `.env` if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    llm_cache_path: Path
    llm_offline: bool
    enable_claude_sdk: bool
    ollama_api_base: str
    data_dir: Path

    @classmethod
    def load(cls, env_file: Path | None = None) -> Settings:
        load_dotenv(env_file or REPO_ROOT / ".env", override=False)
        cache = Path(os.getenv("PLEXUS_LLM_CACHE", "data/cache/llm.sqlite"))
        if not cache.is_absolute():
            cache = REPO_ROOT / cache
        return cls(
            sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
            llm_cache_path=cache,
            llm_offline=_flag("PLEXUS_LLM_OFFLINE"),
            enable_claude_sdk=_flag("PLEXUS_ENABLE_CLAUDE_SDK"),
            ollama_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            data_dir=REPO_ROOT / "data",
        )
