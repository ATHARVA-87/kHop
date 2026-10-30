"""Download the fact-checker model once and convert it to safetensors (`make minicheck`).

The Hugging Face repo ships only a pickle (`pytorch_model.bin`). Pickles can run code when
loaded, so this script loads it with `weights_only=True` (tensors only, nothing executed),
saves a safetensors copy to data/models/minicheck-roberta-large, and deletes the download.
The app only ever loads the safetensors copy (docs/security.md).

Model: lytang/MiniCheck-RoBERTa-Large (MIT licence), pinned to one commit.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from plexus.settings import REPO_ROOT

REPO = "lytang/MiniCheck-RoBERTa-Large"
REVISION = "74c8919647e61ed0f71bc177d94f10930f090068"
FILES = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]  # fmt: skip
OUT = REPO_ROOT / "data" / "models" / "minicheck-roberta-large"


def main() -> int:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    if (OUT / "model.safetensors").exists():
        print(f"already there: {OUT.relative_to(REPO_ROOT)}")
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="minicheck-"))
    try:
        for name in FILES:
            hf_hub_download(REPO, name, revision=REVISION, local_dir=tmp)
        state = torch.load(tmp / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(tmp))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or [k for k in missing if not k.endswith("position_ids")]:
            raise RuntimeError(f"weights do not match the model: {missing[:5]} {unexpected[:5]}")
        OUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUT, safe_serialization=True)
        AutoTokenizer.from_pretrained(tmp).save_pretrained(OUT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # the pickle is never kept
    print(f"saved {OUT.relative_to(REPO_ROOT)} (safetensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
