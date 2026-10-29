"""Verifier: is each cited sentence of an answer supported by the passages it cites?

Two parts, no LLM call (research report 2026-10-29: an LLM verifier would use free-tier quota,
and our LLM judges caught only 3 of 5 wrong answers):
1. MiniCheck-Flan-T5-Large (Tang et al. 2024, MIT licence): a 770M fact-checking model, best
   under 1B parameters on LLM-AggreFact. Scoring copied from the authors' code
   (github.com/Liyan06/MiniCheck, minicheck/inference.py): input "predict: " + passage + </s>
   + claim; support probability = softmax over the first decoder token's logits for ids
   3 (not supported) and 209 (supported); passages split into ~400-word chunks, max over chunks.
   Weights: converted once to safetensors in data/models/ (the hub only ships a pickle).
2. Deterministic checks: the sentence cites at least one passage and every citation exists.

A sentence is SUPPORTED if it has valid citations and support >= 0.5 (MiniCheck's threshold,
fixed before validation). Validated on 60 blind human-labelled pairs (results/
verifier_validation.json): catches 15/15 unsupported, passes 42/45 supported, kappa 0.88.
A number-in-source check was measured on the same pairs, added nothing, and was removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from plexus.settings import REPO_ROOT

MODEL_DIR = REPO_ROOT / "data" / "models" / "minicheck-flan-t5-large"
THRESHOLD = 0.5
CHUNK_WORDS = 400

_CITE = re.compile(r"\[(\d+)\]")
# sentence end = . ! ? (optionally followed by citations) then a capital, digit, quote, bullet
# or "["; not after common abbreviations like U.S. or Inc. (review 2026-10-29)
_ABBREV = ("U.S.", "U.K.", "Inc.", "Corp.", "Co.", "Ltd.", "No.", "vs.", "e.g.", "i.e.", "approx.")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(*\[])|\n+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def split_sentences(text: str) -> list[str]:
    """Simple splitter (no nltk): sentence ends, then bullets / new lines. A piece that is
    only citations ("[1]") is glued back to the sentence before it."""
    out: list[str] = []
    for piece in _SENT.split(text):
        piece = (piece or "").strip(" -*\t")
        lead = re.match(r"((?:\[\d+\]\s*)+)(?=\S)", piece)  # "[2] Next item" -> "[2]" | rest
        if out and lead and piece[lead.end() :].strip():
            out[-1] = f"{out[-1]} {lead.group(1).strip()}"
            piece = piece[lead.end() :].strip()
        if not piece:
            continue
        if out and (re.fullmatch(r"(\[\d+\]\s*)+\.?", piece) or out[-1].endswith(_ABBREV)):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def strip_citations(sentence: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", sentence).replace("**", "").strip()


def numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    support: float | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.issues and self.support is not None and self.support >= THRESHOLD


class MiniCheck:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"no MiniCheck weights at {model_dir}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, dtype=dtype, use_safetensors=True
        ).to(self.device)
        self.model.eval()

    def _chunks(self, doc: str) -> list[str]:
        """Sentence-aware chunks of about CHUNK_WORDS words, as in the reference code."""
        chunks, cur, n = [], [], 0
        for sent in split_sentences(doc) or [""]:
            w = len(sent.split())
            if cur and n + w > CHUNK_WORDS:
                chunks.append(" ".join(cur))
                cur, n = [], 0
            cur.append(sent)
            n += w
        return [*chunks, " ".join(cur)] if cur else chunks or [""]

    def score(self, doc: str, claim: str) -> float:
        torch = self.torch
        texts = [f"predict: {c}{self.tok.eos_token}{claim}" for c in self._chunks(doc)]
        enc = self.tok(texts, max_length=2048, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        dec = torch.zeros((len(texts), 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(**enc, decoder_input_ids=dec).logits[:, 0, :].float()
        probs = torch.softmax(logits[:, [3, 209]], dim=-1)[:, 1]
        if not torch.isfinite(probs).all():  # fp16 T5 can overflow: report "unknown", not "no"
            return float("nan")
        return float(probs.max())


_LOADED: dict[str, MiniCheck] = {}


def load_minicheck() -> MiniCheck:
    """One MiniCheck per process (the attack eval and the API build many flows)."""
    if "m" not in _LOADED:
        _LOADED["m"] = MiniCheck()
    return _LOADED["m"]


def check_answer(answer: str, passages: list[str], scorer: MiniCheck | None) -> list[SentenceCheck]:
    """passages[i] is the text behind citation [i+1]."""
    out = []
    for sent in split_sentences(answer):
        cited = sorted({int(n) for n in _CITE.findall(sent)})
        claim = strip_citations(sent)
        if len(claim.split()) < 4:  # headings, "Sources:", list stubs
            continue
        sc = SentenceCheck(claim, cited)
        valid = [n for n in cited if 1 <= n <= len(passages)]
        if not cited:
            sc.issues.append("no_citation")
        elif len(valid) < len(cited):
            sc.issues.append("bad_citation")
        if valid:
            doc = "\n\n".join(passages[n - 1] for n in valid)
            if scorer is not None:
                sc.support = scorer.score(doc, claim)
                if sc.support != sc.support:  # NaN
                    sc.support = None
                    sc.issues.append("score_unknown")
        out.append(sc)
    return out
