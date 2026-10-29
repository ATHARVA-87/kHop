"""Frozen test sets for the reframe layer, built BEFORE the layer exists (2026-10-29).

    python -m plexus.eval.reframe_sets      # -> data/golden/reframe/*.json(l)

- typos.jsonl: one-edit typos (delete / insert / substitute / swap, seed 0) of a company name
  or a content word of 5+ letters, from the held-out TEST questions only. Typos that happen to
  be a real corpus word are dropped and counted.
- traps.json: words that must never change (tickers, short names, finance terms, codes).
- consistency.jsonl: 40 dev questions (the owner-checked cache pairs) x variants: lowercase,
  company/companies swap, one company-name typo, and the owner-checked rewording.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys

import yaml

from plexus.settings import REPO_ROOT

OUT = REPO_ROOT / "data" / "golden" / "reframe"
TRAPS = ["ON", "TI", "onto", "Diodes", "Amkor", "capex", "EBITDA", "CoWoS", "HBM3E", "FY2024", "10-K",
         "SiTime", "MACOM", "onsemi", "KLA", "NAND", "DRAM"]  # fmt: skip
STOP = {"which", "what", "their", "there", "these", "those", "about", "under", "where", "while",
        "would", "could", "should", "other", "describe", "describes", "companies", "company"}  # fmt: skip


def corpus_words() -> set[str]:
    """Lowercase words of the parsed filings (used only to drop typos that are real words)."""
    from plexus.config import ExperimentConfig
    from plexus.index.build import index_dir

    cfg = ExperimentConfig.load(REPO_ROOT / "configs" / "app.yaml")
    words: set[str] = set()
    with open(index_dir(REPO_ROOT / "data", cfg) / "chunks.jsonl") as fh:
        for line in fh:
            words.update(re.findall(r"[a-z][a-z'-]+", json.loads(line)["text"].lower()))
    return words


def one_edit(word: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(word) - 1)  # keep the first letter, as people usually do
    kind = rng.choice(["delete", "insert", "substitute", "swap"])
    letters = "abcdefghijklmnopqrstuvwxyz"
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    if kind == "insert":
        return word[:i] + rng.choice(letters) + word[i:]
    if kind == "substitute":
        return word[:i] + rng.choice([c for c in letters if c != word[i].lower()]) + word[i + 1 :]
    return (
        word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if i + 1 < len(word)
        else word[: i - 1] + word[i] + word[i - 1]
    )


def edit_anywhere(word: str, rng: random.Random) -> str:
    """One typo at ANY position, the first letter included (owner, 2026-10-29)."""
    i = rng.randrange(0, len(word))
    kind = (
        rng.choice(["delete", "insert", "substitute", "swap"])
        if len(word) > 2
        else rng.choice(["insert", "substitute", "swap"])
    )
    letters = "abcdefghijklmnopqrstuvwxyz"
    keep_case = (lambda c: c.upper()) if word[i].isupper() else (lambda c: c)
    if kind == "delete":
        return word[:i] + word[i + 1 :]
    if kind == "insert":
        return word[:i] + keep_case(rng.choice(letters)) + word[i:]
    if kind == "substitute":
        return (
            word[:i]
            + keep_case(rng.choice([c for c in letters if c != word[i].lower()]))
            + word[i + 1 :]
        )
    j = i + 1 if i + 1 < len(word) else i - 1
    a, b = sorted((i, j))
    return word[:a] + word[b] + word[a] + word[b + 1 :]


def anywhere_set(questions: list[dict], known: set[str], seed: int = 2) -> tuple[list[dict], int]:
    """Per question: one typo in a random word, and two typos in two different words."""
    rng = random.Random(seed)
    out, dropped = [], 0
    for q in questions:
        words = [m for m in re.finditer(r"[A-Za-z]{2,}", q["question"])]
        for n_typos in (1, 2):
            if len(words) < n_typos:
                continue
            picks = sorted(rng.sample(words, n_typos), key=lambda m: m.start())
            text, edits, ok = q["question"], [], True
            for m in reversed(picks):
                bad = edit_anywhere(m.group(), rng)
                if bad.lower() in known or bad == m.group():
                    ok = False
                    break
                text = text[: m.start()] + bad + text[m.end() :]
                edits.append({"original": m.group(), "typo": bad, "length": len(m.group())})
            if not ok:
                dropped += 1
                continue
            out.append(
                {
                    "qid": q["qid"],
                    "typos": n_typos,
                    "edits": edits,
                    "question": q["question"],
                    "typo_question": text,
                }
            )
    return out, dropped


def company_names() -> list[str]:
    names = list(yaml.safe_load((REPO_ROOT / "configs" / "companies.yaml").read_text()).values())
    return sorted(
        {w for n in names for w in n.split() if len(w) >= 5 and w.isalpha()}, key=str.lower
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    known = corpus_words()
    names = company_names()
    test = [
        json.loads(x)
        for x in (REPO_ROOT / "data/golden/golden_test.jsonl").read_text().splitlines()
        if x.strip()
    ]
    typos, dropped = [], 0
    for q in test:
        words = re.findall(r"[A-Za-z]+", q["question"])
        name_words = [w for w in words if w in names]
        content = [
            w
            for w in words
            if len(w) >= 5 and w.lower() not in STOP and w not in names and w.islower()
        ]
        for kind, pool in (("company", name_words), ("word", content)):
            if not pool:
                continue
            target = rng.choice(pool)
            bad = one_edit(target, rng)
            if bad.lower() in known or bad == target:
                dropped += 1
                continue
            typos.append({"qid": q["qid"], "kind": kind, "original": target, "typo": bad,
                          "question": q["question"], "typo_question": q["question"].replace(target, bad, 1)})  # fmt: skip
    (OUT / "typos.jsonl").write_text("".join(json.dumps(t) + "\n" for t in typos))
    # fresh set for the second version (v1 failed; its fixes were made after seeing typos.jsonl):
    # dev questions, seed 1, never looked at before the v2 run
    rng2 = random.Random(1)
    dev = [
        json.loads(x)
        for x in (REPO_ROOT / "data/golden/golden.jsonl").read_text().splitlines()
        if x.strip()
    ]
    fresh = []
    for q in dev:
        words = re.findall(r"[A-Za-z]+", q["question"])
        name_words = [w for w in words if w in names]
        content = [
            w
            for w in words
            if len(w) >= 5 and w.lower() not in STOP and w not in names and w.islower()
        ]
        for kind, pool in (("company", name_words), ("word", content)):
            if not pool:
                continue
            target = rng2.choice(pool)
            bad = one_edit(target, rng2)
            if bad.lower() in known or bad == target:
                continue
            fresh.append({"qid": q["qid"], "kind": kind, "original": target, "typo": bad,
                          "question": q["question"], "typo_question": q["question"].replace(target, bad, 1)})  # fmt: skip
    (OUT / "typos_fresh.jsonl").write_text("".join(json.dumps(t) + "\n" for t in fresh))
    # typos ANYWHERE in the sentence (owner, 2026-10-29): any word incl. short ones, any position
    # incl. the first letter, 1 or 2 typos; dev + test questions; seed 2; frozen before measuring
    from plexus.serve.reframe import english_words

    english = set(english_words())
    anywhere, dropped_any = anywhere_set(dev + test, known | english)
    (OUT / "typos_anywhere.jsonl").write_text("".join(json.dumps(t) + "\n" for t in anywhere))
    print(f"anywhere {len(anywhere)} (dropped {dropped_any} typos that were real words)")
    (OUT / "traps.json").write_text(json.dumps({"words": TRAPS, "questions": [
        f"What does the filing say about {w} in 2025?" for w in TRAPS]}, indent=2) + "\n")  # fmt: skip
    with open(REPO_ROOT / "data/golden/cache_pairs.csv", newline="", encoding="utf-8-sig") as fh:
        pairs = [p for p in csv.DictReader(fh) if p["kind"] == "reuse"]
    cons = []
    for p in pairs:
        q = p["original"]
        swapped = re.sub(r"\bcompanies\b", "COMPANY_", q)
        swapped = re.sub(r"\bcompany\b", "companies", swapped).replace("COMPANY_", "company")
        variants = {"lowercase": q.lower(), "rewording_owner_checked": p["variant"]}
        if swapped != q:
            variants["company_plural_swap"] = swapped
        hit = [w for w in re.findall(r"[A-Za-z]+", q) if w in names]
        if hit:
            variants["company_typo"] = q.replace(hit[0], one_edit(hit[0], rng), 1)
        cons.append({"qid": p["qid"], "question": q, "variants": variants})
    (OUT / "consistency.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cons))
    print(
        f"typos {len(typos)} (dropped {dropped} that were real words), traps {len(TRAPS)}, consistency {len(cons)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
