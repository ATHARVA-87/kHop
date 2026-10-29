import numpy as np

from plexus.serve.semantic_cache import SemanticCache, key_of


class Linker:
    def link(self, q):
        return [f"company:{t}" for t in ("AMD", "INTC") if t in q]


def encode(texts):  # bag-of-words toy embedding, unit length
    vocab = [
        "risk",
        "risks",
        "export",
        "amd",
        "intc",
        "china",
        "what",
        "are",
        "the",
        "main",
        "does",
        "list",
    ]
    out = []
    for t in texts:
        v = np.array([t.count(w) for w in vocab], dtype=np.float32) + 1e-3
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def test_reuse_needs_same_companies_and_periods(tmp_path):
    c = SemanticCache(tmp_path / "c.sqlite", encode, Linker(), "snap:abc", threshold=0.9)
    c.store("What are AMD export risks?", {"answer": "A"})
    hit = c.lookup("What are AMD export risks")
    assert hit and hit[0]["answer"] == "A"
    assert c.lookup("What are INTC export risks?") is None  # other company: never reused
    assert key_of("risks in 2024", None) != key_of("risks in 2025", None)


def test_ttl_and_scope(tmp_path):
    c = SemanticCache(tmp_path / "c.sqlite", encode, Linker(), "snap:abc", threshold=0.9, ttl_s=10)
    c.store("What are AMD export risks?", {"answer": "A"}, now=0)
    assert c.lookup("What are AMD export risks?", now=5)
    assert c.lookup("What are AMD export risks?", now=11) is None  # too old
    other = SemanticCache(tmp_path / "c.sqlite", encode, Linker(), "snap:NEW", threshold=0.9)
    assert other.lookup("What are AMD export risks?", now=5) is None  # new corpus/config
