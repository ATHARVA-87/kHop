"""Real-model tests (bge-small, ~130 MB download on first run). Skipped in CI."""

import pytest

from plexus.config import EmbedderConfig
from plexus.ingest.chunk import Chunk

pytestmark = pytest.mark.ml

TEXTS = [
    "We rely on Taiwan Semiconductor Manufacturing Company to fabricate our GPUs.",
    "Export controls imposed by the U.S. government restrict our sales to China.",
    "Our quarterly dividend was increased to one cent per share.",
]


@pytest.fixture(scope="module")
def embedder(tmp_path_factory):
    from plexus.index.dense import Embedder

    return Embedder(
        EmbedderConfig(device="cpu"), cache_path=tmp_path_factory.mktemp("e") / "emb.sqlite"
    )


def test_dense_search_ranks_the_relevant_chunk_first(embedder):
    from plexus.index.dense import DenseIndex

    chunks = [
        Chunk(f"D#c{i}", "D", t, i * 100, i * 100 + len(t), "risk_factors")
        for i, t in enumerate(TEXTS)
    ]
    vecs = embedder.embed_passages(TEXTS)
    assert vecs.shape == (3, 384)
    index = DenseIndex(":memory:", "chunks", embedder.dim)
    index.reset()
    index.add(chunks, vecs)
    assert index.count() == 3
    hits = index.search(embedder.embed_query("Which foundry manufactures the chips?"), k=3)
    assert hits[0].chunk_id == "D#c0"
    hits = index.search(embedder.embed_query("restrictions on selling to China"), k=3)
    assert hits[0].chunk_id == "D#c1" and hits[0].start == 100


def test_embedding_cache_returns_identical_vectors(embedder, monkeypatch):
    first = embedder.embed_passages(TEXTS)

    def boom(*a, **k):
        raise AssertionError("should have been served from cache")

    monkeypatch.setattr(embedder, "_encode", boom)
    again = embedder.embed_passages(TEXTS)
    assert (first == again).all()
