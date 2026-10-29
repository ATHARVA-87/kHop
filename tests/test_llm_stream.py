import pytest

from plexus.llm import FakeProvider, LLMCache, LLMClient, LLMRequest, Message
from plexus.llm.client import CacheMissError


def _req(q="q"):
    return LLMRequest(provider="ollama", model="m", messages=(Message(role="user", content=q),))


def _client(tmp_path, offline=False):
    return LLMClient(
        LLMCache(tmp_path / "c.sqlite"),
        {"ollama": FakeProvider(lambda r: "TSMC makes the chips [1].")},
        offline=offline,
    )


def test_stream_then_cache_then_same_as_complete(tmp_path):
    c = _client(tmp_path)
    pieces = list(c.stream(_req()))
    assert len(pieces) > 1 and "".join(pieces) == "TSMC makes the chips [1]." and not c.last_cached
    assert list(c.stream(_req())) == ["TSMC makes the chips [1]."] and c.last_cached
    assert c.complete(_req()).cached  # one cache entry for both paths


def test_early_stop_caches_nothing(tmp_path):
    c = _client(tmp_path)
    g = c.stream(_req())
    next(g)
    g.close()  # the reader went away
    assert c.cache.get(_req()) is None


def test_offline_miss_raises(tmp_path):
    with pytest.raises(CacheMissError):
        list(_client(tmp_path, offline=True).stream(_req()))
