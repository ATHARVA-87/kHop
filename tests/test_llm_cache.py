import pytest
from pydantic import BaseModel

from plexus.llm import (
    CacheMissError,
    FakeProvider,
    LLMCache,
    LLMClient,
    LLMRequest,
    Message,
    ProviderError,
    StructuredOutputError,
)


def _req(text: str = "hello", **kw) -> LLMRequest:
    return LLMRequest(
        provider="fake", model="m1", messages=(Message(role="user", content=text),), **kw
    )


@pytest.fixture
def client(tmp_path):
    fake = FakeProvider()
    return LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": fake}), fake


def test_second_identical_call_is_served_from_cache(client):
    c, fake = client
    first = c.complete(_req())
    second = c.complete(_req())
    assert fake.calls == 1
    assert not first.cached and second.cached
    assert first.text == second.text == "echo: hello"
    assert c.stats.cache_hit_rate == 0.5


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0.7},
        {"max_tokens": 10},
        {"seed": 1},
        {"model": "m2"},
        {"params": {"top_p": 0.9}},
        {"json_schema": {"type": "object"}},
    ],
)
def test_cache_key_covers_every_parameter(change):
    assert _req().cache_key() != _req().model_copy(update=change).cache_key()


def test_cache_key_is_stable_and_order_independent():
    a = _req(params={"a": 1, "b": 2})
    b = _req(params={"b": 2, "a": 1})
    assert a.cache_key() == b.cache_key()
    assert len(a.cache_key()) == 64


def test_cache_persists_across_client_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    fake = FakeProvider()
    LLMClient(LLMCache(path), {"fake": fake}).complete(_req())
    fresh = FakeProvider()
    resp = LLMClient(LLMCache(path), {"fake": fresh}).complete(_req())
    assert resp.cached and fresh.calls == 0


def test_offline_mode_never_calls_a_provider(tmp_path):
    fake = FakeProvider()
    c = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": fake}, offline=True)
    with pytest.raises(CacheMissError):
        c.complete(_req())
    assert fake.calls == 0


def test_unknown_or_disabled_provider_is_refused(tmp_path):
    c = LLMClient(LLMCache(tmp_path / "c.sqlite"), {})
    with pytest.raises(ProviderError, match="PLEXUS_ENABLE_CLAUDE_SDK"):
        c.complete(_req().model_copy(update={"provider": "claude_sdk"}))


class Answer(BaseModel):
    answer: str
    confidence: float


def test_complete_json_strips_fences_and_validates(tmp_path):
    fake = FakeProvider(lambda r: 'Sure!\n```json\n{"answer": "42", "confidence": 0.9}\n```')
    c = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": fake})
    out = c.complete_json(_req(), Answer)
    assert out == Answer(answer="42", confidence=0.9)


def test_complete_json_rejects_wrong_shape(tmp_path):
    fake = FakeProvider(lambda r: '{"answer": "42"}')
    c = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": fake})
    with pytest.raises(StructuredOutputError):
        c.complete_json(_req(), Answer)
