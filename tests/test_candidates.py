import json
import random
import re

from plexus.eval.candidates import (
    CandidateConfig,
    generate,
    load_passages,
    sample_global,
    sample_multi_hop,
    summarise,
)
from plexus.ingest.process import process_snapshot
from plexus.llm import FakeProvider, LLMCache, LLMClient

PAD = " ".join(f"Filler sentence {i} on general corporate topics." for i in range(40))


def _filing(name, risk):
    return (
        f"<html><body><p>Item 1. Business</p><p>{name} designs chips. {PAD}</p>"
        f"<p>Item 1A. Risk Factors</p><p>{risk} {PAD}</p>"
        f"<p>Item 7. Management's Discussion and Analysis</p><p>{name} revenue grew. {PAD}</p>"
        "</body></html>"
    )


FILINGS = {
    "NVDA": (
        "NVIDIA Corp",
        "We rely on TSMC for wafers. U.S. export control rules limit sales to China.",
    ),
    "AMD": (
        "Advanced Micro Devices",
        "We also rely on TSMC for production. Export control rules affect us.",
    ),
    "QCOM": (
        "Qualcomm Inc",
        "Our modems are fabricated by TSMC and Samsung. New export control rules may hurt.",
    ),
}


def _corpus(tmp_path):
    data = tmp_path / "data"
    (data / "snapshots").mkdir(parents=True)
    rows = []
    for t, (name, risk) in FILINGS.items():
        p = data / "raw" / "s" / f"{t}.htm"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_filing(name, risk))
        rows.append({"doc_id": f"{t}_10K", "ticker": t, "company": name, "form": "10-K",
                     "filing_date": "2025-02-01", "report_date": "", "sha256": "x", "path": f"s/{t}.htm"})  # fmt: skip
    (data / "snapshots" / "s.manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    process_snapshot(data, "s")
    return data


CFG = CandidateConfig(
    snapshot="s",
    provider="fake",
    model="m",
    counts={"local": 4, "multi_hop": 2, "global": 1},
    passage_words=60,
    counterparties=["TSMC", "Samsung"],
    themes={"export control": "U.S. export controls"},
)


def _passages(prompt: str) -> list[str]:
    return re.findall(r"<passage>\n(.*?)\n</passage>", prompt, re.DOTALL)


def responder(req):
    prompt = req.messages[-1].content
    ps = _passages(prompt)
    q = lambda text: " ".join(text.split()[:12])  # noqa: E731 - a verbatim 12-word quote
    if '"quote_a"' in prompt:
        return json.dumps({"question": "Compare A and B?", "sub_questions": ["A?", "B?"],
                           "answer": "Both.", "quote_a": q(ps[0]), "quote_b": q(ps[1])})  # fmt: skip
    if '"quotes"' in prompt:
        quotes = [{"passage": i + 1, "quote": q(p)} for i, p in enumerate(ps)]
        return json.dumps(
            {"question": "Which firms cite export controls?", "answer": "All.", "quotes": quotes}
        )
    if "Filler sentence 3 " in ps[0]:  # simulate a hallucinated quote for some passages
        return json.dumps(
            {
                "question": "Bad?",
                "answer": "x",
                "quote": "words that never appear in any filing at all",
            }
        )
    return json.dumps(
        {
            "question": f"What does it say about {q(ps[0])[:30]}?",
            "answer": "It says so.",
            "quote": q(ps[0]),
        }
    )


def test_generate_keeps_only_locatable_candidates(tmp_path):
    data = _corpus(tmp_path)
    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": FakeProvider(responder)})
    queries, log = generate(CFG, llm, data)
    by_type = {t: [q for q in queries if q.qtype == t] for t in ("local", "multi_hop", "global")}
    assert by_type["multi_hop"] and by_type["global"]
    hop = by_type["multi_hop"][0]
    assert len({e.doc_id for e in hop.evidence}) == 2 and hop.sub_questions == ["A?", "B?"]
    assert "entity=" in hop.source
    assert len(by_type["global"][0].evidence) == 3
    summary = summarise(log)
    assert summary["local"].get("kept", 0) == len(by_type["local"])
    assert all(len(q.evidence) >= 1 for q in queries)
    statuses = {r["status"] for r in log}
    assert statuses <= {"kept", "quote_not_found", "duplicate", "bad_json", "too_few_valid_quotes"}


def test_rerun_is_free_and_identical(tmp_path):
    data = _corpus(tmp_path)
    fake = FakeProvider(responder)
    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": fake})
    first, _ = generate(CFG, llm, data)
    calls = fake.calls
    second, _ = generate(CFG, llm, data)
    assert fake.calls == calls  # every call served from cache
    assert [q.model_dump() for q in first] == [q.model_dump() for q in second]


def test_multi_hop_pairs_are_cross_company_and_global_needs_three_firms(tmp_path):
    data = _corpus(tmp_path)
    passages, _ = load_passages(data, CFG)
    pairs = sample_multi_hop(passages, CFG, random.Random(0), 10)
    assert pairs and all(a.ticker != b.ticker for a, b, _ in pairs)
    assert {e for _, _, e in pairs} <= {"TSMC"}  # Samsung appears in only one filing
    glob = sample_global(passages, CFG, random.Random(0), 1)
    assert len({p.ticker for p in glob[0][0]}) == 3


def test_hallucinated_quotes_are_all_rejected(tmp_path):
    data = _corpus(tmp_path)

    def liar(req):
        fake = "this sentence does not appear anywhere in these filings at all"
        return json.dumps({"question": "q?", "answer": "a", "quote": fake, "quote_a": fake,
                           "quote_b": fake, "sub_questions": ["x?", "y?"],
                           "quotes": [{"passage": 1, "quote": fake}]})  # fmt: skip

    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": FakeProvider(liar)})
    queries, log = generate(CFG, llm, data)
    assert queries == []
    s = summarise(log)
    assert s["auto_filter_rejection_rate"] == 1.0
    assert s["local"]["quote_not_found"] == 4 and s["global"]["too_few_valid_quotes"] == 1


def test_overloaded_main_model_falls_back_and_failures_are_not_fatal(tmp_path):
    from plexus.llm import ProviderError

    data = _corpus(tmp_path)

    class Overloaded:
        name = "fake"
        calls = 0

        def complete(self, req):
            self.calls += 1
            if req.model == "m":
                raise ProviderError("503 high demand")
            if req.model == "dead":
                raise ProviderError("still down")
            return FakeProvider(responder).complete(req)

    prov = Overloaded()
    llm = LLMClient(LLMCache(tmp_path / "c.sqlite"), {"fake": prov})
    llm._call_with_retry = lambda p, r: p.complete(r)  # no retry sleeps in tests
    cfg = CFG.model_copy(update={"fallback_models": ["lite"]})
    queries, log = generate(cfg, llm, data)
    assert queries and all("fake/lite" in q.source for q in queries)

    cfg_dead = CFG.model_copy(update={"fallback_models": ["dead"]})
    queries, log = generate(
        cfg_dead, LLMClient(LLMCache(tmp_path / "d.sqlite"), {"fake": prov}), data
    )
    assert queries == [] and {r["status"] for r in log} == {"provider_error"}


def _bridge_responder(leak: bool):
    def responder_(req):
        ps = _passages(req.messages[-1].content)

        def q(text):
            words = text.split()
            i = next(n for n, w in enumerate(words) if "TSMC" in w)
            return " ".join(words[max(0, i - 4) : i + 6])

        question = "Which foundry is named by both NVIDIA and AMD?" + (
            " Is it TSMC?" if leak else ""
        )
        return json.dumps({"question": question, "sub_questions": ["NVIDIA foundry?", "AMD foundry?"],
                           "answer": "TSMC", "quote_a": q(ps[0]), "quote_b": q(ps[1])})  # fmt: skip

    return responder_


def test_bridge_questions_hide_the_linking_entity(tmp_path):
    data = _corpus(tmp_path)
    cfg = CFG.model_copy(update={"counts": {"bridge": 2}})
    llm = LLMClient(
        LLMCache(tmp_path / "a.sqlite"), {"fake": FakeProvider(_bridge_responder(leak=False))}
    )
    queries, _ = generate(cfg, llm, data)
    assert queries and all(q.tags == ["bridge"] and q.qtype == "multi_hop" for q in queries)
    assert all("tsmc" not in q.question.lower() and q.qid.startswith("B-") for q in queries)

    llm = LLMClient(
        LLMCache(tmp_path / "b.sqlite"), {"fake": FakeProvider(_bridge_responder(leak=True))}
    )
    queries, log = generate(cfg, llm, data)
    assert queries == [] and {r["status"] for r in log} == {"names_linking_entity"}


def test_old_multi_hop_rows_default_to_comparison_tag(tmp_path):
    from plexus.eval.golden import row_to_query

    row = {"qid": "M-1", "qtype": "multi_hop", "question": "q?", "answer": "a",
           "evidence_doc_ids": "D", "evidence_quotes": "x", "verdict": "accept"}  # fmt: skip
    assert row_to_query(row).tags == ["comparison"]
