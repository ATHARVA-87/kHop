import json

from plexus.extract.resolve import Resolver, build, concept_key, fuzzy_merges, norm_key

FILERS = {
    "NVDA": {"company": "NVIDIA CORP", "ticker": "NVDA"},
    "ON": {"company": "ON SEMICONDUCTOR CORP", "ticker": "ON"},
}
ALIASES = {
    "company": {"TSMC": ["Taiwan Semiconductor Manufacturing Company", "Taiwan Semiconductor"]},
    "geography": {"China": ["PRC", "mainland China"]},
}


def test_norm_key_strips_trailing_legal_words_only():
    assert norm_key("The Microsoft Corporation") == "microsoft"
    assert norm_key("Hon Hai Precision Industry Co., Ltd.") == "hon hai precision industry"
    assert (
        norm_key("Taiwan Semiconductor Manufacturing Company Limited")
        == "taiwan semiconductor manufacturing"
    )
    assert norm_key("ON Semiconductor Corp") == "on semiconductor"  # inner word kept


def test_resolver_maps_filers_aliases_and_keeps_anonymous_apart():
    r = Resolver(FILERS, ALIASES)
    assert r.resolve_exact("Nvidia Corp", "company")[0] == "company:NVDA"
    assert r.resolve_exact("onsemi", "company")[0] == "company:ON"  # common name
    tsmc = r.resolve_exact("Taiwan Semiconductor Manufacturing Company Limited", "company")
    assert tsmc[:2] == ("company:tsmc", "TSMC") and tsmc[2] == "alias"
    assert r.resolve_exact("PRC", "geography")[0] == "geography:china"
    a = r.resolve_exact("NVDA Customer A", "company")[0]
    b = r.resolve_exact("ON Customer A", "company")[0]
    assert a != b and a.startswith("company:anon:")


def test_concepts_fold_plurals_and_types_never_mix():
    assert concept_key("Export Controls") == concept_key("export control")
    r = Resolver(FILERS, ALIASES)
    assert r.resolve_exact("China", "geography")[0] != r.resolve_exact("China", "risk_factor")[0]


def test_fuzzy_merges_only_near_identical_same_first_word():
    ids = {"company": {"company:a": "wistron", "company:b": "wistron corporation x",
                       "company:c": "wistronn", "company:d": "fabrinet", "company:e": "intelsat", "company:f": "intel"},
           "regulation": {"r:1": "country groups d1", "r:4": "country groups d4"}}  # fmt: skip
    remap = fuzzy_merges(ids)
    assert remap == {"company:c": ("company:a", remap["company:c"][1])}


def test_build_merges_variants_and_keeps_provenance(tmp_path):
    rel = lambda s, o, q: {"subject": s, "subject_type": "company", "relation": "manufactured_by",  # noqa: E731
                           "object": o, "object_type": "company", "qualifier": "wafers",
                           "quote": q, "start": 10, "end": 20}  # fmt: skip
    recs = [
        {"passage_id": "NVDA#p1", "doc_id": "NVDA_10K", "relations": [rel("Nvidia Corp", "TSMC", "q1")]},
        {"passage_id": "NVDA#p2", "doc_id": "NVDA_10K",
         "relations": [rel("NVIDIA", "Taiwan Semiconductor Manufacturing Company Limited", "q2")]},
    ]  # fmt: skip
    f = tmp_path / "ex.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    summary = build([f], Resolver(FILERS, ALIASES), tmp_path / "g")
    edges = [json.loads(x) for x in (tmp_path / "g" / "edges.jsonl").read_text().splitlines()]
    assert summary["edges"] == 1
    assert edges[0]["src"] == "company:NVDA" and edges[0]["dst"] == "company:tsmc"
    assert edges[0]["support"] == 2 and {e["quote"] for e in edges[0]["evidence"]} == {"q1", "q2"}
    ents = {
        json.loads(x)["id"]: json.loads(x)
        for x in (tmp_path / "g" / "entities.jsonl").read_text().splitlines()
    }
    assert ents["company:NVDA"]["ticker"] == "NVDA" and ents["company:tsmc"]["ticker"] is None


def test_distinctive_filer_tokens_need_proper_noun_use():
    from plexus.extract.resolve import distinctive_filer_tokens

    filers = {"SWKS": {"company": "SKYWORKS SOLUTIONS, INC."}, "ONTO": {"company": "ONTO INNOVATION INC."},
              "AMAT": {"company": "APPLIED MATERIALS INC /DE"}}  # fmt: skip
    stats = {"skyworks": [50, 50], "onto": [400, 30], "applied": [300, 40], "materials": [900, 100]}
    toks = distinctive_filer_tokens(filers, stats)
    assert toks == {"skyworks": "SWKS"}  # onto/applied are common words; materials generic
    r = Resolver(filers, {}, filer_tokens=toks)
    assert r.resolve_exact("Skyworks Semiconductor", "company")[:3:2] == (
        "company:SWKS",
        "filer_token",
    )
    assert r.resolve_exact("Skyworks Japan", "company")[0] == "company:skyworks-japan"
