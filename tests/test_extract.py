from plexus.extract.extract import Passage, validate
from plexus.extract.schema import Extraction, Relation, is_generic

TEXT = (
    "We rely on Taiwan Semiconductor Manufacturing Company Limited to produce our wafers. "
    "Our competitors include Broadcom and Marvell in networking. Sales to customers in "
    "China accounted for a significant share of revenue. We work with automakers worldwide."
)
P = Passage("X_10K#p0001", "X_10K", "X", "Example Corp", "business", 1000, 1000 + len(TEXT), TEXT)
TSMC = "Taiwan Semiconductor Manufacturing Company Limited"


def _rel(s, st, r, o, ot, q):
    return Relation(subject=s, subject_type=st, relation=r, object=o, object_type=ot, quote=q)


def test_validate_keeps_verified_relations_with_doc_offsets():
    q = f"rely on {TSMC} to produce our wafers"
    ex = Extraction(
        relations=[_rel("Example Corp", "company", "manufactured_by", TSMC, "company", q)]
    )
    rec, stats = validate(P, ex)
    assert stats["relations_kept"] == 1
    r = rec["relations"][0]
    # offsets point into the FILING text (passage start + offset inside the passage)
    assert r["start"] == 1000 + TEXT.index("rely on")
    assert {e["name"] for e in rec["entities"]} == {TSMC, "Example Corp"}


def test_validate_drops_and_records_bad_relations():
    comp = "Our competitors include Broadcom and Marvell in networking"
    ex = Extraction(
        relations=[
            _rel("Example Corp", "company", "competes_with", "Broadcom", "company",
                 "We compete fiercely with Broadcom in every market segment"),  # invented quote
            _rel("Example Corp", "company", "competes_with", "China", "geography", comp),  # bad types
            _rel("Example Corp", "company", "partners_with", "automakers", "company",
                 "We work with automakers worldwide and more"),  # generic group
            _rel("Example Corp", "company", "competes_with", "Marvell", "company", comp),
        ]
    )  # fmt: skip
    rec, stats = validate(P, ex)
    assert [r["object"] for r in rec["relations"]] == ["Marvell"]
    assert stats["drop_quote_not_found"] == 1
    assert stats["drop_type_mismatch"] == 1
    assert stats["drop_generic_entity"] == 1
    assert sorted(d["why"] for d in rec["dropped"]) == [
        "drop_generic_entity", "drop_quote_not_found", "drop_type_mismatch"]  # fmt: skip


def test_is_generic():
    assert is_generic("OEMs", "company") and is_generic("automakers", "company")
    assert is_generic("we", "company")
    assert not is_generic("TSMC", "company") and not is_generic("Microsoft", "company")
    assert not is_generic("export controls", "risk_factor")  # lowercase is fine for risks


def test_object_must_be_named_near_the_quote_and_subject_must_be_filer():
    ex = Extraction(
        relations=[
            # quote exists but never names the object
            _rel("Example Corp", "company", "operates_in", "Japan", "geography",
                 "Sales to customers in China accounted for a significant share of revenue"),
            # third-party subject that the quote does not name
            _rel("Intel", "company", "competes_with", "Broadcom", "company",
                 "Our competitors include Broadcom and Marvell in networking"),
            # filer named differently ("Example Corporation") still counts as the filer
            _rel("Example Corporation", "company", "competes_with", "Broadcom", "company",
                 "Our competitors include Broadcom and Marvell in networking"),
        ]
    )  # fmt: skip
    rec, stats = validate(P, ex)
    assert stats["drop_object_not_in_quote"] == 1
    assert stats["drop_subject_not_filer"] == 1
    assert [(r["subject"], r["object"]) for r in rec["relations"]] == [("Example Corp", "Broadcom")]


def test_anonymous_customers_are_namespaced_and_lowercase_brands_kept():
    text = "Customer A accounted for 14% of revenue. We compete with onsemi in power devices."
    p = Passage("Y#p1", "Y_10K", "Y", "Example Corp", "business", 0, len(text), text)
    ex = Extraction(
        relations=[
            Relation(subject="Example Corp", subject_type="company", relation="sells_to",
                     object="Customer A", object_type="company", qualifier="14% of revenue",
                     quote="Customer A accounted for 14% of revenue"),
            _rel("Example Corp", "company", "competes_with", "onsemi", "company",
                 "We compete with onsemi in power devices"),
        ]
    )  # fmt: skip
    rec, _ = validate(p, ex)
    objs = {r["object"]: r for r in rec["relations"]}
    assert set(objs) == {"Y Customer A", "onsemi"}
    assert objs["Y Customer A"]["qualifier"] == "14% of revenue"


def test_split_passage_halves_at_a_paragraph_and_keeps_offsets():
    from plexus.extract.extract import split_passage

    text = "para one is here.\npara two is here.\npara three.\npara four."
    p = Passage("Z#p1", "Z", "Z", "Z Corp", "business", 100, 100 + len(text), text)
    a, b = split_passage(p)
    assert a.text + b.text == text
    assert (a.start, a.end, b.start, b.end) == (100, 100 + len(a.text), 100 + len(a.text), p.end)
    assert a.passage_id.endswith("a") and b.passage_id.endswith("b")


def test_pilot_sample_is_stratified_and_skips_tuning_company():
    from plexus.extract.extract import pilot_sample

    ps = [Passage(f"{t}#p{i}", f"{t}_10K", t, t, sec, 0, 1, "x")
          for t in ("NVDA", "AMD", "INTC", "QCOM", "TXN") for sec in ("business", "risk_factors", "mdna")
          for i in range(3)]  # fmt: skip
    out = pilot_sample(ps, 9)
    assert all(p.ticker != "NVDA" for p in out)
    assert {p.section for p in out} == {"business", "risk_factors", "mdna"}
    for sec in ("business", "risk_factors", "mdna"):  # one passage per company per section
        tick = [p.ticker for p in out if p.section == sec]
        assert len(tick) == len(set(tick))


def test_groups_places_and_nationalities_are_not_companies():
    for junk in ["China", "Chinese OEMs", "China-based customers", "Taiwanese suppliers",
                 "U.S. government", "AMPI Facility", "the United States"]:  # fmt: skip
        assert is_generic(junk, "company"), junk
    for real in [
        "TSMC",
        "Samsung Electronics",
        "NVDA Customer A",
        "Customer B",
        "Hon Hai",
        "onsemi",
    ]:
        assert not is_generic(real, "company"), real
    assert not is_generic("China", "geography")  # fine as a geography
