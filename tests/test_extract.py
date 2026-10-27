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
