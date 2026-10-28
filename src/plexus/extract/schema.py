"""Typed entities and relations extracted from filing passages (Phase 3).

Every relation must carry a verbatim quote from the passage. The quote is checked against
the passage text: a relation whose quote is not found is dropped (an extraction model's
invented facts never reach the graph), and a kept quote gives each graph edge exact
provenance (doc id + character span) that graph search can return as evidence.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "company",  # any organisation: filer, supplier, customer, competitor, foundry
    "person",  # executives, directors
    "product",  # product lines, platforms, technologies sold
    "segment",  # reporting segments, end markets ("Data Center", "Automotive")
    "risk_factor",  # a named risk ("export controls", "customer concentration")
    "geography",  # countries/regions with business exposure (China, Taiwan)
    "regulation",  # laws, rules, agencies (BIS Entity List, CHIPS Act)
]

RelationType = Literal[
    # Read "subject <relation> object". Every relation has a COMPANY as subject, normally the
    # filer: a small model reliably puts the filer first (v3 wrote "NVIDIA manufactures_for
    # TSMC" despite examples), so relations are named from the subject company's side.
    "sells_to",  # object is a customer of the subject
    "supplied_by",  # object supplies the subject (materials, components, equipment, IP)
    "manufactured_by",  # object is a foundry / assembler / contract manufacturer for the subject
    "competes_with",
    "partners_with",  # alliances, joint development, ecosystem partners (named)
    "licenses_from",  # subject licenses technology/IP from object
    "licenses_to",  # subject licenses its technology/IP to object (v6: Rambus-style licensors)
    "acquired",  # subject acquired object (company, business or product line)
    "sold_business_to",  # subject sold a business/asset to object (buyer); qualifier = what
    "has_segment",  # reporting segment or end market
    "makes_product",
    "exposed_to",  # a named risk, regulation or geography the subject is exposed to
    "operates_in",  # geography where the subject sells, makes or sources
    "subject_to",  # law, rule or agency
    "has_executive",  # object is a named executive or director of the subject
]

RELATION_TYPES: tuple[str, ...] = RelationType.__args__  # type: ignore[attr-defined]
ENTITY_TYPES: tuple[str, ...] = EntityType.__args__  # type: ignore[attr-defined]

# allowed (subject type, object type) per relation; anything else is a schema violation
RELATION_SIGNATURES: dict[str, tuple[set[str], set[str]]] = {
    "sells_to": ({"company"}, {"company"}),
    "supplied_by": ({"company"}, {"company"}),
    "manufactured_by": ({"company"}, {"company"}),
    "competes_with": ({"company"}, {"company"}),
    "partners_with": ({"company"}, {"company"}),
    "licenses_from": ({"company"}, {"company"}),
    "licenses_to": ({"company"}, {"company"}),
    "acquired": ({"company"}, {"company", "product", "segment"}),
    "sold_business_to": ({"company"}, {"company"}),
    "has_segment": ({"company"}, {"segment"}),
    "makes_product": ({"company"}, {"product"}),
    # a company object covers e.g. "Huawei is on the Entity List, which restricts our sales"
    "exposed_to": ({"company"}, {"risk_factor", "regulation", "geography", "company"}),
    "operates_in": ({"company"}, {"geography"}),
    "subject_to": ({"company"}, {"regulation"}),
    "has_executive": ({"company"}, {"person"}),
}


class Entity(BaseModel):
    name: str = Field(
        description="as written in the passage; the filer is its company name, never 'we'"
    )
    type: EntityType


class Relation(BaseModel):
    """Entity types sit on the relation itself: a separate entity list doubled the output
    and the model often used names in relations that it never listed (v1: 12 of 36 dropped)."""

    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType
    quote: str = Field(description="verbatim sentence fragment from the passage, 8-40 words")
    # optional detail the bare edge cannot hold: "12% of revenue", "sole source", "wafers",
    # "PHY IP business", "fiscal 2024"
    qualifier: str = ""


class Extraction(BaseModel):
    relations: list[Relation] = Field(default_factory=list)


# Groups, not organisations: "partners with automakers" is not a graph edge an analyst can use.
GENERIC_NAMES = {
    "customers", "customer", "suppliers", "supplier", "partners", "partner", "competitors",
    "competitor", "distributors", "distributor", "oems", "odms", "isvs", "csps", "hyperscalers",
    "automakers", "foundries", "foundry", "manufacturers", "vendors", "third parties",
    "third-party suppliers", "subcontractors", "universities", "governments", "start-ups",
    "startups", "tier-1 suppliers", "tier-one suppliers", "system integrators", "end customers",
    "cloud service providers", "the company", "we", "us", "our", "company",
}  # fmt: skip


# real organisations whose names are written in lower case
LOWERCASE_NAMES = {"onsemi", "imec", "ebay", "e.on"}

_ANON_CUSTOMER = re.compile(r"^(?:direct |indirect |end )?(?:customer|distributor) [a-z]$", re.I)


def strip_article(name: str) -> str:
    return re.sub(r"^the\s+", "", name.strip(), flags=re.I)


def is_anonymous(name: str) -> bool:
    """ "Customer A" in one filing is not "Customer A" in another."""
    return bool(_ANON_CUSTOMER.match(strip_article(name)))


# Places and nationalities: "China" or "Chinese OEMs" typed as a company is a group/place.
PLACE_NAMES = {
    "china", "taiwan", "japan", "korea", "south korea", "united states", "us", "u s", "usa",
    "u s a", "america", "europe", "european union", "eu", "asia", "germany", "india",
    "singapore", "malaysia", "vietnam", "philippines", "thailand", "israel", "mexico", "canada",
    "united kingdom", "uk", "u k", "france", "netherlands", "ireland", "hong kong", "russia",
    "ukraine", "middle east", "greater china", "mainland china", "prc",
}  # fmt: skip
DEMONYMS = {"chinese", "taiwanese", "japanese", "korean", "american", "european", "asian",
            "german", "indian", "israeli", "domestic", "foreign", "international", "global"}  # fmt: skip
# words that make a "company" name a group of companies or a non-company
GROUP_WORDS = {
    "customers", "customer", "suppliers", "supplier", "oems", "odms", "distributors",
    "government", "governments", "agencies", "agency", "companies", "competitors",
    "manufacturers", "vendors", "partners", "foundries", "facility", "facilities", "factories",
    "plants", "providers", "operators", "makers", "entities", "authorities", "regulators",
}  # fmt: skip


def is_group_or_place(name: str) -> bool:
    """'China', 'Chinese OEMs', 'China-based customers', 'U.S. government', 'AMPI Facility'."""
    words = re.findall(r"[a-z0-9]+", strip_article(name).lower())
    joined = " ".join(words)
    if not words:
        return True
    return (
        joined in PLACE_NAMES
        or words[0] in DEMONYMS
        or "based" in words
        or bool(set(words) & GROUP_WORDS)
    )


def is_generic(name: str, etype: str) -> bool:
    """True for a group/pronoun/place rather than a named organisation or person."""
    name = strip_article(name)
    low = name.lower()
    if low in GENERIC_NAMES:
        return True
    if etype not in ("company", "person") or low in LOWERCASE_NAMES:
        return False
    # anonymised names ("Customer A", namespaced "NVDA Customer A") are kept on purpose
    anonymous = re.search(r"\b(?:customer|distributor) [a-z]$", low) is not None
    if etype == "company" and not anonymous and is_group_or_place(name):
        return True
    # named organisations and people start with a capital letter or a digit ("3M")
    return not (name[:1].isupper() or name[:1].isdigit())
