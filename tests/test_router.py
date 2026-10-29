from plexus.eval.routing import confusion
from plexus.extract.resolve import Resolver
from plexus.index.dense import Hit
from plexus.retrieve.graph_search import EntityLinker
from plexus.retrieve.router import FixedRouter, OracleRouter, RoutedRetriever, RuleRouter

FILERS = {"AVGO": {"company": "Broadcom Inc."}, "AMD": {"company": "ADVANCED MICRO DEVICES INC"},
          "SITM": {"company": "SITIME Corp"}}  # fmt: skip
LINK = EntityLinker(Resolver(FILERS, {"company": {"TSMC": ["Taiwan Semiconductor"]},
                                      "filers": {"AMD": ["AMD"]}}), [])  # fmt: skip


def test_rule_router_counts_named_companies():
    r = RuleRouter(LINK)
    assert r.route("How do all semiconductor companies describe tariffs?") == "global"
    assert r.route("What did Broadcom say about its revenue?") == "local"
    assert r.route("What do AMD and SiTime each rely on TSMC for?") == "multi_hop"
    # one filer + a non-filer company is still a single-company question
    assert r.route("What did Broadcom pay for the assets it bought from TSMC?") == "local"


def test_routed_retriever_dispatches_and_reports_route():
    class R:
        def __init__(self, tag):
            self.tag = tag

        def retrieve(self, q, k):
            return [Hit(self.tag, "D", 0, 1, "s", 1.0, "t")]

    rr = RoutedRetriever(FixedRouter("global"), {r: R(r) for r in ("local", "multi_hop", "global")})
    assert rr.retrieve("q", 5)[0].chunk_id == "global" and rr.last_route == "global"
    rr.router = OracleRouter(lambda q: "local")
    assert rr.retrieve("q", 5)[0].chunk_id == "local"


def test_confusion_matrix():
    c = confusion([("local", "local"), ("global", "local"), ("multi_hop", "multi_hop")])
    assert c["accuracy"] == round(2 / 3, 4) and c["gold_x_predicted"]["global"]["local"] == 1


def test_short_all_caps_aliases_link_like_tickers():
    res = Resolver({"TXN": {"company": "TEXAS INSTRUMENTS INC"}}, {"filers": {"TXN": ["TI"]}})
    link = EntityLinker(res, [])
    assert link.link("What does TI say drives its revenue?") == ["company:TXN"]
    assert link.link("What is ti?") == []  # lowercase never links


def test_non_filer_short_aliases_never_count_as_filers():
    filers = {"NVDA": {"company": "NVIDIA CORP"}, "SWKS": {"company": "SKYWORKS SOLUTIONS, INC."}}
    res = Resolver(
        filers,
        {"company": {"Amazon": ["AWS"], "U.S. Department of Commerce": ["DOC"]}},
        filer_tokens={"skyworks": "SWKS"},
    )  # as load_resolver builds it
    link = EntityLinker(res, [])
    assert link.filer_ids == {"company:NVDA", "company:SWKS"}
    assert "DOC" not in link.tickers and "AWS" not in link.tickers
    # Amazon is the linking entity, not a third filer
    assert RuleRouter(link).route("How do NVIDIA and Skyworks each describe Amazon?") == "multi_hop"


def test_sector_cue_and_lowercase_distinctive_names():
    filers = {"QCOM": {"company": "QUALCOMM INC/DE"}}
    res = Resolver(filers, {"company": {"TSMC": []}}, filer_tokens={"qualcomm": "QCOM"},
                   token_caps={"qualcomm": 1.0})  # fmt: skip
    r = RuleRouter(EntityLinker(res, []))
    assert r.route("What are the main risks for qualcomm?") == "local"
    assert r.route("Which companies depend on TSMC for wafers?") == "global"
    assert r.route("What does Qualcomm say about TSMC?") == "local"
