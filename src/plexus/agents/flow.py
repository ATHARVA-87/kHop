"""The Plexus question flow as a LangGraph graph: route -> retrieve -> answer.

    python -m plexus.agents.flow --config configs/a6_router.yaml "Which companies rely on TSMC?"

Each node is a plain function over our own components (rule router, retrievers, NaiveRAG
writer), so every LLM call still goes through the cached LLMClient (project rule 4). LangGraph
only wires the steps and streams one trace entry per step, which the UI shows as the numbered
agent trace. In-memory only: no checkpointer, no server, LangSmith tracing off.

The flow is a straight line on purpose: the planned grade -> rewrite loop is added only if it
is measured to help (docs/failures.md lists the fixes that did not).
"""

from __future__ import annotations

import os

# nothing leaves the machine: forced off even if the shell exports tracing on (review 2026-10-29)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse
import json
import operator
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from plexus.baseline.naive import Answer, NaiveRAG
from plexus.index.dense import Hit


class FlowState(TypedDict, total=False):
    question: str
    route: str
    hits: list[Hit]
    answer: Answer
    trace: Annotated[list[dict[str, Any]], operator.add]  # each node appends one step


def _step(name: str, t0: float, **detail) -> dict:
    return {"trace": [{"step": name, "ms": round((time.perf_counter() - t0) * 1000, 1), **detail}]}


def build_flow(rag: NaiveRAG):
    """Nodes call the same public methods as NaiveRAG.answer, so the two paths cannot drift:
    route = the A6 rule router (or "fixed" for a single retriever), retrieve = the
    retriever's own retrieve(), answer = NaiveRAG.generate()."""
    retriever = rag.retriever
    router = getattr(retriever, "router", None)  # RoutedRetriever (A6) has one
    k = rag.cfg.retrieval.top_k

    use_gate = rag.cfg.guard.input_gate

    def guard(s: FlowState) -> dict:
        from plexus.guardrails.input_gate import gate

        t0 = time.perf_counter()
        hit = gate(s["question"], getattr(router, "linker", None))
        if hit is None:
            return _step("guard", t0, refused=False)
        ans = Answer(question=s["question"], text=hit[1], hits=[])
        return {"answer": ans, **_step("guard", t0, refused=True, reason=hit[0])}

    def route(s: FlowState) -> dict:
        t0 = time.perf_counter()
        r = router.route(s["question"]) if router else "fixed"
        return {"route": r, **_step("route", t0, route=r)}

    def retrieve(s: FlowState) -> dict:
        t0 = time.perf_counter()
        hits = retriever.retrieve(s["question"], k)  # RoutedRetriever routes again: same rule
        top = [{"doc_id": h.doc_id, "section": h.section} for h in hits[:3]]
        return {"hits": hits, **_step("retrieve", t0, n_hits=len(hits), top=top)}

    def answer(s: FlowState) -> dict:
        t0 = time.perf_counter()
        ans = rag.generate(s["question"], s["hits"])
        ans.retrieval_ms = next(t["ms"] for t in s["trace"] if t["step"] == "retrieve")
        return {"answer": ans, **_step("answer", t0, cited=ans.cited_chunk_ids)}

    g = StateGraph(FlowState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    if use_gate:  # refused questions end here, before any search or LLM call
        g.add_node("guard", guard)
        g.add_edge(START, "guard")
        g.add_conditional_edges(
            "guard", lambda s: END if "answer" in s else "route", ["route", END]
        )
    else:
        g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()  # no checkpointer: state lives only for one question


def ask(flow, question: str) -> tuple[Answer, list[dict]]:
    out = flow.invoke({"question": question, "trace": []})
    return out["answer"], out["trace"]


def check(flow, results: Path) -> int:
    """Same answers as a saved eval run, and route = golden question type? Writes results/flow_check.json."""
    from plexus.settings import REPO_ROOT

    saved = json.loads(results.read_text())["per_query"]
    rows = []
    for r in saved:
        ans, trace = ask(flow, r["question"])
        rows.append({"qid": r["qid"], "same_answer": ans.text == r["answer"],
                     # saved runs store no route; on dev the A6 router should equal the type
                     "route_is_qtype": trace[0]["route"] == r["qtype"]})  # fmt: skip
    out = {
        "results": str(results),
        "n": len(rows),
        "same_answer": sum(x["same_answer"] for x in rows),
        "route_is_qtype": sum(x["route_is_qtype"] for x in rows),
        "different": [x["qid"] for x in rows if not (x["same_answer"] and x["route_is_qtype"])],
    }
    (REPO_ROOT / "results" / "flow_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))
    return 0 if not out["different"] else 1


def main(argv: list[str] | None = None) -> int:
    from plexus.config import ExperimentConfig
    from plexus.llm import LLMClient
    from plexus.retrieve.pipeline import make_retriever
    from plexus.settings import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/a6_router.yaml"))
    ap.add_argument("question", nargs="?")
    ap.add_argument(
        "--check",
        action="store_true",
        help="run every question of --results through the flow (cache-only) and compare answers",
    )
    ap.add_argument("--results", type=Path, default=Path("results/a6_router.json"))
    a = ap.parse_args(argv)
    if a.check:
        os.environ["PLEXUS_LLM_OFFLINE"] = "1"  # cache-only: must reproduce, never re-generate
    elif not a.question:
        ap.error("give a question, or --check")
    settings = Settings.load()
    cfg = ExperimentConfig.load(a.config)
    retriever = make_retriever(cfg, settings.data_dir)
    try:
        rag = NaiveRAG(cfg, retriever, LLMClient.from_settings(settings))
        flow = build_flow(rag)
        if a.check:
            return check(flow, a.results)
        ans, trace = ask(flow, a.question)
    finally:
        if hasattr(retriever, "close"):
            retriever.close()
    for i, t in enumerate(trace, 1):
        print(f"{i}. {json.dumps(t)}")
    print("\n" + ans.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
