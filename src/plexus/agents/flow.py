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

os.environ.setdefault("LANGSMITH_TRACING", "false")  # nothing leaves the machine

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
    """rag.retriever is a RoutedRetriever (A6) or any single retriever (route = "fixed")."""
    retriever = rag.retriever
    routes = getattr(retriever, "retrievers", None)
    k = rag.cfg.retrieval.top_k

    def route(s: FlowState) -> dict:
        t0 = time.perf_counter()
        r = retriever.router.route(s["question"]) if routes else "fixed"
        return {"route": r, **_step("route", t0, route=r)}

    def retrieve(s: FlowState) -> dict:
        t0 = time.perf_counter()
        hits = (routes[s["route"]] if routes else retriever).retrieve(s["question"], k)
        top = [{"doc_id": h.doc_id, "section": h.section} for h in hits[:3]]
        return {"hits": hits, **_step("retrieve", t0, n_hits=len(hits), top=top)}

    def answer(s: FlowState) -> dict:
        t0 = time.perf_counter()
        ans = rag.generate(s["question"], s["hits"])
        return {"answer": ans, **_step("answer", t0, cited=ans.cited_chunk_ids)}

    g = StateGraph(FlowState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()  # no checkpointer: state lives only for one question


def ask(flow, question: str) -> tuple[Answer, list[dict]]:
    out = flow.invoke({"question": question, "trace": []})
    return out["answer"], out["trace"]


def main(argv: list[str] | None = None) -> int:
    from plexus.config import ExperimentConfig
    from plexus.llm import LLMClient
    from plexus.retrieve.pipeline import make_retriever
    from plexus.settings import Settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/a6_router.yaml"))
    ap.add_argument("question")
    a = ap.parse_args(argv)
    settings = Settings.load()
    cfg = ExperimentConfig.load(a.config)
    retriever = make_retriever(cfg, settings.data_dir)
    try:
        rag = NaiveRAG(cfg, retriever, LLMClient.from_settings(settings))
        ans, trace = ask(build_flow(rag), a.question)
    finally:
        if hasattr(retriever, "close"):
            retriever.close()
    for i, t in enumerate(trace, 1):
        print(f"{i}. {json.dumps(t)}")
    print("\n" + ans.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
