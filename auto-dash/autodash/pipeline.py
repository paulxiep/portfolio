"""Pipeline graph (LangGraph StateGraph).

Graph topology (MVP):
    load -> plan -> explore -> chart -> comply -> output -> END

Stub nodes return empty dicts. Real implementations replace them in MVP.2-9.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from autodash.models import PipelineState


async def load_node(state: PipelineState) -> dict:
    """Node: Load and profile data (MVP.2). Stub."""
    return {}


async def plan_node(state: PipelineState) -> dict:
    """Node: Plan analysis steps (MVP.3). Stub."""
    return {}


async def explore_node(state: PipelineState) -> dict:
    """Node: Execute analysis and produce insights (MVP.4). Stub."""
    return {}


async def chart_node(state: PipelineState) -> dict:
    """Node: Plan charts and generate code (MVP.5). Stub."""
    return {}


async def comply_node(state: PipelineState) -> dict:
    """Node: Run plotlint convergence on each chart (MVP.6-8). Stub."""
    return {}


async def output_node(state: PipelineState) -> dict:
    """Node: Write final output (MVP.9). Stub."""
    return {}


def build_pipeline_graph(
    config: Optional[Any] = None,
    llm_client: Optional[Any] = None,
    enable_hitl: bool = False,
    extra_nodes: Optional[dict] = None,
) -> CompiledStateGraph:
    """Build the AutoDash pipeline as a LangGraph StateGraph.

    Args:
        config: PipelineConfig instance.
        llm_client: Shared LLMClient for all LLM-calling nodes.
        enable_hitl: Insert interrupt() at checkpoint nodes (DI-4.2).
        extra_nodes: Additional nodes to insert.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("load", load_node)
    graph.add_node("plan", plan_node)
    graph.add_node("explore", explore_node)
    graph.add_node("chart", chart_node)
    graph.add_node("comply", comply_node)
    graph.add_node("output", output_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "plan")
    graph.add_edge("plan", "explore")
    graph.add_edge("explore", "chart")
    graph.add_edge("chart", "comply")
    graph.add_edge("comply", "output")
    graph.add_edge("output", END)

    return graph.compile()
