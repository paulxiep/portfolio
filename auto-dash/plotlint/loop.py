"""Convergence loop graph (LangGraph StateGraph).

Graph topology:
    render -> inspect -> decide
                            ├── "patch" -> patch -> render (loop back)
                            └── "stop"  -> END

Stub nodes return empty dicts. Real implementations replace them in MVP.6-8.
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from plotlint.config import ConvergenceConfig
from plotlint.models import ConvergenceState


async def render_node(state: ConvergenceState) -> dict:
    """Stub: Execute chart code and capture figure + PNG.

    Replaced by real implementation in MVP.6.
    """
    return {}


async def inspect_node(state: ConvergenceState) -> dict:
    """Stub: Run Inspector on rendered figure.

    Replaced by real implementation in MVP.7.
    """
    return {}


async def patch_node(state: ConvergenceState) -> dict:
    """Stub: Generate a fix for the highest-severity issue.

    Replaced by real implementation in MVP.8.
    """
    return {}


def _make_should_continue(config: ConvergenceConfig):
    """Create a should_continue function that captures config.

    G1: LangGraph conditional edge functions only receive state.
    Config access is closed over via this factory.
    """

    def should_continue(state: ConvergenceState) -> str:
        """Conditional edge: decide whether to patch or stop.

        Returns:
            "patch" -> continue to patch_node
            "stop"  -> end the loop

        Stop conditions (checked in order):
        1. score >= target_score (perfect)
        2. iteration >= max_iterations
        3. render_error is set (code doesn't execute)
        4. score stagnant for stagnation_window iterations
        """
        score = state.get("score", 0.0)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", config.max_iterations)
        render_error = state.get("render_error")
        score_history = state.get("score_history", [])

        # 1. Perfect score
        if score >= config.target_score:
            return "stop"

        # 2. Max iterations reached
        if iteration >= max_iterations:
            return "stop"

        # 3. Render error (code doesn't execute)
        if render_error is not None:
            return "stop"

        # 4. Score stagnation
        if len(score_history) >= config.stagnation_window:
            recent = score_history[-config.stagnation_window :]
            if len(recent) >= 2:
                max_improvement = max(recent) - min(recent)
                if max_improvement < config.score_improvement_threshold:
                    return "stop"

        return "patch"

    return should_continue


def build_convergence_graph(
    config: ConvergenceConfig = ConvergenceConfig(),
    bundle: Optional[Any] = None,
    llm_client: Optional[Any] = None,
) -> CompiledStateGraph:
    """Build the plotlint convergence loop as a LangGraph StateGraph.

    Args:
        config: Controls stop conditions.
        bundle: RendererBundle (Renderer+Extractor pair). Default: matplotlib.
        llm_client: LLMClient for the patcher node.

    Returns a compiled StateGraph ready to invoke.
    """
    graph = StateGraph(ConvergenceState)

    graph.add_node("render", render_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("patch", patch_node)

    graph.add_edge(START, "render")
    graph.add_edge("render", "inspect")
    graph.add_conditional_edges(
        "inspect",
        _make_should_continue(config),
        {"patch": "patch", "stop": END},
    )
    graph.add_edge("patch", "render")

    return graph.compile()
