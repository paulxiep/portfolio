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
    """Node: Load and profile data (MVP.2).

    Reads source_path from state, produces data_profile.
    """
    from autodash.data import load_and_profile

    source_path = state.get("source_path")
    if not source_path:
        return {"errors": state.get("errors", []) + ["No source_path provided"]}
    try:
        profile = load_and_profile(source_path)
        return {"data_profile": profile}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Data loading failed: {e}"]}


async def plan_node(state: PipelineState) -> dict:
    """Fallback plan node when no LLM client is configured."""
    return {"errors": state.get("errors", []) + ["No LLM client configured for planning"]}


def _make_plan_node(config: Any, llm_client: Any):
    """Create a plan node that captures config and llm_client via closure.

    Matches the closure factory pattern in plotlint/loop.py.
    LangGraph nodes must have signature (state) -> dict, so dependencies
    are captured via closure rather than passed as arguments.
    """
    if llm_client is None:
        return plan_node

    async def _plan_node(state: PipelineState) -> dict:
        """Node: Plan analysis steps (MVP.3)."""
        from autodash.planner import plan_analysis

        profile = state.get("data_profile")
        questions = state.get("questions", "")

        if not profile:
            return {
                "errors": state.get("errors", []) + [
                    "No data_profile available for planning"
                ]
            }

        if not questions:
            return {
                "errors": state.get("errors", []) + [
                    "No questions provided for analysis planning"
                ]
            }

        try:
            max_steps = config.max_analysis_steps if config else 1
            steps = await plan_analysis(
                profile=profile,
                questions=questions,
                llm_client=llm_client,
                max_steps=max_steps,
            )
            return {"analysis_steps": steps}
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [
                    f"Analysis planning failed: {e}"
                ]
            }

    return _plan_node


async def explore_node(state: PipelineState) -> dict:
    """Fallback explore node when no LLM client is configured."""
    return {"errors": state.get("errors", []) + ["No LLM client configured for exploration"]}


def _make_explore_node(config: Any, llm_client: Any):
    """Create an explore node that captures config and llm_client via closure.

    Matches the closure factory pattern used by _make_plan_node.
    The node re-loads the DataFrame from source_path because PipelineState
    does not carry the raw DataFrame (keeping state serializable).
    """
    if llm_client is None:
        return explore_node

    async def _explore_node(state: PipelineState) -> dict:
        """Node: Execute analysis steps and produce insights (MVP.4)."""
        from pathlib import Path

        from autodash.data import load_dataframe
        from autodash.explorer import explore_step

        profile = state.get("data_profile")
        steps = state.get("analysis_steps", [])
        source_path = state.get("source_path")

        if not profile:
            return {
                "errors": state.get("errors", []) + [
                    "No data_profile available for exploration"
                ]
            }

        if not steps:
            return {
                "errors": state.get("errors", []) + [
                    "No analysis_steps available for exploration"
                ]
            }

        if not source_path:
            return {
                "errors": state.get("errors", []) + [
                    "No source_path available for exploration"
                ]
            }

        try:
            df, _ = load_dataframe(Path(source_path))
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [
                    f"Failed to reload data for exploration: {e}"
                ]
            }

        max_attempts = config.max_exploration_attempts if config else 3

        insights: list = []
        errors: list = list(state.get("errors", []))
        for step in steps:
            try:
                insight = await explore_step(
                    step=step,
                    df=df,
                    profile=profile,
                    llm_client=llm_client,
                    max_attempts=max_attempts,
                )
                insights.append(insight)
            except Exception as e:
                errors.append(f"Exploration failed for step '{step.description}': {e}")

        result: dict = {"insights": insights}
        if errors != list(state.get("errors", [])):
            result["errors"] = errors
        return result

    return _explore_node


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
    graph.add_node("plan", _make_plan_node(config, llm_client))
    graph.add_node("explore", _make_explore_node(config, llm_client))
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
