"""Tests for autodash.pipeline — pipeline graph topology and stub passthrough."""

import pytest

from autodash.pipeline import (
    build_pipeline_graph,
    chart_node,
    comply_node,
    explore_node,
    load_node,
    output_node,
    plan_node,
)


class TestGraphTopology:
    def test_graph_has_expected_nodes(self):
        graph = build_pipeline_graph()
        node_names = set(graph.get_graph().nodes.keys())
        for name in ["load", "plan", "explore", "chart", "comply", "output"]:
            assert name in node_names

    def test_graph_compiles(self):
        graph = build_pipeline_graph()
        assert graph is not None


class TestStubPassthrough:
    @pytest.mark.asyncio
    async def test_load_stub(self):
        assert await load_node({}) == {}

    @pytest.mark.asyncio
    async def test_plan_stub(self):
        assert await plan_node({}) == {}

    @pytest.mark.asyncio
    async def test_explore_stub(self):
        assert await explore_node({}) == {}

    @pytest.mark.asyncio
    async def test_chart_stub(self):
        assert await chart_node({}) == {}

    @pytest.mark.asyncio
    async def test_comply_stub(self):
        assert await comply_node({}) == {}

    @pytest.mark.asyncio
    async def test_output_stub(self):
        assert await output_node({}) == {}


class TestPipelineExecution:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_stubs(self):
        """Run the full pipeline with stub nodes. State flows through."""
        graph = build_pipeline_graph()
        initial_state = {
            "source_path": "test.csv",
            "questions": "What trends are there?",
        }
        result = await graph.ainvoke(initial_state)
        # Stubs don't modify state, so input fields should pass through
        assert result["source_path"] == "test.csv"
        assert result["questions"] == "What trends are there?"
