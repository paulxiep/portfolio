"""Tests for autodash.pipeline — pipeline graph topology and stub passthrough."""

from pathlib import Path

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

SAMPLE_CSV = str(Path(__file__).parent / "test_data" / "sample.csv")


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
    async def test_load_no_source_path(self):
        result = await load_node({})
        assert "errors" in result
        assert any("No source_path" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_plan_fallback_returns_error(self):
        result = await plan_node({})
        assert "errors" in result
        assert any("No LLM client" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_explore_fallback_returns_error(self):
        result = await explore_node({})
        assert "errors" in result
        assert any("No LLM client" in e for e in result["errors"])

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
    async def test_full_pipeline_with_real_load(self):
        """Run the full pipeline with real load node and fallback plan."""
        graph = build_pipeline_graph()
        initial_state = {
            "source_path": SAMPLE_CSV,
            "questions": "What trends are there?",
        }
        result = await graph.ainvoke(initial_state)
        assert result["source_path"] == SAMPLE_CSV
        assert result["questions"] == "What trends are there?"
        assert result["data_profile"] is not None
        assert result["data_profile"].row_count == 20
        # Without llm_client, plan node adds an error
        assert any("No LLM client" in e for e in result.get("errors", []))

    @pytest.mark.asyncio
    async def test_pipeline_with_mock_llm_plans_successfully(self):
        """Plan node produces analysis steps with a mock LLM client."""
        from autodash.config import PipelineConfig

        class MockLLMClient:
            async def complete(self, system, user, **kwargs):
                return """[{
                    "description": "Revenue trend",
                    "target_columns": ["revenue"],
                    "aggregation": "sum",
                    "group_by_columns": [],
                    "rationale": "Total revenue"
                }]"""

            async def complete_with_image(self, **kwargs):
                return ""

        config = PipelineConfig(max_analysis_steps=1)
        graph = build_pipeline_graph(config=config, llm_client=MockLLMClient())
        initial_state = {
            "source_path": SAMPLE_CSV,
            "questions": "What is total revenue?",
        }
        result = await graph.ainvoke(initial_state)
        assert result["data_profile"] is not None
        assert len(result.get("analysis_steps", [])) == 1
        assert result["analysis_steps"][0].description == "Revenue trend"
