"""Tests for chart planning and code generation prompt templates."""

from autodash.models import ChartType, ChartPriority
from autodash.prompts.chart_planning import (
    SYSTEM_PROMPT as PLANNING_SYSTEM_PROMPT,
    OUTPUT_FORMAT,
    build_user_prompt as build_planning_prompt,
)
from autodash.prompts.code_generation import (
    SYSTEM_PROMPT as CODEGEN_SYSTEM_PROMPT,
    build_user_prompt as build_codegen_prompt,
)


# =============================================================================
# Chart planning prompt
# =============================================================================


class TestChartPlanningSystemPrompt:
    def test_has_chart_types(self):
        for ct in ChartType:
            assert ct.value in PLANNING_SYSTEM_PROMPT

    def test_has_priorities(self):
        for p in ChartPriority:
            assert p.value in PLANNING_SYSTEM_PROMPT

    def test_has_mapping_rules(self):
        assert "require x AND y" in PLANNING_SYSTEM_PROMPT
        assert "require values AND categories" in PLANNING_SYSTEM_PROMPT
        assert "require x OR y" in PLANNING_SYSTEM_PROMPT

    def test_has_visualization_role(self):
        assert "visualization" in PLANNING_SYSTEM_PROMPT.lower()


class TestChartPlanningOutputFormat:
    def test_has_chart_type_field(self):
        assert "chart_type" in OUTPUT_FORMAT

    def test_has_data_mapping_field(self):
        assert "data_mapping" in OUTPUT_FORMAT

    def test_has_source_step_index(self):
        assert "source_step_index" in OUTPUT_FORMAT


class TestChartPlanningUserPrompt:
    def test_includes_all_insights(self):
        contexts = ["Insight A details", "Insight B details"]
        prompt = build_planning_prompt(contexts, "test question", 2)
        assert "Insight A details" in prompt
        assert "Insight B details" in prompt

    def test_includes_result_indices(self):
        contexts = ["First insight", "Second insight"]
        prompt = build_planning_prompt(contexts, "test", 2)
        assert "Analysis Result 0" in prompt
        assert "Analysis Result 1" in prompt

    def test_includes_questions(self):
        questions = "What is the revenue trend?"
        prompt = build_planning_prompt(["ctx"], questions, 1)
        assert questions in prompt

    def test_includes_max_charts(self):
        prompt = build_planning_prompt(["ctx"], "test", max_charts=3)
        assert "3" in prompt

    def test_includes_output_format(self):
        prompt = build_planning_prompt(["ctx"], "test", 1)
        assert "chart_type" in prompt
        assert "data_mapping" in prompt


# =============================================================================
# Code generation prompt
# =============================================================================


class TestCodeGenSystemPrompt:
    def test_has_matplotlib_role(self):
        assert "matplotlib" in CODEGEN_SYSTEM_PROMPT.lower()

    def test_no_savefig_instruction(self):
        assert "NOT" in CODEGEN_SYSTEM_PROMPT
        assert "savefig" in CODEGEN_SYSTEM_PROMPT

    def test_no_show_instruction(self):
        assert "show()" in CODEGEN_SYSTEM_PROMPT

    def test_no_backend_instruction(self):
        assert "backend" in CODEGEN_SYSTEM_PROMPT.lower()

    def test_tight_layout_instruction(self):
        assert "tight_layout" in CODEGEN_SYSTEM_PROMPT

    def test_self_contained_instruction(self):
        assert "self-contained" in CODEGEN_SYSTEM_PROMPT


class TestCodeGenUserPrompt:
    def test_includes_spec(self):
        prompt = build_codegen_prompt(
            spec_json='{"chart_type": "bar"}',
            data_dict="{'x': [1, 2]}",
            column_info="  - x: dtype=int64",
        )
        assert "bar" in prompt

    def test_includes_data(self):
        data = "{'region': ['North', 'South'], 'revenue': [100, 200]}"
        prompt = build_codegen_prompt(
            spec_json="{}",
            data_dict=data,
            column_info="  - region: dtype=object",
        )
        assert "North" in prompt
        assert "revenue" in prompt

    def test_includes_column_info(self):
        col_info = "  - revenue: dtype=float64"
        prompt = build_codegen_prompt(
            spec_json="{}",
            data_dict="{}",
            column_info=col_info,
        )
        assert "float64" in prompt

    def test_includes_renderer_type(self):
        prompt = build_codegen_prompt(
            spec_json="{}",
            data_dict="{}",
            column_info="",
            renderer_type="matplotlib",
        )
        assert "matplotlib" in prompt
