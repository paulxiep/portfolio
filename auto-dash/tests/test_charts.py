"""Tests for autodash.charts — chart planning and code generation logic."""

import math

import pandas as pd
import pytest

from autodash.charts import (
    _serialize_df_for_prompt,
    _to_python_native,
    _validate_data_mapping,
    build_chart_planning_prompt,
    build_code_generation_prompt,
    generate_chart_code,
    parse_chart_specs,
    plan_and_generate,
    plan_charts,
)
from autodash.models import (
    AggregationType,
    AnalysisStep,
    ChartPlan,
    ChartPriority,
    ChartSpec,
    ChartType,
    DataMapping,
    InsightResult,
    RendererType,
)
from plotlint.core.errors import ChartGenerationError, LLMError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_step():
    return AnalysisStep(
        description="Revenue by region",
        target_columns=["revenue"],
        aggregation=AggregationType.GROUP_BY,
        group_by_columns=["region"],
        step_index=0,
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [37500.0, 31250.0, 31250.0, 25000.0],
    })


@pytest.fixture
def sample_insight(sample_step, sample_df):
    return InsightResult(
        step=sample_step,
        result_df=sample_df,
        summary="Revenue grouped by region: 4 rows, 2 columns.",
        code_used="result = df.groupby('region')['revenue'].sum().reset_index()",
        attempts=1,
    )


@pytest.fixture
def two_insights(sample_insight):
    """Two insights for multi-chart testing."""
    step2 = AnalysisStep(
        description="Monthly trend",
        target_columns=["revenue"],
        aggregation=AggregationType.TIME_SERIES,
        step_index=1,
    )
    df2 = pd.DataFrame({
        "month": ["Jan", "Feb", "Mar"],
        "revenue": [10000.0, 12000.0, 15000.0],
    })
    insight2 = InsightResult(
        step=step2,
        result_df=df2,
        summary="Monthly revenue trend.",
        code_used="result = df.groupby('month')['revenue'].sum()",
        attempts=1,
    )
    return [sample_insight, insight2]


class MockLLMClient:
    """Mock LLM client that returns a fixed response."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    async def complete(
        self,
        system: str,
        user: str,
        model=None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response

    async def complete_with_image(self, **kwargs) -> str:
        return self.response


class SequentialMockLLMClient:
    """Mock LLM client that returns different responses in sequence."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_index = 0
        self.calls: list[dict] = []

    async def complete(
        self,
        system: str,
        user: str,
        model=None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        self.calls.append({"system": system, "user": user})
        response = self.responses[self.call_index]
        self.call_index += 1
        return response

    async def complete_with_image(self, **kwargs) -> str:
        return self.responses[0]


# =============================================================================
# _to_python_native
# =============================================================================


class TestToPythonNative:
    def test_none(self):
        assert _to_python_native(None) is None

    def test_nan(self):
        assert _to_python_native(float("nan")) is None

    def test_inf(self):
        assert _to_python_native(float("inf")) is None

    def test_int(self):
        assert _to_python_native(42) == 42

    def test_float(self):
        assert _to_python_native(3.14) == 3.14

    def test_string(self):
        assert _to_python_native("hello") == "hello"

    def test_bool(self):
        assert _to_python_native(True) is True

    def test_datetime(self):
        import datetime
        dt = datetime.datetime(2024, 1, 15, 10, 30)
        result = _to_python_native(dt)
        assert "2024-01-15" in result

    def test_numpy_int(self):
        import numpy as np
        result = _to_python_native(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_unsupported_type(self):
        result = _to_python_native([1, 2, 3])
        assert result == "[1, 2, 3]"


# =============================================================================
# _serialize_df_for_prompt
# =============================================================================


class TestSerializeDfForPrompt:
    def test_simple_dataframe(self, sample_df):
        data_str, col_info = _serialize_df_for_prompt(sample_df)
        assert "region" in data_str
        assert "North" in data_str
        assert "37500.0" in data_str
        assert "revenue" in data_str

    def test_column_info_includes_dtypes(self, sample_df):
        _, col_info = _serialize_df_for_prompt(sample_df)
        assert "region" in col_info
        assert "revenue" in col_info
        assert "dtype=" in col_info

    def test_nan_handling(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        data_str, _ = _serialize_df_for_prompt(df)
        assert "None" in data_str
        assert "nan" not in data_str.lower()

    def test_truncation(self):
        df = pd.DataFrame({"x": range(100)})
        data_str, _ = _serialize_df_for_prompt(df, max_rows=5)
        # Should only have 5 values
        assert "4" in data_str
        # Value 99 should NOT be present (truncated)
        assert "99" not in data_str

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": [], "y": []})
        data_str, col_info = _serialize_df_for_prompt(df)
        assert "x" in data_str
        assert "y" in data_str

    def test_datetime_columns(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "value": [10, 20],
        })
        data_str, _ = _serialize_df_for_prompt(df)
        assert "2024-01-01" in data_str


# =============================================================================
# _validate_data_mapping
# =============================================================================


class TestValidateDataMapping:
    def test_bar_valid(self):
        mapping = DataMapping(x="region", y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.BAR, ["region", "revenue"])
        assert len(errors) == 0

    def test_bar_missing_y(self):
        mapping = DataMapping(x="region")
        errors = _validate_data_mapping(mapping, ChartType.BAR, ["region", "revenue"])
        assert any("requires 'y'" in e for e in errors)

    def test_bar_missing_x(self):
        mapping = DataMapping(y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.BAR, ["region", "revenue"])
        assert any("requires 'x'" in e for e in errors)

    def test_line_valid(self):
        mapping = DataMapping(x="date", y="value")
        errors = _validate_data_mapping(mapping, ChartType.LINE, ["date", "value"])
        assert len(errors) == 0

    def test_scatter_valid(self):
        mapping = DataMapping(x="height", y="weight")
        errors = _validate_data_mapping(mapping, ChartType.SCATTER, ["height", "weight"])
        assert len(errors) == 0

    def test_pie_valid(self):
        mapping = DataMapping(values="revenue", categories="region")
        errors = _validate_data_mapping(mapping, ChartType.PIE, ["region", "revenue"])
        assert len(errors) == 0

    def test_pie_missing_values(self):
        mapping = DataMapping(categories="region")
        errors = _validate_data_mapping(mapping, ChartType.PIE, ["region", "revenue"])
        assert any("requires 'values'" in e for e in errors)

    def test_pie_missing_categories(self):
        mapping = DataMapping(values="revenue")
        errors = _validate_data_mapping(mapping, ChartType.PIE, ["region", "revenue"])
        assert any("requires 'categories'" in e for e in errors)

    def test_histogram_x_only(self):
        mapping = DataMapping(x="revenue")
        errors = _validate_data_mapping(mapping, ChartType.HISTOGRAM, ["revenue"])
        assert len(errors) == 0

    def test_histogram_y_only(self):
        mapping = DataMapping(y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.HISTOGRAM, ["revenue"])
        assert len(errors) == 0

    def test_histogram_neither(self):
        mapping = DataMapping()
        errors = _validate_data_mapping(mapping, ChartType.HISTOGRAM, ["revenue"])
        assert any("requires 'x' or 'y'" in e for e in errors)

    def test_grouped_bar_needs_color(self):
        mapping = DataMapping(x="region", y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.GROUPED_BAR, ["region", "revenue"])
        assert any("requires 'color'" in e for e in errors)

    def test_grouped_bar_valid(self):
        mapping = DataMapping(x="region", y="revenue", color="category")
        errors = _validate_data_mapping(
            mapping, ChartType.GROUPED_BAR, ["region", "revenue", "category"]
        )
        assert len(errors) == 0

    def test_stacked_bar_needs_color(self):
        mapping = DataMapping(x="region", y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.STACKED_BAR, ["region", "revenue"])
        assert any("requires 'color'" in e for e in errors)

    def test_box_y_only(self):
        mapping = DataMapping(y="revenue")
        errors = _validate_data_mapping(mapping, ChartType.BOX, ["revenue"])
        assert len(errors) == 0

    def test_box_missing_y(self):
        mapping = DataMapping(x="region")
        errors = _validate_data_mapping(mapping, ChartType.BOX, ["region"])
        assert any("requires 'y'" in e for e in errors)

    def test_heatmap_valid(self):
        mapping = DataMapping(x="col_a", y="col_b")
        errors = _validate_data_mapping(mapping, ChartType.HEATMAP, ["col_a", "col_b"])
        assert len(errors) == 0

    def test_area_valid(self):
        mapping = DataMapping(x="date", y="value")
        errors = _validate_data_mapping(mapping, ChartType.AREA, ["date", "value"])
        assert len(errors) == 0

    def test_nonexistent_column(self):
        mapping = DataMapping(x="region", y="nonexistent")
        errors = _validate_data_mapping(mapping, ChartType.BAR, ["region", "revenue"])
        assert any("non-existent column" in e for e in errors)
        assert any("nonexistent" in e for e in errors)

    def test_multiple_errors(self):
        mapping = DataMapping()  # No fields set
        errors = _validate_data_mapping(mapping, ChartType.BAR, ["region"])
        assert len(errors) == 2  # missing x AND missing y


# =============================================================================
# parse_chart_specs
# =============================================================================


class TestParseChartSpecs:
    def test_valid_single_spec(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "title": "Revenue by Region",
            "data_mapping": {
                "x": "region",
                "y": "revenue",
                "color": null,
                "size": null,
                "label": null,
                "values": null,
                "categories": null
            },
            "priority": "primary",
            "x_label": "Region",
            "y_label": "Revenue ($)",
            "subtitle": null,
            "source_step_index": 0
        }]"""
        specs = parse_chart_specs(response, [sample_insight], max_charts=1)
        assert len(specs) == 1
        assert specs[0].chart_type == ChartType.BAR
        assert specs[0].title == "Revenue by Region"
        assert specs[0].data_mapping.x == "region"
        assert specs[0].data_mapping.y == "revenue"
        assert specs[0].priority == ChartPriority.PRIMARY
        assert specs[0].x_label == "Region"
        assert specs[0].y_label == "Revenue ($)"
        assert specs[0].source_step_index == 0

    def test_spec_in_code_fence(self, sample_insight):
        response = """```json
[{
    "chart_type": "bar",
    "title": "Test Chart",
    "data_mapping": {"x": "region", "y": "revenue"},
    "source_step_index": 0
}]
```"""
        specs = parse_chart_specs(response, [sample_insight], max_charts=1)
        assert len(specs) == 1
        assert specs[0].chart_type == ChartType.BAR

    def test_invalid_chart_type_raises(self, sample_insight):
        response = """[{
            "chart_type": "invalid_type",
            "title": "Test",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        with pytest.raises(ChartGenerationError, match="Invalid chart type"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_missing_title_raises(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        with pytest.raises(ChartGenerationError, match="Invalid chart spec"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_empty_title_raises(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "title": "",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        with pytest.raises(ChartGenerationError, match="non-empty string"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_invalid_column_reference_raises(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "title": "Test",
            "data_mapping": {"x": "region", "y": "nonexistent"},
            "source_step_index": 0
        }]"""
        with pytest.raises(ChartGenerationError, match="non-existent column"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_invalid_source_step_index_raises(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "title": "Test",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 5
        }]"""
        with pytest.raises(ChartGenerationError, match="out of range"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_empty_array_raises(self, sample_insight):
        with pytest.raises(ChartGenerationError, match="empty chart spec"):
            parse_chart_specs("[]", [sample_insight], max_charts=1)

    def test_non_array_raises(self, sample_insight):
        response = """{"chart_type": "bar"}"""
        with pytest.raises(ChartGenerationError, match="Expected JSON array"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_invalid_json_raises(self, sample_insight):
        with pytest.raises(ChartGenerationError, match="Failed to parse JSON"):
            parse_chart_specs("not json at all", [sample_insight], max_charts=1)

    def test_truncates_to_max_charts(self, sample_insight):
        response = """[
            {"chart_type": "bar", "title": "Chart 1", "data_mapping": {"x": "region", "y": "revenue"}, "source_step_index": 0},
            {"chart_type": "line", "title": "Chart 2", "data_mapping": {"x": "region", "y": "revenue"}, "source_step_index": 0}
        ]"""
        specs = parse_chart_specs(response, [sample_insight], max_charts=1)
        assert len(specs) == 1
        assert specs[0].title == "Chart 1"

    def test_optional_fields_default(self, sample_insight):
        response = """[{
            "chart_type": "bar",
            "title": "Simple Chart",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        specs = parse_chart_specs(response, [sample_insight], max_charts=1)
        assert specs[0].priority == ChartPriority.PRIMARY
        assert specs[0].x_label is None
        assert specs[0].y_label is None
        assert specs[0].subtitle is None

    def test_data_mapping_validation_fails_raises(self, sample_insight):
        response = """[{
            "chart_type": "pie",
            "title": "Test Pie",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        with pytest.raises(ChartGenerationError, match="Data mapping validation failed"):
            parse_chart_specs(response, [sample_insight], max_charts=1)

    def test_multiple_insights_correct_index(self, two_insights):
        response = """[{
            "chart_type": "line",
            "title": "Monthly Trend",
            "data_mapping": {"x": "month", "y": "revenue"},
            "source_step_index": 1
        }]"""
        specs = parse_chart_specs(response, two_insights, max_charts=1)
        assert specs[0].source_step_index == 1
        assert specs[0].data_mapping.x == "month"


# =============================================================================
# build_chart_planning_prompt
# =============================================================================


class TestBuildChartPlanningPrompt:
    def test_includes_insight_context(self, sample_insight):
        prompt = build_chart_planning_prompt([sample_insight], "test", 1)
        assert "Revenue by region" in prompt
        assert "region" in prompt
        assert "revenue" in prompt

    def test_includes_questions(self, sample_insight):
        questions = "What is the revenue breakdown?"
        prompt = build_chart_planning_prompt([sample_insight], questions, 1)
        assert questions in prompt

    def test_includes_max_charts(self, sample_insight):
        prompt = build_chart_planning_prompt([sample_insight], "test", max_charts=3)
        assert "3" in prompt


# =============================================================================
# build_code_generation_prompt
# =============================================================================


class TestBuildCodeGenerationPrompt:
    def test_includes_spec_details(self, sample_insight):
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Revenue by Region",
        )
        prompt = build_code_generation_prompt(spec, sample_insight)
        assert "bar" in prompt
        assert "Revenue by Region" in prompt

    def test_includes_data_dict(self, sample_insight):
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Test",
        )
        prompt = build_code_generation_prompt(spec, sample_insight)
        assert "North" in prompt
        assert "37500" in prompt

    def test_includes_column_info(self, sample_insight):
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Test",
        )
        prompt = build_code_generation_prompt(spec, sample_insight)
        assert "dtype=" in prompt

    def test_respects_max_rows(self):
        step = AnalysisStep(
            description="Test",
            target_columns=["x"],
            aggregation=AggregationType.COUNT,
            step_index=0,
        )
        big_df = pd.DataFrame({"x": range(100), "y": range(100)})
        insight = InsightResult(
            step=step, result_df=big_df, summary="test",
            code_used="test", attempts=1,
        )
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="x", y="y"),
            title="Test",
        )
        prompt = build_code_generation_prompt(spec, insight, max_rows=5)
        # Value 99 should not be in the data (truncated at 5 rows)
        assert "99" not in prompt


# =============================================================================
# plan_charts (async, mock LLM)
# =============================================================================


class TestPlanCharts:
    @pytest.mark.asyncio
    async def test_successful_planning(self, sample_insight):
        valid_response = """[{
            "chart_type": "bar",
            "title": "Revenue by Region",
            "data_mapping": {"x": "region", "y": "revenue"},
            "priority": "primary",
            "source_step_index": 0
        }]"""
        client = MockLLMClient(valid_response)

        specs = await plan_charts(
            [sample_insight], "What is revenue by region?", client, max_charts=1
        )

        assert len(specs) == 1
        assert specs[0].chart_type == ChartType.BAR
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_raises(self, sample_insight):
        class FailingClient:
            async def complete(self, system, user, **kwargs):
                raise RuntimeError("API Error")

            async def complete_with_image(self, **kwargs):
                raise RuntimeError("API Error")

        with pytest.raises(LLMError, match="LLM call failed"):
            await plan_charts([sample_insight], "test", FailingClient(), max_charts=1)

    @pytest.mark.asyncio
    async def test_invalid_response_raises(self, sample_insight):
        client = MockLLMClient("this is not json")

        with pytest.raises(ChartGenerationError):
            await plan_charts([sample_insight], "test", client, max_charts=1)

    @pytest.mark.asyncio
    async def test_system_prompt_sent(self, sample_insight):
        valid_response = """[{
            "chart_type": "bar",
            "title": "Test",
            "data_mapping": {"x": "region", "y": "revenue"},
            "source_step_index": 0
        }]"""
        client = MockLLMClient(valid_response)

        await plan_charts([sample_insight], "test", client, max_charts=1)

        assert "visualization" in client.calls[0]["system"].lower()


# =============================================================================
# generate_chart_code (async, mock LLM)
# =============================================================================


class TestGenerateChartCode:
    @pytest.mark.asyncio
    async def test_successful_generation(self, sample_insight):
        code_response = """```python
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({'region': ['North', 'South'], 'revenue': [37500, 31250]})
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(data['region'], data['revenue'])
plt.tight_layout()
```"""
        client = MockLLMClient(code_response)
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Revenue by Region",
        )

        plan = await generate_chart_code(spec, sample_insight, client)

        assert isinstance(plan, ChartPlan)
        assert plan.spec is spec
        assert "plt.subplots" in plan.code
        assert plan.renderer_type == RendererType.MATPLOTLIB

    @pytest.mark.asyncio
    async def test_llm_failure_raises(self, sample_insight):
        class FailingClient:
            async def complete(self, system, user, **kwargs):
                raise RuntimeError("API Error")

            async def complete_with_image(self, **kwargs):
                raise RuntimeError("API Error")

        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Test",
        )

        with pytest.raises(LLMError, match="LLM call failed"):
            await generate_chart_code(spec, sample_insight, FailingClient())

    @pytest.mark.asyncio
    async def test_code_parse_failure_raises(self, sample_insight):
        client = MockLLMClient("This response has no code at all, just prose.")
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Test",
        )

        with pytest.raises(ChartGenerationError, match="Failed to parse code"):
            await generate_chart_code(spec, sample_insight, client)

    @pytest.mark.asyncio
    async def test_system_prompt_sent(self, sample_insight):
        code_response = """```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar(['a'], [1])
plt.tight_layout()
```"""
        client = MockLLMClient(code_response)
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            data_mapping=DataMapping(x="region", y="revenue"),
            title="Test",
        )

        await generate_chart_code(spec, sample_insight, client)

        assert "matplotlib" in client.calls[0]["system"].lower()


# =============================================================================
# plan_and_generate (async, mock LLM)
# =============================================================================


class TestPlanAndGenerate:
    @pytest.mark.asyncio
    async def test_end_to_end_single_chart(self, sample_insight):
        planning_response = """[{
            "chart_type": "bar",
            "title": "Revenue by Region",
            "data_mapping": {"x": "region", "y": "revenue"},
            "priority": "primary",
            "source_step_index": 0
        }]"""
        code_response = """```python
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame({'region': ['North', 'South'], 'revenue': [37500, 31250]})
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(data['region'], data['revenue'])
plt.tight_layout()
```"""
        client = SequentialMockLLMClient([planning_response, code_response])

        plans = await plan_and_generate(
            [sample_insight], "What is revenue by region?", client, max_charts=1
        )

        assert len(plans) == 1
        assert plans[0].spec.chart_type == ChartType.BAR
        assert "plt.subplots" in plans[0].code
        assert plans[0].renderer_type == RendererType.MATPLOTLIB
        # Should have made 2 LLM calls (planning + code gen)
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_planning_failure_propagates(self, sample_insight):
        client = MockLLMClient("not json")

        with pytest.raises(ChartGenerationError):
            await plan_and_generate([sample_insight], "test", client, max_charts=1)
