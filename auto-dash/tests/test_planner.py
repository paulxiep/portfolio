"""Tests for autodash.planner — analysis planning logic."""

import pytest

from autodash.models import (
    AggregationType,
    AnalysisStep,
    ColumnProfile,
    DataProfile,
    SemanticType,
)
from autodash.planner import (
    _validate_step,
    build_planning_prompt,
    parse_analysis_response,
    plan_analysis,
)
from plotlint.core.errors import LLMError, PlanningError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_profile():
    """DataProfile with numeric, categorical, and datetime columns."""
    return DataProfile(
        source_path="test.csv",
        row_count=100,
        file_format="csv",
        columns=[
            ColumnProfile(
                name="revenue",
                pandas_dtype="float64",
                semantic_type=SemanticType.NUMERIC,
                null_count=0,
                null_fraction=0.0,
                unique_count=95,
                cardinality_fraction=0.95,
                min=100.0,
                max=5000.0,
                mean=1250.5,
                median=1200.0,
                std=500.0,
            ),
            ColumnProfile(
                name="region",
                pandas_dtype="object",
                semantic_type=SemanticType.CATEGORICAL,
                null_count=0,
                null_fraction=0.0,
                unique_count=4,
                cardinality_fraction=0.04,
                top_values=[
                    {"value": "North", "count": 30},
                    {"value": "South", "count": 25},
                    {"value": "East", "count": 25},
                    {"value": "West", "count": 20},
                ],
            ),
            ColumnProfile(
                name="date",
                pandas_dtype="datetime64[ns]",
                semantic_type=SemanticType.DATETIME,
                null_count=0,
                null_fraction=0.0,
                unique_count=100,
                cardinality_fraction=1.0,
                date_min="2024-01-01",
                date_max="2024-12-31",
                date_granularity="daily",
            ),
            ColumnProfile(
                name="name",
                pandas_dtype="object",
                semantic_type=SemanticType.TEXT,
                null_count=5,
                null_fraction=0.05,
                unique_count=90,
                cardinality_fraction=0.9,
            ),
        ],
    )


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


# =============================================================================
# build_planning_prompt
# =============================================================================


class TestBuildPlanningPrompt:
    def test_includes_profile_data(self, sample_profile):
        prompt = build_planning_prompt(sample_profile, "test question", 1)
        assert "test.csv" in prompt
        assert "revenue" in prompt
        assert "region" in prompt
        assert "date" in prompt

    def test_includes_questions(self, sample_profile):
        questions = "What is the total revenue by region?"
        prompt = build_planning_prompt(sample_profile, questions, 1)
        assert questions in prompt

    def test_includes_max_steps(self, sample_profile):
        prompt = build_planning_prompt(sample_profile, "test", max_steps=3)
        assert "3" in prompt

    def test_includes_column_stats(self, sample_profile):
        prompt = build_planning_prompt(sample_profile, "test", 1)
        # Numeric: range and mean
        assert "100.0" in prompt
        assert "5000.0" in prompt
        # Categorical: unique count
        assert "4 unique" in prompt
        # Datetime: date range
        assert "2024-01-01" in prompt


# =============================================================================
# _validate_step
# =============================================================================


class TestValidateStep:
    def test_valid_step_no_errors(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.SUM,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(missing) == 0
        assert len(warnings) == 0

    def test_missing_target_column(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["nonexistent"],
            aggregation=AggregationType.SUM,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert "nonexistent" in missing

    def test_missing_group_by_column(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.GROUP_BY,
            group_by_columns=["missing_group"],
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert "missing_group" in missing

    def test_missing_sort_by_column(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.SUM,
            sort_by="missing_sort",
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert "missing_sort" in missing

    def test_sum_on_text_warns(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["name"],
            aggregation=AggregationType.SUM,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(missing) == 0
        assert len(warnings) == 1
        assert "SUM" in warnings[0]
        assert "name" in warnings[0]

    def test_mean_on_numeric_no_warning(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.MEAN,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(warnings) == 0

    def test_correlation_on_categorical_warns(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["region"],
            aggregation=AggregationType.CORRELATION,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(warnings) == 1
        assert "CORRELATION" in warnings[0]

    def test_time_series_on_non_datetime_warns(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.TIME_SERIES,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(warnings) == 1
        assert "TIME_SERIES" in warnings[0]

    def test_time_series_on_datetime_no_warning(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["date"],
            aggregation=AggregationType.TIME_SERIES,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(warnings) == 0

    def test_group_by_no_warning(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["revenue"],
            aggregation=AggregationType.GROUP_BY,
            group_by_columns=["region"],
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(missing) == 0
        assert len(warnings) == 0

    def test_count_on_any_type_no_warning(self, sample_profile):
        step = AnalysisStep(
            description="Test",
            target_columns=["name"],
            aggregation=AggregationType.COUNT,
        )
        missing, warnings = _validate_step(step, sample_profile)
        assert len(warnings) == 0


# =============================================================================
# parse_analysis_response
# =============================================================================


class TestParseAnalysisResponse:
    def test_valid_single_step(self, sample_profile):
        response = """[{
            "description": "Average revenue by region",
            "target_columns": ["revenue"],
            "aggregation": "group_by",
            "group_by_columns": ["region"],
            "filter_expression": null,
            "sort_by": "revenue",
            "limit": null,
            "rationale": "Regional performance"
        }]"""
        steps = parse_analysis_response(response, sample_profile, max_steps=1)
        assert len(steps) == 1
        assert steps[0].description == "Average revenue by region"
        assert steps[0].target_columns == ["revenue"]
        assert steps[0].aggregation == AggregationType.GROUP_BY
        assert steps[0].group_by_columns == ["region"]
        assert steps[0].sort_by == "revenue"

    def test_response_in_code_fence(self, sample_profile):
        response = """```json
[{
    "description": "Total revenue",
    "target_columns": ["revenue"],
    "aggregation": "sum",
    "group_by_columns": [],
    "rationale": "Sum all revenue"
}]
```"""
        steps = parse_analysis_response(response, sample_profile, max_steps=1)
        assert len(steps) == 1
        assert steps[0].aggregation == AggregationType.SUM

    def test_missing_required_field_raises(self, sample_profile):
        response = """[{
            "description": "Test",
            "target_columns": ["revenue"]
        }]"""
        with pytest.raises(PlanningError, match="Invalid analysis step"):
            parse_analysis_response(response, sample_profile, max_steps=1)

    def test_invalid_aggregation_type_raises(self, sample_profile):
        response = """[{
            "description": "Test",
            "target_columns": ["revenue"],
            "aggregation": "invalid_agg",
            "group_by_columns": []
        }]"""
        with pytest.raises(PlanningError, match="Invalid aggregation type"):
            parse_analysis_response(response, sample_profile, max_steps=1)

    def test_nonexistent_column_raises(self, sample_profile):
        response = """[{
            "description": "Test",
            "target_columns": ["nonexistent"],
            "aggregation": "sum",
            "group_by_columns": []
        }]"""
        with pytest.raises(PlanningError, match="non-existent columns"):
            parse_analysis_response(response, sample_profile, max_steps=1)

    def test_truncates_to_max_steps(self, sample_profile):
        response = """[
            {"description": "Step 1", "target_columns": ["revenue"], "aggregation": "sum", "group_by_columns": []},
            {"description": "Step 2", "target_columns": ["revenue"], "aggregation": "mean", "group_by_columns": []},
            {"description": "Step 3", "target_columns": ["revenue"], "aggregation": "median", "group_by_columns": []}
        ]"""
        steps = parse_analysis_response(response, sample_profile, max_steps=2)
        assert len(steps) == 2
        assert steps[0].step_index == 0
        assert steps[1].step_index == 1

    def test_empty_array_raises(self, sample_profile):
        with pytest.raises(PlanningError, match="empty analysis plan"):
            parse_analysis_response("[]", sample_profile, max_steps=1)

    def test_non_array_raises(self, sample_profile):
        response = """{"description": "Test"}"""
        with pytest.raises(PlanningError, match="Expected JSON array"):
            parse_analysis_response(response, sample_profile, max_steps=1)

    def test_invalid_json_raises(self, sample_profile):
        with pytest.raises(PlanningError, match="Failed to parse JSON"):
            parse_analysis_response("not json at all", sample_profile, max_steps=1)

    def test_target_columns_not_list_raises(self, sample_profile):
        response = """[{
            "description": "Test",
            "target_columns": "revenue",
            "aggregation": "sum",
            "group_by_columns": []
        }]"""
        with pytest.raises(PlanningError, match="target_columns must be a list"):
            parse_analysis_response(response, sample_profile, max_steps=1)

    def test_optional_fields_default(self, sample_profile):
        response = """[{
            "description": "Count all",
            "target_columns": ["revenue"],
            "aggregation": "count",
            "group_by_columns": []
        }]"""
        steps = parse_analysis_response(response, sample_profile, max_steps=1)
        assert steps[0].filter_expression is None
        assert steps[0].sort_by is None
        assert steps[0].limit is None
        assert steps[0].rationale == ""


# =============================================================================
# plan_analysis (integration with mock)
# =============================================================================


class TestPlanAnalysis:
    @pytest.mark.asyncio
    async def test_successful_planning(self, sample_profile):
        valid_response = """[{
            "description": "Revenue by region",
            "target_columns": ["revenue"],
            "aggregation": "group_by",
            "group_by_columns": ["region"],
            "filter_expression": null,
            "sort_by": null,
            "limit": null,
            "rationale": "Regional performance"
        }]"""
        client = MockLLMClient(valid_response)

        steps = await plan_analysis(
            sample_profile, "What is revenue by region?", client, max_steps=1
        )

        assert len(steps) == 1
        assert steps[0].description == "Revenue by region"
        assert len(client.calls) == 1
        assert "revenue" in client.calls[0]["user"]

    @pytest.mark.asyncio
    async def test_llm_failure_raises_llmerror(self, sample_profile):
        class FailingClient:
            async def complete(self, system, user, **kwargs):
                raise RuntimeError("API Error")

            async def complete_with_image(self, **kwargs):
                raise RuntimeError("API Error")

        with pytest.raises(LLMError, match="LLM call failed"):
            await plan_analysis(
                sample_profile, "test", FailingClient(), max_steps=1
            )

    @pytest.mark.asyncio
    async def test_invalid_response_raises_planningerror(self, sample_profile):
        client = MockLLMClient("this is not json")

        with pytest.raises(PlanningError):
            await plan_analysis(sample_profile, "test", client, max_steps=1)

    @pytest.mark.asyncio
    async def test_system_prompt_sent_to_llm(self, sample_profile):
        valid_response = """[{
            "description": "Count rows",
            "target_columns": ["revenue"],
            "aggregation": "count",
            "group_by_columns": []
        }]"""
        client = MockLLMClient(valid_response)

        await plan_analysis(sample_profile, "How many rows?", client, max_steps=1)

        assert "data analyst" in client.calls[0]["system"].lower()
