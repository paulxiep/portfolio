"""Tests for autodash.explorer — data exploration logic."""

import pandas as pd
import pytest

from autodash.models import (
    AggregationType,
    AnalysisStep,
    ColumnProfile,
    DataProfile,
    InsightResult,
    SemanticType,
)
from autodash.explorer import (
    _exploration_profile_summary,
    _normalize_result,
    _step_details,
    build_exploration_prompt,
    explore_step,
    summarize_result,
)
from plotlint.core.errors import ExplorationError


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_profile():
    """DataProfile with numeric and categorical columns."""
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
        ],
    )


@pytest.fixture
def sample_step():
    """A simple AnalysisStep for testing."""
    return AnalysisStep(
        description="Average revenue by region",
        target_columns=["revenue"],
        aggregation=AggregationType.GROUP_BY,
        group_by_columns=["region"],
        step_index=0,
    )


@pytest.fixture
def sample_df():
    """Small DataFrame for sandbox testing."""
    return pd.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0],
        "region": ["North", "South", "North", "South"],
    })


class MockLLMClient:
    """Mock LLM client supporting a list of responses for retry testing."""

    def __init__(self, responses: list[str] | str):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.calls: list[dict] = []
        self._call_index = 0

    async def complete(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user})
        response = self.responses[min(self._call_index, len(self.responses) - 1)]
        self._call_index += 1
        return response

    async def complete_with_image(self, **kwargs):
        return ""


# =============================================================================
# _exploration_profile_summary
# =============================================================================


class TestExplorationProfileSummary:
    def test_includes_column_dtypes(self, sample_profile):
        summary = _exploration_profile_summary(sample_profile)
        assert "float64" in summary
        assert "object" in summary

    def test_includes_semantic_types(self, sample_profile):
        summary = _exploration_profile_summary(sample_profile)
        assert "numeric" in summary
        assert "categorical" in summary

    def test_includes_numeric_range(self, sample_profile):
        summary = _exploration_profile_summary(sample_profile)
        assert "100.0" in summary
        assert "5000.0" in summary

    def test_includes_categorical_values(self, sample_profile):
        summary = _exploration_profile_summary(sample_profile)
        assert "North" in summary

    def test_includes_null_fraction_when_significant(self):
        profile = DataProfile(
            source_path="test.csv",
            row_count=100,
            file_format="csv",
            columns=[
                ColumnProfile(
                    name="col_with_nulls",
                    pandas_dtype="float64",
                    semantic_type=SemanticType.NUMERIC,
                    null_count=20,
                    null_fraction=0.2,
                    unique_count=80,
                    cardinality_fraction=0.8,
                ),
            ],
        )
        summary = _exploration_profile_summary(profile)
        assert "20.0% null" in summary


# =============================================================================
# _step_details
# =============================================================================


class TestStepDetails:
    def test_basic_step(self, sample_step):
        details = _step_details(sample_step)
        assert "revenue" in details
        assert "group_by" in details
        assert "region" in details

    def test_step_with_all_fields(self):
        step = AnalysisStep(
            description="Filter and sort",
            target_columns=["revenue"],
            aggregation=AggregationType.SUM,
            group_by_columns=["region"],
            filter_expression="revenue > 100",
            sort_by="revenue",
            limit=10,
            rationale="Top revenue regions",
        )
        details = _step_details(step)
        assert "revenue > 100" in details
        assert "Sort by: revenue" in details
        assert "Limit: 10" in details
        assert "Top revenue regions" in details

    def test_step_without_optional_fields(self):
        step = AnalysisStep(
            description="Count all",
            target_columns=["revenue"],
            aggregation=AggregationType.COUNT,
        )
        details = _step_details(step)
        assert "Group by" not in details
        assert "Filter" not in details
        assert "Sort by" not in details
        assert "Limit" not in details


# =============================================================================
# build_exploration_prompt
# =============================================================================


class TestBuildExplorationPrompt:
    def test_includes_step_description(self, sample_step, sample_profile):
        prompt = build_exploration_prompt(
            sample_step, sample_profile, "sample data"
        )
        assert "Average revenue by region" in prompt

    def test_includes_profile_data(self, sample_step, sample_profile):
        prompt = build_exploration_prompt(
            sample_step, sample_profile, "sample data"
        )
        assert "test.csv" in prompt
        assert "float64" in prompt

    def test_includes_sample_data(self, sample_step, sample_profile):
        prompt = build_exploration_prompt(
            sample_step, sample_profile, "   revenue  region\n0  100  North"
        )
        assert "revenue  region" in prompt

    def test_includes_error_on_retry(self, sample_step, sample_profile):
        prompt = build_exploration_prompt(
            sample_step,
            sample_profile,
            "sample data",
            previous_error="KeyError: 'bad_col'",
            previous_code="df['bad_col']",
        )
        assert "KeyError" in prompt
        assert "bad_col" in prompt
        assert "Fix the code" in prompt

    def test_no_error_block_on_first_attempt(self, sample_step, sample_profile):
        prompt = build_exploration_prompt(
            sample_step, sample_profile, "sample data"
        )
        assert "previous attempt failed" not in prompt.lower()


# =============================================================================
# _normalize_result
# =============================================================================


class TestNormalizeResult:
    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _normalize_result(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a"]

    def test_series_converted_to_dataframe(self):
        series = pd.Series([1, 2, 3], name="col")
        result = _normalize_result(series)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 1)

    def test_scalar_int_wrapped(self):
        result = _normalize_result(42)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0, 0] == 42

    def test_scalar_float_wrapped(self):
        result = _normalize_result(3.14)
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0, 0] == 3.14

    def test_scalar_string_wrapped(self):
        result = _normalize_result("hello")
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0, 0] == "hello"

    def test_none_raises_exploration_error(self):
        with pytest.raises(ExplorationError, match="unsupported type"):
            _normalize_result(None)

    def test_dict_raises_exploration_error(self):
        with pytest.raises(ExplorationError, match="unsupported type"):
            _normalize_result({"a": 1})

    def test_list_raises_exploration_error(self):
        with pytest.raises(ExplorationError, match="unsupported type"):
            _normalize_result([1, 2, 3])


# =============================================================================
# summarize_result
# =============================================================================


class TestSummarizeResult:
    def test_basic_summary(self, sample_step):
        df = pd.DataFrame({"region": ["North", "South"], "revenue": [200.0, 300.0]})
        summary = summarize_result(sample_step, df)
        assert "Average revenue by region" in summary
        assert "2 row(s)" in summary
        assert "2 column(s)" in summary

    def test_includes_numeric_stats(self, sample_step):
        df = pd.DataFrame({"region": ["North", "South"], "revenue": [200.0, 300.0]})
        summary = summarize_result(sample_step, df)
        assert "min=" in summary
        assert "max=" in summary
        assert "mean=" in summary

    def test_single_row_result(self, sample_step):
        df = pd.DataFrame({"total": [500.0]})
        summary = summarize_result(sample_step, df)
        assert "1 row(s)" in summary

    def test_empty_dataframe(self, sample_step):
        df = pd.DataFrame({"revenue": pd.Series([], dtype="float64")})
        summary = summarize_result(sample_step, df)
        assert "0 row(s)" in summary


# =============================================================================
# explore_step (integration tests with mock LLM)
# =============================================================================


class TestExploreStep:
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, sample_step, sample_profile, sample_df):
        code = (
            "```python\n"
            "import pandas as pd\n"
            "__result__ = df.groupby('region')['revenue'].mean().reset_index()\n"
            "```"
        )
        client = MockLLMClient(code)

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert isinstance(result, InsightResult)
        assert result.attempts == 1
        assert isinstance(result.result_df, pd.DataFrame)
        assert len(result.result_df) == 2  # North, South
        assert result.code_used is not None
        assert result.summary is not None
        assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_retry_on_runtime_error(self, sample_step, sample_profile, sample_df):
        bad_code = "```python\n__result__ = df['nonexistent_col'].sum()\n```"
        good_code = (
            "```python\n"
            "import pandas as pd\n"
            "__result__ = df.groupby('region')['revenue'].mean().reset_index()\n"
            "```"
        )
        client = MockLLMClient([bad_code, good_code])

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert result.attempts == 2
        assert len(client.calls) == 2

    @pytest.mark.asyncio
    async def test_retry_on_missing_result_variable(self, sample_step, sample_profile, sample_df):
        no_result = "```python\nx = df['revenue'].sum()\n```"
        good_code = (
            "```python\n"
            "__result__ = df.groupby('region')['revenue'].mean().reset_index()\n"
            "```"
        )
        client = MockLLMClient([no_result, good_code])

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_all_attempts_fail_raises(self, sample_step, sample_profile, sample_df):
        bad_code = "```python\n__result__ = df['nonexistent'].sum()\n```"
        client = MockLLMClient(bad_code)

        with pytest.raises(ExplorationError, match="after 2 attempts"):
            await explore_step(
                sample_step, sample_df, sample_profile, client, max_attempts=2
            )

    @pytest.mark.asyncio
    async def test_llm_call_failure_raises_immediately(self, sample_step, sample_profile, sample_df):
        class FailingClient:
            async def complete(self, system, user, **kwargs):
                raise RuntimeError("API Error")

            async def complete_with_image(self, **kwargs):
                raise RuntimeError("API Error")

        with pytest.raises(ExplorationError, match="LLM call failed"):
            await explore_step(
                sample_step, sample_df, sample_profile, FailingClient(), max_attempts=3
            )

    @pytest.mark.asyncio
    async def test_code_parse_failure_retried(self, sample_step, sample_profile, sample_df):
        prose = "I think we should analyze the revenue data by grouping it."
        good_code = (
            "```python\n"
            "__result__ = df.groupby('region')['revenue'].mean().reset_index()\n"
            "```"
        )
        client = MockLLMClient([prose, good_code])

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_series_result_accepted(self, sample_step, sample_profile, sample_df):
        code = "```python\n__result__ = df.groupby('region')['revenue'].mean()\n```"
        client = MockLLMClient(code)

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert isinstance(result.result_df, pd.DataFrame)
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_scalar_result_accepted(self, sample_step, sample_profile, sample_df):
        code = "```python\n__result__ = df['revenue'].sum()\n```"
        client = MockLLMClient(code)

        result = await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        assert isinstance(result.result_df, pd.DataFrame)
        assert result.result_df.shape == (1, 1)
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_error_context_includes_previous_code(self, sample_step, sample_profile, sample_df):
        bad_code = "```python\n__result__ = df['nonexistent_col'].sum()\n```"
        good_code = (
            "```python\n"
            "__result__ = df.groupby('region')['revenue'].mean().reset_index()\n"
            "```"
        )
        client = MockLLMClient([bad_code, good_code])

        await explore_step(
            sample_step, sample_df, sample_profile, client, max_attempts=3
        )

        # Second call should include previous error context
        assert len(client.calls) == 2
        second_prompt = client.calls[1]["user"]
        assert "nonexistent_col" in second_prompt

    @pytest.mark.asyncio
    async def test_max_attempts_one(self, sample_step, sample_profile, sample_df):
        bad_code = "```python\n__result__ = df['nonexistent'].sum()\n```"
        client = MockLLMClient(bad_code)

        with pytest.raises(ExplorationError):
            await explore_step(
                sample_step, sample_df, sample_profile, client, max_attempts=1
            )

        assert len(client.calls) == 1
