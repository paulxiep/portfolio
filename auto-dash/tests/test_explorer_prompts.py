"""Tests for autodash.prompts.data_exploration — prompt templates."""

from autodash.prompts.data_exploration import (
    ERROR_RETRY_BLOCK,
    SYSTEM_PROMPT,
    build_user_prompt,
)


class TestSystemPrompt:
    def test_exists_and_nonempty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_mentions_pandas(self):
        assert "pandas" in SYSTEM_PROMPT.lower()

    def test_mentions_result_variable(self):
        assert "__result__" in SYSTEM_PROMPT

    def test_mentions_df_variable(self):
        assert "df" in SYSTEM_PROMPT


class TestErrorRetryBlock:
    def test_has_placeholders(self):
        assert "{error_type}" in ERROR_RETRY_BLOCK
        assert "{error_message}" in ERROR_RETRY_BLOCK
        assert "{previous_code}" in ERROR_RETRY_BLOCK


class TestBuildUserPrompt:
    def test_includes_all_parts(self):
        prompt = build_user_prompt(
            step_description="Average revenue by region",
            step_details="Target columns: ['revenue']\nAggregation: group_by",
            profile_summary="Dataset: test.csv (csv)\nRows: 100",
            sample_data="   revenue  region\n0  100     North",
        )
        assert "Average revenue by region" in prompt
        assert "Target columns" in prompt
        assert "test.csv" in prompt
        assert "revenue" in prompt
        assert "__result__" in prompt

    def test_includes_error_on_retry(self):
        prompt = build_user_prompt(
            step_description="Test",
            step_details="details",
            profile_summary="profile",
            sample_data="data",
            previous_error="KeyError: 'missing_col'",
            previous_code="df['missing_col']",
        )
        assert "KeyError" in prompt
        assert "missing_col" in prompt
        assert "Fix the code" in prompt

    def test_no_error_block_on_first_attempt(self):
        prompt = build_user_prompt(
            step_description="Test",
            step_details="details",
            profile_summary="profile",
            sample_data="data",
        )
        assert "previous attempt failed" not in prompt.lower()
