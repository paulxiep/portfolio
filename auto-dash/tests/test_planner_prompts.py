"""Tests for autodash.prompts.analysis_planning — prompt templates."""

from autodash.models import AggregationType
from autodash.prompts.analysis_planning import (
    OUTPUT_FORMAT,
    SYSTEM_PROMPT,
    build_user_prompt,
)


class TestSystemPrompt:
    def test_exists_and_nonempty(self):
        assert len(SYSTEM_PROMPT) > 0

    def test_contains_role(self):
        assert "data analyst" in SYSTEM_PROMPT.lower()

    def test_contains_all_aggregation_types(self):
        for agg in AggregationType:
            assert agg.value in SYSTEM_PROMPT


class TestOutputFormat:
    def test_exists_and_nonempty(self):
        assert len(OUTPUT_FORMAT) > 0

    def test_contains_required_fields(self):
        assert "description" in OUTPUT_FORMAT
        assert "target_columns" in OUTPUT_FORMAT
        assert "aggregation" in OUTPUT_FORMAT
        assert "group_by_columns" in OUTPUT_FORMAT

    def test_contains_optional_fields(self):
        assert "filter_expression" in OUTPUT_FORMAT
        assert "sort_by" in OUTPUT_FORMAT
        assert "limit" in OUTPUT_FORMAT
        assert "rationale" in OUTPUT_FORMAT


class TestBuildUserPrompt:
    def test_includes_all_parts(self):
        profile_summary = "Dataset: test.csv\nRows: 100"
        questions = "What is the average revenue?"
        max_steps = 1

        prompt = build_user_prompt(profile_summary, questions, max_steps)

        assert profile_summary in prompt
        assert questions in prompt
        assert str(max_steps) in prompt

    def test_includes_output_format(self):
        prompt = build_user_prompt("profile", "questions", 1)
        # Output format should be embedded
        assert "description" in prompt
        assert "target_columns" in prompt
        assert "aggregation" in prompt

    def test_max_steps_in_instruction(self):
        prompt = build_user_prompt("profile", "questions", 5)
        assert "5" in prompt
        assert "analysis step" in prompt.lower()
