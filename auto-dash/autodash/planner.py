"""Analysis planning (MVP.3).

Accepts DataProfile + user questions, produces validated AnalysisStep(s).
Uses LLM to understand intent and map to structured analysis operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from autodash.models import (
    AggregationType,
    AnalysisStep,
    ColumnProfile,
    DataProfile,
    SemanticType,
)
from autodash.prompts.analysis_planning import SYSTEM_PROMPT, build_user_prompt
from plotlint.core.errors import LLMError, PlanningError
from plotlint.core.llm import LLMClient
from plotlint.core.parsing import parse_json_from_response

logger = logging.getLogger(__name__)


# =============================================================================
# Profile summary for LLM prompt
# =============================================================================


def _profile_summary(profile: DataProfile) -> str:
    """Generate concise profile summary for LLM prompt.

    Focuses on column names, types, and key statistics.
    Each LLM-calling module builds its own view of the profile;
    this one is tailored for planning (types and stats matter most).
    """
    lines = [
        f"Dataset: {profile.source_path} ({profile.file_format})",
        f"Rows: {profile.row_count:,}",
        f"Columns ({len(profile.columns)}):",
    ]

    for col in profile.columns:
        parts = [f"  - {col.name} ({col.semantic_type.value}"]

        if col.semantic_type == SemanticType.NUMERIC:
            if col.min is not None and col.max is not None:
                parts.append(f", range: {col.min}-{col.max}")
            if col.mean is not None:
                parts.append(f", mean: {col.mean:.2f}")
        elif col.semantic_type == SemanticType.CATEGORICAL:
            parts.append(f", {col.unique_count} unique")
            if col.top_values:
                top = [v["value"] for v in col.top_values[:5]]
                parts.append(f": {top}")
        elif col.semantic_type == SemanticType.DATETIME:
            if col.date_min and col.date_max:
                parts.append(f", {col.date_min} to {col.date_max}")
            if col.date_granularity:
                parts.append(f", {col.date_granularity}")

        if col.null_fraction > 0.05:
            parts.append(f", {col.null_fraction:.0%} null")

        parts.append(")")
        lines.append("".join(parts))

    return "\n".join(lines)


# =============================================================================
# Prompt construction
# =============================================================================


def build_planning_prompt(
    profile: DataProfile,
    questions: str,
    max_steps: int = 1,
) -> str:
    """Construct the LLM user prompt for analysis planning.

    Pure function: separated from LLM call for testability.
    """
    summary = _profile_summary(profile)
    return build_user_prompt(summary, questions, max_steps)


# =============================================================================
# Validation
# =============================================================================


_NUMERIC_AGGS = frozenset({
    AggregationType.SUM,
    AggregationType.MEAN,
    AggregationType.MEDIAN,
    AggregationType.MIN,
    AggregationType.MAX,
})

_NUMERIC_COMPATIBLE = frozenset({
    SemanticType.NUMERIC,
    SemanticType.BOOLEAN,
})


def _validate_step(
    step: AnalysisStep,
    profile: DataProfile,
) -> tuple[list[str], list[str]]:
    """Validate column references and semantic compatibility.

    Returns:
        (missing_columns, warnings)
        - missing_columns: columns referenced but not in profile
        - warnings: semantic mismatches (logged, not blocking)
    """
    valid_columns = profile.column_names()

    # Collect all referenced columns
    all_referenced = set(step.target_columns) | set(step.group_by_columns)
    if step.sort_by:
        all_referenced.add(step.sort_by)

    missing = [col for col in all_referenced if col not in valid_columns]

    # Semantic compatibility checks
    warnings: list[str] = []
    for col_name in step.target_columns:
        col = profile.get_column(col_name)
        if col is None:
            continue  # Already caught as missing
        warning = _check_semantic_compatibility(col, step.aggregation)
        if warning:
            warnings.append(warning)

    return missing, warnings


def _check_semantic_compatibility(
    col: ColumnProfile,
    agg: AggregationType,
) -> Optional[str]:
    """Check if aggregation makes sense for column type. Returns warning or None."""
    if agg in _NUMERIC_AGGS and col.semantic_type not in _NUMERIC_COMPATIBLE:
        return (
            f"{agg.value.upper()} on {col.semantic_type.value} column "
            f"'{col.name}' may not be meaningful"
        )

    if agg == AggregationType.CORRELATION:
        if col.semantic_type not in _NUMERIC_COMPATIBLE:
            return (
                f"CORRELATION requires numeric columns, "
                f"'{col.name}' is {col.semantic_type.value}"
            )

    if agg == AggregationType.TIME_SERIES:
        if col.semantic_type != SemanticType.DATETIME:
            return (
                f"TIME_SERIES expects datetime column, "
                f"'{col.name}' is {col.semantic_type.value}"
            )

    return None


# =============================================================================
# Response parsing
# =============================================================================


def parse_analysis_response(
    raw_response: str,
    profile: DataProfile,
    max_steps: int = 1,
) -> list[AnalysisStep]:
    """Parse and validate LLM response into AnalysisStep objects.

    Raises:
        PlanningError: If parsing or validation fails.
    """
    try:
        data = parse_json_from_response(raw_response)
    except ValueError as e:
        raise PlanningError(f"Failed to parse JSON from LLM response: {e}") from e

    if not isinstance(data, list):
        raise PlanningError(
            f"Expected JSON array of analysis steps, got {type(data).__name__}"
        )

    if len(data) == 0:
        raise PlanningError("LLM returned empty analysis plan")

    if len(data) > max_steps:
        logger.warning(
            "LLM returned %d steps, truncating to %d", len(data), max_steps
        )
        data = data[:max_steps]

    steps = []
    for idx, step_data in enumerate(data):
        try:
            step = _parse_single_step(step_data, profile, idx)
            steps.append(step)
        except (ValueError, KeyError, TypeError) as e:
            raise PlanningError(
                f"Invalid analysis step at index {idx}: {e}"
            ) from e

    return steps


def _parse_single_step(
    step_data: dict[str, Any],
    profile: DataProfile,
    index: int,
) -> AnalysisStep:
    """Parse and validate a single analysis step from LLM JSON.

    Raises:
        ValueError: If validation fails.
        KeyError: If required field missing.
    """
    # Required fields
    description = step_data["description"]
    target_columns = step_data["target_columns"]
    aggregation_str = step_data["aggregation"]

    # Validate aggregation type
    try:
        aggregation = AggregationType(aggregation_str)
    except ValueError:
        valid = [t.value for t in AggregationType]
        raise ValueError(
            f"Invalid aggregation type '{aggregation_str}'. Valid: {valid}"
        )

    # Validate list types
    if not isinstance(target_columns, list):
        raise ValueError(
            f"target_columns must be a list, got {type(target_columns).__name__}"
        )

    group_by_columns = step_data.get("group_by_columns", [])
    if not isinstance(group_by_columns, list):
        raise ValueError(
            f"group_by_columns must be a list, got {type(group_by_columns).__name__}"
        )

    # Optional fields
    filter_expression = step_data.get("filter_expression")
    sort_by = step_data.get("sort_by")
    limit = step_data.get("limit")
    rationale = step_data.get("rationale", "")

    step = AnalysisStep(
        description=description,
        target_columns=target_columns,
        aggregation=aggregation,
        group_by_columns=group_by_columns,
        filter_expression=filter_expression,
        sort_by=sort_by,
        limit=limit,
        rationale=rationale,
        step_index=index,
    )

    # Validate columns and semantics
    missing, warnings = _validate_step(step, profile)

    if missing:
        raise ValueError(
            f"References non-existent columns: {missing}. "
            f"Available: {profile.column_names()}"
        )

    for warning in warnings:
        logger.warning("Step %d: %s", index, warning)

    return step


# =============================================================================
# Main entry point
# =============================================================================


async def plan_analysis(
    profile: DataProfile,
    questions: str,
    llm_client: LLMClient,
    max_steps: int = 1,
) -> list[AnalysisStep]:
    """Plan analysis steps for the given data and questions.

    Main entry point for MVP.3. Returns list for forward-compatibility
    with DI-1.1 (multi-step planning).

    Args:
        profile: DataProfile from MVP.2.
        questions: User's natural language questions.
        llm_client: LLM client for planning.
        max_steps: Number of analysis steps to plan (MVP=1, DI-1.1=N).

    Raises:
        PlanningError: If planning or validation fails.
        LLMError: If LLM API call fails.
    """
    user_prompt = build_planning_prompt(profile, questions, max_steps)

    try:
        raw_response = await llm_client.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
            max_tokens=2048,
        )
    except Exception as e:
        raise LLMError(f"LLM call failed during analysis planning: {e}") from e

    steps = parse_analysis_response(raw_response, profile, max_steps)

    logger.info("Generated %d analysis step(s)", len(steps))
    return steps
