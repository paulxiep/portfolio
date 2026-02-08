"""Chart planning and code generation (MVP.5).

Accepts InsightResult(s) + user questions, uses LLM to produce
renderer-agnostic ChartSpec(s), then generates matplotlib code.

Two-step process:
  1. plan_charts()  → list[ChartSpec]   (LLM → JSON)
  2. generate_chart_code() → ChartPlan  (LLM → Python code)

Kept separate so DI-1.3 (multi-chart) can intervene between steps.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict
from typing import Any, Optional

import pandas as pd

from autodash.models import (
    ChartPlan,
    ChartPriority,
    ChartSpec,
    ChartType,
    DataMapping,
    InsightResult,
    RendererType,
)
from autodash.prompts.chart_planning import SYSTEM_PROMPT as PLANNING_SYSTEM_PROMPT
from autodash.prompts.chart_planning import build_user_prompt as build_planning_user_prompt
from autodash.prompts.code_generation import SYSTEM_PROMPT as CODEGEN_SYSTEM_PROMPT
from autodash.prompts.code_generation import build_user_prompt as build_codegen_user_prompt
from plotlint.core.errors import ChartGenerationError, LLMError
from plotlint.core.llm import LLMClient
from plotlint.core.parsing import parse_code_from_response, parse_json_from_response

logger = logging.getLogger(__name__)


# =============================================================================
# Data serialization
# =============================================================================


def _serialize_df_for_prompt(
    result_df: pd.DataFrame,
    max_rows: int = 50,
) -> tuple[str, str]:
    """Serialize DataFrame as Python dict literal for code embedding.

    Returns:
        (data_dict_str, column_info_str)
        - data_dict_str: Python dict literal for pd.DataFrame({...})
        - column_info_str: column names + dtypes for LLM context

    Truncates to max_rows with logged warning.
    Handles NaN → None, datetime → ISO string, numpy types → Python natives.
    """
    df = result_df
    if len(df) > max_rows:
        logger.warning(
            "DataFrame has %d rows, truncating to %d for inline embedding",
            len(df), max_rows,
        )
        df = df.head(max_rows)

    data_dict: dict[str, list[Any]] = {}
    for col in df.columns:
        values = []
        for val in df[col]:
            values.append(_to_python_native(val))
        data_dict[str(col)] = values

    data_dict_str = repr(data_dict)

    # Column info for LLM context
    info_lines = []
    for col in result_df.columns:
        info_lines.append(f"  - {col}: dtype={result_df[col].dtype}")
    column_info_str = "\n".join(info_lines)

    return data_dict_str, column_info_str


def _to_python_native(val: Any) -> Any:
    """Convert a single value to a JSON/Python-safe native type."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if hasattr(val, "isoformat"):  # datetime-like
        return val.isoformat()
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


# =============================================================================
# Prompt construction (pure functions for testability)
# =============================================================================


def build_chart_planning_prompt(
    insights: list[InsightResult],
    questions: str,
    max_charts: int = 1,
) -> str:
    """Construct the LLM user prompt for chart spec generation.

    Pure function: separated from LLM call for testability.
    """
    contexts = [insight.to_prompt_context() for insight in insights]
    return build_planning_user_prompt(
        insight_contexts=contexts,
        questions=questions,
        max_charts=max_charts,
    )


def build_code_generation_prompt(
    spec: ChartSpec,
    insight: InsightResult,
    renderer_type: RendererType = RendererType.MATPLOTLIB,
    max_rows: int = 50,
) -> str:
    """Construct the LLM user prompt for chart code generation.

    Pure function: separated from LLM call for testability.
    """
    # Serialize spec as JSON (excluding default_factory fields that are empty)
    spec_dict = asdict(spec)
    spec_json = json.dumps(spec_dict, indent=2, default=str)

    # Serialize data
    data_dict_str, column_info_str = _serialize_df_for_prompt(
        insight.result_df, max_rows=max_rows
    )

    return build_codegen_user_prompt(
        spec_json=spec_json,
        data_dict=data_dict_str,
        column_info=column_info_str,
        renderer_type=renderer_type.value,
    )


# =============================================================================
# Validation
# =============================================================================


# Chart types that require x AND y
_REQUIRE_XY = frozenset({
    ChartType.BAR,
    ChartType.LINE,
    ChartType.AREA,
    ChartType.SCATTER,
    ChartType.HEATMAP,
})

# Chart types that require x, y, AND color
_REQUIRE_XY_COLOR = frozenset({
    ChartType.GROUPED_BAR,
    ChartType.STACKED_BAR,
})


def _validate_data_mapping(
    mapping: DataMapping,
    chart_type: ChartType,
    available_columns: list[str],
) -> list[str]:
    """Validate mapping completeness per chart type + column existence.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Structural validation per chart type
    if chart_type in _REQUIRE_XY:
        if not mapping.x:
            errors.append(f"{chart_type.value} chart requires 'x' in data_mapping")
        if not mapping.y:
            errors.append(f"{chart_type.value} chart requires 'y' in data_mapping")

    elif chart_type in _REQUIRE_XY_COLOR:
        if not mapping.x:
            errors.append(f"{chart_type.value} chart requires 'x' in data_mapping")
        if not mapping.y:
            errors.append(f"{chart_type.value} chart requires 'y' in data_mapping")
        if not mapping.color:
            errors.append(f"{chart_type.value} chart requires 'color' in data_mapping")

    elif chart_type == ChartType.PIE:
        if not mapping.values:
            errors.append("pie chart requires 'values' in data_mapping")
        if not mapping.categories:
            errors.append("pie chart requires 'categories' in data_mapping")

    elif chart_type == ChartType.HISTOGRAM:
        if not mapping.x and not mapping.y:
            errors.append("histogram chart requires 'x' or 'y' in data_mapping")

    elif chart_type == ChartType.BOX:
        if not mapping.y:
            errors.append("box chart requires 'y' in data_mapping")

    # Column existence validation
    for field_name in ("x", "y", "color", "size", "label", "values", "categories"):
        col_name = getattr(mapping, field_name)
        if col_name is not None and col_name not in available_columns:
            errors.append(
                f"data_mapping.{field_name} references non-existent column "
                f"'{col_name}'. Available: {available_columns}"
            )

    return errors


# =============================================================================
# Response parsing
# =============================================================================


def parse_chart_specs(
    raw_response: str,
    insights: list[InsightResult],
    max_charts: int = 1,
) -> list[ChartSpec]:
    """Parse and validate LLM response into ChartSpec objects.

    Raises:
        ChartGenerationError: If parsing or validation fails.
    """
    try:
        data = parse_json_from_response(raw_response)
    except ValueError as e:
        raise ChartGenerationError(
            f"Failed to parse JSON from LLM response: {e}"
        ) from e

    if not isinstance(data, list):
        raise ChartGenerationError(
            f"Expected JSON array of chart specs, got {type(data).__name__}"
        )

    if len(data) == 0:
        raise ChartGenerationError("LLM returned empty chart spec list")

    if len(data) > max_charts:
        logger.warning(
            "LLM returned %d specs, truncating to %d", len(data), max_charts
        )
        data = data[:max_charts]

    specs = []
    for idx, spec_data in enumerate(data):
        try:
            spec = _parse_single_spec(spec_data, insights, idx)
            specs.append(spec)
        except (ValueError, KeyError, TypeError) as e:
            raise ChartGenerationError(
                f"Invalid chart spec at index {idx}: {e}"
            ) from e

    return specs


def _parse_single_spec(
    spec_data: dict[str, Any],
    insights: list[InsightResult],
    index: int,
) -> ChartSpec:
    """Parse and validate a single chart spec from LLM JSON.

    Raises:
        ValueError: If validation fails.
        KeyError: If required field missing.
    """
    # Required fields
    chart_type_str = spec_data["chart_type"]
    title = spec_data["title"]
    mapping_data = spec_data["data_mapping"]

    if not title or not isinstance(title, str):
        raise ValueError("title must be a non-empty string")

    # Validate chart type
    try:
        chart_type = ChartType(chart_type_str)
    except ValueError:
        valid = [t.value for t in ChartType]
        raise ValueError(
            f"Invalid chart type '{chart_type_str}'. Valid: {valid}"
        )

    # Validate and construct DataMapping
    if not isinstance(mapping_data, dict):
        raise ValueError(
            f"data_mapping must be a dict, got {type(mapping_data).__name__}"
        )

    data_mapping = DataMapping(
        x=mapping_data.get("x"),
        y=mapping_data.get("y"),
        color=mapping_data.get("color"),
        size=mapping_data.get("size"),
        label=mapping_data.get("label"),
        values=mapping_data.get("values"),
        categories=mapping_data.get("categories"),
    )

    # Validate source_step_index
    source_step_index = spec_data.get("source_step_index", 0)
    if not isinstance(source_step_index, int):
        raise ValueError(
            f"source_step_index must be an int, got {type(source_step_index).__name__}"
        )
    if source_step_index < 0 or source_step_index >= len(insights):
        raise ValueError(
            f"source_step_index {source_step_index} out of range "
            f"(0 to {len(insights) - 1})"
        )

    # Cross-reference columns against the correct insight
    insight = insights[source_step_index]
    mapping_errors = _validate_data_mapping(
        data_mapping, chart_type, insight.column_names
    )
    if mapping_errors:
        raise ValueError(
            f"Data mapping validation failed: {'; '.join(mapping_errors)}"
        )

    # Optional fields with defaults
    priority_str = spec_data.get("priority", "primary")
    try:
        priority = ChartPriority(priority_str)
    except ValueError:
        priority = ChartPriority.PRIMARY

    return ChartSpec(
        chart_type=chart_type,
        data_mapping=data_mapping,
        title=title,
        priority=priority,
        x_label=spec_data.get("x_label"),
        y_label=spec_data.get("y_label"),
        subtitle=spec_data.get("subtitle"),
        source_step_index=source_step_index,
    )


# =============================================================================
# Async LLM entry points
# =============================================================================


async def plan_charts(
    insights: list[InsightResult],
    questions: str,
    llm_client: LLMClient,
    max_charts: int = 1,
) -> list[ChartSpec]:
    """Plan chart specifications from exploration insights.

    Step 1 of chart creation. Returns specs without code.

    Args:
        insights: Results from MVP.4.
        questions: Original user questions.
        llm_client: LLM client for planning.
        max_charts: Maximum number of charts. MVP=1, DI-1.3=N.

    Raises:
        ChartGenerationError: If planning or validation fails.
        LLMError: If LLM API call fails.
    """
    user_prompt = build_chart_planning_prompt(insights, questions, max_charts)

    try:
        raw_response = await llm_client.complete(
            system=PLANNING_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
            max_tokens=2048,
        )
    except Exception as e:
        raise LLMError(
            f"LLM call failed during chart planning: {e}"
        ) from e

    specs = parse_chart_specs(raw_response, insights, max_charts)

    logger.info("Planned %d chart spec(s)", len(specs))
    return specs


async def generate_chart_code(
    spec: ChartSpec,
    insight: InsightResult,
    llm_client: LLMClient,
    renderer_type: RendererType = RendererType.MATPLOTLIB,
    max_rows: int = 50,
) -> ChartPlan:
    """Generate renderer-specific code for a ChartSpec.

    Step 2 of chart creation. Takes a spec and produces executable code.

    Args:
        spec: The chart specification from plan_charts().
        insight: The data insight to visualize.
        llm_client: LLM client for code generation.
        renderer_type: Which renderer to target.
        max_rows: Maximum rows to embed inline.

    Raises:
        ChartGenerationError: If code generation or parsing fails.
        LLMError: If LLM API call fails.
    """
    user_prompt = build_code_generation_prompt(
        spec, insight, renderer_type, max_rows
    )

    try:
        raw_response = await llm_client.complete(
            system=CODEGEN_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as e:
        raise LLMError(
            f"LLM call failed during chart code generation: {e}"
        ) from e

    try:
        code = parse_code_from_response(raw_response)
    except ValueError as e:
        raise ChartGenerationError(
            f"Failed to parse code from LLM response: {e}"
        ) from e

    logger.info("Generated %s code for chart '%s'", renderer_type.value, spec.title)
    return ChartPlan(spec=spec, code=code, renderer_type=renderer_type)


async def plan_and_generate(
    insights: list[InsightResult],
    questions: str,
    llm_client: LLMClient,
    max_charts: int = 1,
    renderer_type: RendererType = RendererType.MATPLOTLIB,
    max_rows: int = 50,
) -> list[ChartPlan]:
    """Combined entry point: plan specs then generate code.

    This is what the LangGraph pipeline node calls.
    Sequential generation (DI-1.3 can switch to asyncio.gather).

    Args:
        insights: Results from MVP.4.
        questions: Original user questions.
        llm_client: LLM client.
        max_charts: Maximum charts to plan. MVP=1, DI-1.3=N.
        renderer_type: Which renderer to target.
        max_rows: Maximum rows to embed inline per chart.

    Raises:
        ChartGenerationError: If planning or code generation fails.
        LLMError: If LLM API call fails.
    """
    specs = await plan_charts(insights, questions, llm_client, max_charts)

    chart_plans = []
    for spec in specs:
        insight = insights[spec.source_step_index]
        plan = await generate_chart_code(
            spec, insight, llm_client, renderer_type, max_rows
        )
        chart_plans.append(plan)

    logger.info("Generated %d chart plan(s)", len(chart_plans))
    return chart_plans
