"""Data exploration (MVP.4).

Accepts AnalysisStep + DataFrame, uses LLM to generate pandas code,
executes in sandbox with retry, produces InsightResult.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from autodash.models import AnalysisStep, DataProfile, InsightResult, SemanticType
from autodash.prompts.data_exploration import SYSTEM_PROMPT, build_user_prompt
from plotlint.core.errors import ExplorationError
from plotlint.core.llm import LLMClient
from plotlint.core.parsing import parse_code_from_response
from plotlint.core.sandbox import ExecutionStatus, execute_code

logger = logging.getLogger(__name__)


# =============================================================================
# Profile summary for LLM prompt
# =============================================================================


def _exploration_profile_summary(profile: DataProfile) -> str:
    """Generate profile summary tailored for code generation.

    Unlike the planner's summary (which emphasizes semantics for analysis choice),
    this one includes pandas dtypes because the LLM needs them for correct code.
    """
    lines = [
        f"Dataset: {profile.source_path} ({profile.file_format})",
        f"Rows: {profile.row_count:,}",
        f"Columns ({len(profile.columns)}):",
    ]
    for col in profile.columns:
        parts = [f"  - {col.name}: dtype={col.pandas_dtype}, semantic={col.semantic_type.value}"]
        if col.null_fraction > 0.01:
            parts.append(f", {col.null_fraction:.1%} null")
        if col.semantic_type == SemanticType.NUMERIC and col.min is not None:
            parts.append(f", range=[{col.min}, {col.max}]")
        if col.semantic_type == SemanticType.CATEGORICAL and col.top_values:
            top = [v["value"] for v in col.top_values[:5]]
            parts.append(f", values={top}")
        lines.append("".join(parts))
    return "\n".join(lines)


def _step_details(step: AnalysisStep) -> str:
    """Format AnalysisStep fields as structured text for the LLM prompt."""
    lines = [
        f"Target columns: {step.target_columns}",
        f"Aggregation: {step.aggregation.value}",
    ]
    if step.group_by_columns:
        lines.append(f"Group by: {step.group_by_columns}")
    if step.filter_expression:
        lines.append(f"Filter: {step.filter_expression}")
    if step.sort_by:
        lines.append(f"Sort by: {step.sort_by}")
    if step.limit:
        lines.append(f"Limit: {step.limit}")
    if step.rationale:
        lines.append(f"Rationale: {step.rationale}")
    return "\n".join(lines)


# =============================================================================
# Prompt construction
# =============================================================================


def build_exploration_prompt(
    step: AnalysisStep,
    profile: DataProfile,
    sample_data: str,
    previous_error: Optional[str] = None,
    previous_code: Optional[str] = None,
) -> str:
    """Construct the LLM user prompt for generating pandas code.

    Pure function: separated from LLM call for testability.
    If previous_error is provided, includes it so the LLM can fix the issue.
    """
    summary = _exploration_profile_summary(profile)
    details = _step_details(step)
    return build_user_prompt(
        step_description=step.description,
        step_details=details,
        profile_summary=summary,
        sample_data=sample_data,
        previous_error=previous_error,
        previous_code=previous_code,
    )


# =============================================================================
# Result normalization
# =============================================================================


def _normalize_result(raw_value: object) -> pd.DataFrame:
    """Normalize sandbox return value to a DataFrame.

    Accepts: pd.DataFrame (passthrough), pd.Series (to_frame()),
    scalar (wrap in 1x1 DataFrame).
    Raises ExplorationError for unsupported types.
    """
    if isinstance(raw_value, pd.DataFrame):
        return raw_value
    if isinstance(raw_value, pd.Series):
        return raw_value.to_frame()
    if isinstance(raw_value, (int, float, str, bool)):
        return pd.DataFrame({"result": [raw_value]})
    raise ExplorationError(
        f"Exploration code returned unsupported type: {type(raw_value).__name__}. "
        f"Expected DataFrame, Series, or scalar."
    )


# =============================================================================
# Summarization
# =============================================================================


def summarize_result(step: AnalysisStep, result_df: pd.DataFrame) -> str:
    """Generate a template-based summary of the exploration result.

    Template-based for MVP (no extra LLM call). DI extensions can wrap
    with an LLM summarizer for richer descriptions.
    """
    rows, cols = result_df.shape
    col_names = list(result_df.columns)

    parts = [
        f"Computed '{step.description}': "
        f"{rows} row(s), {cols} column(s) [{', '.join(str(c) for c in col_names)}]."
    ]

    for col in col_names:
        if pd.api.types.is_numeric_dtype(result_df[col]) and rows > 0:
            parts.append(
                f"  {col}: min={result_df[col].min():.4g}, "
                f"max={result_df[col].max():.4g}, "
                f"mean={result_df[col].mean():.4g}"
            )
            break

    return "\n".join(parts)


# =============================================================================
# Main entry point
# =============================================================================


async def explore_step(
    step: AnalysisStep,
    df: pd.DataFrame,
    profile: DataProfile,
    llm_client: LLMClient,
    max_attempts: int = 3,
) -> InsightResult:
    """Execute a single analysis step with retry logic.

    Main entry point for MVP.4.

    Flow:
    1. Build prompt with step + profile + sample data
    2. LLM generates pandas code
    3. Parse code from LLM response
    4. Execute code in sandbox with df injected
    5. If error: rebuild prompt with error message + failed code, retry
    6. Normalize result to DataFrame
    7. Summarize the result
    8. Return InsightResult

    Args:
        step: The analysis step from MVP.3 planner.
        df: The loaded DataFrame from MVP.2.
        profile: The data profile (for prompt context).
        llm_client: LLM client for code generation.
        max_attempts: Maximum execution attempts before giving up.

    Raises:
        ExplorationError: If all attempts fail.
    """
    sample_data = df.head(5).to_string()
    previous_error: Optional[str] = None
    previous_code: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        # 1. Build prompt
        user_prompt = build_exploration_prompt(
            step=step,
            profile=profile,
            sample_data=sample_data,
            previous_error=previous_error,
            previous_code=previous_code,
        )

        # 2. LLM call
        try:
            raw_response = await llm_client.complete(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.0,
                max_tokens=2048,
            )
        except Exception as e:
            raise ExplorationError(
                f"LLM call failed during exploration (attempt {attempt}): {e}"
            ) from e

        # 3. Parse code
        try:
            code = parse_code_from_response(raw_response)
        except ValueError as e:
            previous_error = f"CodeParseError: {e}"
            previous_code = raw_response[:500]
            logger.warning("Attempt %d: failed to parse code: %s", attempt, e)
            if attempt == max_attempts:
                raise ExplorationError(
                    f"Failed to parse code from LLM response after {max_attempts} attempts: {e}"
                ) from e
            continue

        # 4. Execute in sandbox
        result = execute_code(
            code=code,
            timeout_seconds=30,
            inject_globals={"df": df},
        )

        # 5. Check result
        if result.status == ExecutionStatus.SUCCESS and result.return_value is not None:
            try:
                result_df = _normalize_result(result.return_value)
            except ExplorationError:
                previous_error = (
                    f"ResultTypeError: code returned {type(result.return_value).__name__}, "
                    f"expected DataFrame"
                )
                previous_code = code
                logger.warning("Attempt %d: result normalization failed", attempt)
                if attempt == max_attempts:
                    raise
                continue

            summary = summarize_result(step, result_df)

            logger.info(
                "Step '%s' succeeded on attempt %d: %s",
                step.description, attempt, result_df.shape,
            )
            return InsightResult(
                step=step,
                result_df=result_df,
                summary=summary,
                code_used=code,
                attempts=attempt,
            )

        # Execution failed or __result__ not assigned
        if result.status == ExecutionStatus.SUCCESS and result.return_value is None:
            previous_error = (
                "Code executed successfully but did not assign a value to __result__. "
                "Make sure to assign your final result to __result__."
            )
            previous_code = code
        else:
            error_msg = result.error_message or "Unknown error"
            error_type = result.error_type or result.status.value
            previous_error = f"{error_type}: {error_msg}"
            previous_code = code

        logger.warning(
            "Attempt %d/%d failed for step '%s': %s",
            attempt, max_attempts, step.description, previous_error,
        )

    raise ExplorationError(
        f"Exploration failed for step '{step.description}' after {max_attempts} attempts. "
        f"Last error: {previous_error}"
    )
