"""Prompt templates for analysis planning (MVP.3).

Separated from planner.py so prompt iteration doesn't touch logic.
"""

from autodash.models import AggregationType

# Build aggregation type list from the enum to stay in sync
_AGG_TYPES = ", ".join(t.value for t in AggregationType)

SYSTEM_PROMPT = f"""You are an expert data analyst. Given a dataset profile and user questions, produce a structured analysis plan.

Guidelines:
- Reference only columns that exist in the dataset
- Choose aggregation types that match column types
- Provide clear descriptions and rationale
- Keep analysis focused and relevant to the questions

Available aggregation types: {_AGG_TYPES}

Aggregation guidance:
- sum, mean, median, min, max: numeric columns
- count: any column
- group_by: group rows and aggregate
- correlation: relationship between numeric columns
- distribution: value distribution / histogram
- time_series: trends over datetime columns
- comparison: compare groups or categories
- custom: freeform pandas (escape hatch)
"""

OUTPUT_FORMAT = """Respond with a JSON array of analysis steps. Each step:

[{{
    "description": "Brief description of what this step computes",
    "target_columns": ["column_name"],
    "aggregation": "aggregation_type",
    "group_by_columns": ["grouping_column"],
    "filter_expression": null,
    "sort_by": null,
    "limit": null,
    "rationale": "Why this step answers the user's question"
}}]

Rules:
- group_by_columns: empty array [] if not grouping
- filter_expression: valid pandas query string or null
- sort_by: column name or null
- limit: integer or null
- All column names must exist in the dataset"""


def build_user_prompt(
    profile_summary: str,
    questions: str,
    max_steps: int,
) -> str:
    """Build the user prompt for analysis planning.

    Args:
        profile_summary: Formatted dataset profile from _profile_summary().
        questions: User's natural language questions.
        max_steps: Maximum number of analysis steps to generate.
    """
    return (
        f"Dataset Profile:\n{profile_summary}\n\n"
        f"User Questions:\n{questions}\n\n"
        f"Generate up to {max_steps} analysis step(s) that best answer these questions.\n\n"
        f"{OUTPUT_FORMAT}"
    )
