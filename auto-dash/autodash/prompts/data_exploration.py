"""Prompt templates for data exploration (MVP.4).

Separated from explorer.py so prompt iteration doesn't touch logic.
"""

SYSTEM_PROMPT = """You are an expert pandas developer. Given a dataset description and an analysis task, write Python code using pandas to compute the result.

Rules:
- The DataFrame is pre-loaded as `df`
- Import pandas (`import pandas as pd`) if you need pandas operations beyond what `df` already provides
- Assign your final result to `__result__`
- `__result__` must be a pandas DataFrame or Series
- Do not read files, write files, or make network requests
- Do not modify `df` in-place; create new variables instead
- Handle potential edge cases (empty groups, NaN values)
- Keep the code concise and focused on the specific task
"""

ERROR_RETRY_BLOCK = """
The previous attempt failed with the following error:

Error type: {error_type}
Error message: {error_message}

Previous code that failed:
```python
{previous_code}
```

Fix the code to avoid this error. Do not repeat the same mistake.
"""


def build_user_prompt(
    step_description: str,
    step_details: str,
    profile_summary: str,
    sample_data: str,
    previous_error: str | None = None,
    previous_code: str | None = None,
) -> str:
    """Build the user prompt for pandas code generation.

    Args:
        step_description: The AnalysisStep.description.
        step_details: Structured details (columns, aggregation, filters, etc.).
        profile_summary: Formatted dataset profile.
        sample_data: String representation of df.head() rows.
        previous_error: If retrying, the error string.
        previous_code: If retrying, the code that failed.
    """
    parts = [
        f"Dataset Profile:\n{profile_summary}",
        f"Sample data (first rows of df):\n{sample_data}",
        f"Task: {step_description}",
        f"Details:\n{step_details}",
        "Write pandas code that computes this analysis. "
        "The DataFrame is available as `df`. "
        "Assign the final result to `__result__`.",
    ]

    if previous_error and previous_code:
        parts.append(
            ERROR_RETRY_BLOCK.format(
                error_type=previous_error.split(":")[0] if ":" in previous_error else "Unknown",
                error_message=previous_error,
                previous_code=previous_code,
            )
        )

    return "\n\n".join(parts)
