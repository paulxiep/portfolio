"""Prompt templates for chart planning (MVP.5).

Separated from charts.py so prompt iteration doesn't touch logic.
"""

from autodash.models import ChartPriority, ChartType

# Build enum value lists from enums to stay in sync
_CHART_TYPES = ", ".join(t.value for t in ChartType)
_PRIORITIES = ", ".join(p.value for p in ChartPriority)

SYSTEM_PROMPT = f"""You are an expert data visualization specialist. Given analysis results and user questions, choose the best chart type and produce a structured chart specification.

Chart type guidance:
- bar: comparing categories (e.g., revenue by region)
- grouped_bar: comparing sub-categories side by side (requires color for grouping)
- stacked_bar: showing composition across categories (requires color for stacking)
- line: trends over ordered/time data
- scatter: relationship between two numeric variables
- pie: proportional composition (use sparingly, <=7 slices)
- histogram: distribution of a single numeric variable
- heatmap: matrix of values across two dimensions
- box: distribution and outliers across groups
- area: cumulative trends over time

Available chart types: {_CHART_TYPES}
Available priorities: {_PRIORITIES}

Data mapping rules (which fields are required per chart type):
- bar, line, area, scatter, heatmap: require x AND y
- grouped_bar, stacked_bar: require x, y, AND color
- pie: require values AND categories (NOT x/y)
- histogram: require x OR y (the variable to bin)
- box: require y (x is optional, used for grouping)

All data_mapping fields that are set must reference column names that exist in the analysis result.
"""

OUTPUT_FORMAT = """Respond with a JSON array of chart specifications. Each spec:

[{{
    "chart_type": "one of: {chart_types}",
    "title": "Descriptive chart title",
    "data_mapping": {{
        "x": "column_name or null",
        "y": "column_name or null",
        "color": "column_name or null",
        "size": "column_name or null",
        "label": "column_name or null",
        "values": "column_name or null",
        "categories": "column_name or null"
    }},
    "priority": "primary|secondary|supporting",
    "x_label": "axis label or null",
    "y_label": "axis label or null",
    "subtitle": "optional subtitle or null",
    "source_step_index": 0
}}]

Rules:
- source_step_index: which analysis result (0-indexed) this chart visualizes
- data_mapping fields: set to null if not applicable for the chart type
- All column names must exist in the referenced analysis result
- title must be non-empty and descriptive""".format(chart_types=_CHART_TYPES)


def build_user_prompt(
    insight_contexts: list[str],
    questions: str,
    max_charts: int,
) -> str:
    """Build the user prompt for chart planning.

    Args:
        insight_contexts: InsightResult.to_prompt_context() for each insight.
        questions: User's original questions.
        max_charts: Maximum number of charts to plan.
    """
    insights_section = "\n\n".join(
        f"--- Analysis Result {i} ---\n{ctx}"
        for i, ctx in enumerate(insight_contexts)
    )

    return (
        f"Analysis Results:\n{insights_section}\n\n"
        f"User Questions:\n{questions}\n\n"
        f"Generate up to {max_charts} chart specification(s) that best visualize these results "
        f"to answer the user's questions.\n\n"
        f"{OUTPUT_FORMAT}"
    )
