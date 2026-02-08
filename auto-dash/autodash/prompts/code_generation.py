"""Prompt templates for chart code generation (MVP.5).

Separated from charts.py so prompt iteration doesn't touch logic.
The renderer_type parameter selects the appropriate template.
PL-1.6 adds a Plotly template without modifying existing code.
"""

SYSTEM_PROMPT = """You are an expert matplotlib developer. Given a chart specification and data, write Python code that creates the chart using matplotlib.

Rules:
- Import matplotlib.pyplot as plt and pandas as pd
- Create the DataFrame from the provided data dict literal
- Create the figure with fig, ax = plt.subplots(figsize=(...))
- Do NOT call plt.savefig(), plt.show(), or plt.close()
- Do NOT set the matplotlib backend (no matplotlib.use())
- Call plt.tight_layout() at the end
- Code must be completely self-contained — all data embedded inline
- Handle edge cases: rotate x-tick labels if more than 6 categories
- Use clear, readable code with no unnecessary complexity
- For pie charts, use ax.pie() with autopct for percentage labels
- For heatmaps, use ax.imshow() or plt.pcolormesh() with a colorbar
- For box plots, use ax.boxplot() or ax.violinplot()
"""


def build_user_prompt(
    spec_json: str,
    data_dict: str,
    column_info: str,
    renderer_type: str = "matplotlib",
) -> str:
    """Build the user prompt for code generation.

    Args:
        spec_json: ChartSpec serialized as JSON string.
        data_dict: result_df serialized as Python dict literal for inline embedding.
        column_info: Column names + dtypes summary.
        renderer_type: Which renderer to target (future: "plotly").
    """
    return (
        f"Chart Specification:\n{spec_json}\n\n"
        f"Data (use this exact dict to create the DataFrame):\n{data_dict}\n\n"
        f"Column Info:\n{column_info}\n\n"
        f"Write {renderer_type} code that implements this chart specification. "
        f"Embed the data inline as pd.DataFrame({{...}}). "
        f"The code should produce exactly one figure."
    )
