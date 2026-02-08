"""All autodash data models and pipeline state.

This is the shared contract file. All dataclasses that cross module
boundaries live here. G3: full type definitions are provided upfront;
construction logic (LLM prompts, parsing, validation) comes in MVP.2-5, 9.

G12: TYPE_CHECKING guard for pandas (InsightResult.result_df).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, TypedDict

if TYPE_CHECKING:
    import pandas as pd


# =============================================================================
# MVP.2: Data Intelligence
# =============================================================================


class SemanticType(str, Enum):
    """Detected semantic meaning of a column, beyond its pandas dtype."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"


@dataclass(frozen=True)
class ColumnProfile:
    """Statistical profile of a single column.

    Immutable. JSON-serializable via asdict().
    """

    name: str
    pandas_dtype: str
    semantic_type: SemanticType
    null_count: int
    null_fraction: float
    unique_count: int
    cardinality_fraction: float

    # Numeric stats (None for non-numeric)
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None

    # Categorical stats (None for non-categorical)
    top_values: Optional[list[dict[str, Any]]] = None

    # Date stats (None for non-date)
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    date_granularity: Optional[str] = None


@dataclass(frozen=True)
class DataProfile:
    """Complete profile of a loaded dataset.

    Primary output of MVP.2 and primary input to MVP.3 (planner).
    JSON-serializable via to_json().
    """

    source_path: str
    row_count: int
    columns: list[ColumnProfile]
    file_format: str

    memory_bytes: Optional[int] = None
    sample_rows: Optional[list[dict[str, Any]]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, raw: str) -> DataProfile:
        """Deserialize from JSON string."""
        data = json.loads(raw)
        columns = [
            ColumnProfile(
                **{
                    **col,
                    "semantic_type": SemanticType(col["semantic_type"]),
                }
            )
            for col in data["columns"]
        ]
        return cls(
            **{k: v for k, v in data.items() if k != "columns"},
            columns=columns,
        )

    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        return next((col for col in self.columns if col.name == name), None)


# =============================================================================
# MVP.3: Analysis Planning
# =============================================================================


class AggregationType(str, Enum):
    """Standard aggregation operations."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    GROUP_BY = "group_by"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"
    CUSTOM = "custom"


@dataclass(frozen=True)
class AnalysisStep:
    """A single analysis operation to be performed on the data.

    Describes WHAT to compute, not HOW.
    """

    description: str
    target_columns: list[str]
    aggregation: AggregationType
    group_by_columns: list[str] = field(default_factory=list)
    filter_expression: Optional[str] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    rationale: str = ""
    step_index: int = 0


# =============================================================================
# MVP.4: Data Exploration
# =============================================================================


@dataclass
class InsightResult:
    """Output of a single data exploration step.

    Not frozen: has __post_init__. result_df is not serializable.
    """

    step: AnalysisStep
    result_df: pd.DataFrame
    summary: str
    code_used: str
    attempts: int

    result_shape: tuple[int, int] = field(init=False)
    column_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.result_shape = self.result_df.shape
        self.column_names = list(self.result_df.columns)

    def to_prompt_context(self) -> str:
        """Serialize for use in LLM prompts (MVP.5 chart planning)."""
        sample = self.result_df.head(10).to_string(index=False)
        return (
            f"Analysis: {self.step.description}\n"
            f"Result shape: {self.result_shape[0]} rows x {self.result_shape[1]} columns\n"
            f"Columns: {', '.join(self.column_names)}\n"
            f"Summary: {self.summary}\n"
            f"Sample data:\n{sample}"
        )


# =============================================================================
# MVP.5: Chart Planning
# =============================================================================


class ChartType(str, Enum):
    """Supported chart types. Renderer-agnostic."""

    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"


class ChartPriority(str, Enum):
    """Visual hierarchy priority for dashboard layout."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUPPORTING = "supporting"


class RendererType(str, Enum):
    """Which rendering backend to use for code generation."""

    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"


@dataclass(frozen=True)
class DataMapping:
    """How data columns map to chart axes/dimensions."""

    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    label: Optional[str] = None
    values: Optional[str] = None
    categories: Optional[str] = None


@dataclass(frozen=True)
class ChartSpec:
    """Renderer-agnostic specification for a single chart.

    Describes WHAT to render, not HOW.
    """

    chart_type: ChartType
    data_mapping: DataMapping
    title: str
    priority: ChartPriority = ChartPriority.PRIMARY

    x_label: Optional[str] = None
    y_label: Optional[str] = None
    subtitle: Optional[str] = None
    figsize: tuple[float, float] = (10, 6)

    source_step_index: int = 0
    color_palette: Optional[str] = None
    annotations: list[str] = field(default_factory=list)


@dataclass
class ChartPlan:
    """Pairs a ChartSpec with its generated code.

    Not frozen because the code may be replaced by the Patcher (MVP.8).
    """

    spec: ChartSpec
    code: str
    renderer_type: RendererType = RendererType.MATPLOTLIB


# =============================================================================
# MVP.9: Output
# =============================================================================


class OutputFormat(str, Enum):
    """Supported output formats."""

    PNG = "png"
    SOURCE = "source"


@dataclass(frozen=True)
class OutputArtifact:
    """Record of a single output file produced."""

    path: str
    format: OutputFormat
    size_bytes: int
    description: str


@dataclass
class OutputResult:
    """Complete output from the output stage."""

    artifacts: list[OutputArtifact] = field(default_factory=list)
    summary: str = ""

    def add(self, artifact: OutputArtifact) -> None:
        self.artifacts.append(artifact)


# =============================================================================
# Pipeline State (LangGraph TypedDict)
# =============================================================================


class PipelineState(TypedDict, total=False):
    """LangGraph state for the AutoDash end-to-end pipeline.

    Each node reads/writes only its relevant fields.
    DI phases add fields additively.
    """

    # Inputs
    source_path: str
    questions: str

    # MVP.2: Data Intelligence
    data_profile: Optional[DataProfile]

    # MVP.3: Analysis Planning
    analysis_steps: list[AnalysisStep]

    # MVP.4: Data Exploration
    insights: list[InsightResult]

    # MVP.5: Chart Planning
    chart_plans: list[ChartPlan]

    # MVP.6-8: Convergence (per chart)
    polished_charts: list[ChartPlan]

    # MVP.9: Output
    output_path: Optional[str]

    # Error tracking
    errors: list[str]
