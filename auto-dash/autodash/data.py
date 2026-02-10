"""Data loading and profiling (MVP.2).

Loads tabular data from files, profiles each column, detects semantic types,
and produces a DataProfile consumed by downstream pipeline modules.

OCP extension: new file formats implement DataLoader and call register_loader().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd

from autodash.config import ProfileConfig
from autodash.models import ColumnProfile, DataProfile, SemanticType
from plotlint.core.errors import DataError


# =============================================================================
# DataLoader protocol + implementations
# =============================================================================


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading tabular data from a file.

    OCP: new file formats implement this protocol and register themselves
    via register_loader() — no modification to existing loaders.
    """

    def supports(self, path: Path) -> bool:
        """Return True if this loader can handle the given file path."""
        ...

    def load(self, path: Path) -> pd.DataFrame:
        """Load the file into a pandas DataFrame.

        Raises:
            DataError: If loading fails.
        """
        ...

    @property
    def format_name(self) -> str:
        """Short identifier for this format (e.g. 'csv', 'xlsx')."""
        ...


class CsvLoader:
    """Load CSV files."""

    format_name: str = "csv"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def load(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise DataError(f"Failed to load CSV from {path}: {e}") from e


class ExcelLoader:
    """Load Excel (.xlsx, .xls) files."""

    format_name: str = "xlsx"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in (".xlsx", ".xls")

    def load(self, path: Path) -> pd.DataFrame:
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise DataError(
                "Excel support requires openpyxl. "
                "Install with: pip install auto-dash[excel]"
            ) from e
        try:
            return pd.read_excel(path, sheet_name=0)
        except Exception as e:
            raise DataError(f"Failed to load Excel from {path}: {e}") from e


class ParquetLoader:
    """Load Parquet files."""

    format_name: str = "parquet"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".parquet"

    def load(self, path: Path) -> pd.DataFrame:
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise DataError(
                "Parquet support requires pyarrow. "
                "Install with: pip install auto-dash[parquet]"
            ) from e
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise DataError(f"Failed to load Parquet from {path}: {e}") from e


# =============================================================================
# Loader registry
# =============================================================================

_LOADERS: list[DataLoader] = [CsvLoader(), ExcelLoader(), ParquetLoader()]


def register_loader(loader: DataLoader) -> None:
    """Register a new loader. OCP extension point."""
    _LOADERS.append(loader)


def load_dataframe(path: Path) -> tuple[pd.DataFrame, str]:
    """Find an appropriate loader and load the file.

    Returns:
        Tuple of (dataframe, format_name).

    Raises:
        DataError: If no loader supports the file or loading fails.
    """
    for loader in _LOADERS:
        if loader.supports(path):
            return loader.load(path), loader.format_name
    supported = [loader.format_name for loader in _LOADERS]
    raise DataError(
        f"No loader for {path.suffix}. Supported formats: {supported}"
    )


# =============================================================================
# Semantic type detection
# =============================================================================


def detect_semantic_type(
    series: pd.Series,
    unique_ratio: float,
    config: ProfileConfig,
) -> SemanticType:
    """Infer the semantic type of a pandas Series.

    Pure function (deterministic when sampling uses fixed random_state).
    See plan for full decision tree.

    Args:
        series: The column data.
        unique_ratio: unique_count / total_rows.
        config: Profiling thresholds.
    """
    dtype_str = str(series.dtype)

    # 1. Native datetime
    if "datetime" in dtype_str:
        return SemanticType.DATETIME

    # 2. Native boolean
    if dtype_str == "bool":
        return SemanticType.BOOLEAN

    # 3. Numeric
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            unique_vals = set(non_null.unique())
            if (
                len(unique_vals) <= config.boolean_max_unique
                and unique_vals <= {0, 1, 0.0, 1.0}
            ):
                return SemanticType.BOOLEAN
        return SemanticType.NUMERIC

    # 4. Object columns
    non_null = series.dropna()
    if len(non_null) == 0:
        return SemanticType.TEXT

    unique_count = int(series.nunique(dropna=True))

    # 4a. Boolean string detection
    if unique_count <= config.boolean_max_unique:
        lowered = {str(v).strip().lower() for v in non_null.unique()}
        if lowered <= config.boolean_string_values:
            return SemanticType.BOOLEAN

    # 4b. Date parsing on object columns
    sample_size = min(len(non_null), config.date_parse_sample_size)
    sample = non_null.sample(n=sample_size, random_state=42)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    success_ratio = parsed.notna().sum() / len(parsed)
    if success_ratio >= config.date_parse_success_threshold:
        return SemanticType.DATETIME

    # 4c. Identifier
    if unique_ratio >= config.identifier_min_cardinality:
        return SemanticType.IDENTIFIER

    # 4d. Categorical (dual guard)
    if (
        unique_count <= config.categorical_max_unique
        and unique_ratio <= config.categorical_max_cardinality
    ):
        return SemanticType.CATEGORICAL

    # 4e. High-cardinality text
    return SemanticType.TEXT


# =============================================================================
# Date granularity detection
# =============================================================================


def _detect_date_granularity(date_series: pd.Series) -> str:
    """Detect the granularity of a datetime series.

    Returns one of: "yearly", "monthly", "daily", "hourly", "irregular", "unknown".
    """
    if len(date_series) < 2:
        return "unknown"

    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, errors="coerce")

    date_series = date_series.dropna().sort_values()
    if len(date_series) < 2:
        return "unknown"

    diffs = date_series.diff().dropna()
    if len(diffs) == 0:
        return "unknown"

    median_diff = diffs.median()

    has_time = (date_series.dt.hour != 0).any() or (
        date_series.dt.minute != 0
    ).any()

    if median_diff >= pd.Timedelta(days=300):
        return "yearly"
    if median_diff >= pd.Timedelta(days=20):
        return "monthly"
    if median_diff >= pd.Timedelta(days=1) - pd.Timedelta(hours=2):
        return "daily"
    if has_time:
        return "hourly"
    return "irregular"


# =============================================================================
# Column and DataFrame profiling
# =============================================================================


def profile_column(
    series: pd.Series,
    total_rows: int,
    config: ProfileConfig,
) -> ColumnProfile:
    """Profile a single column. Pure function — no side effects.

    Args:
        series: The column data.
        total_rows: Total row count of the DataFrame.
        config: Profiling thresholds.
    """
    null_count = int(series.isna().sum())
    null_fraction = null_count / total_rows if total_rows > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    cardinality_fraction = unique_count / total_rows if total_rows > 0 else 0.0

    semantic_type = detect_semantic_type(series, cardinality_fraction, config)

    # Optional fields
    min_val = max_val = mean_val = median_val = std_val = None
    top_values: Optional[list[dict[str, Any]]] = None
    date_min = date_max = date_granularity = None

    non_null = series.dropna()

    if semantic_type == SemanticType.NUMERIC and len(non_null) > 0:
        min_val = float(non_null.min())
        max_val = float(non_null.max())
        mean_val = float(non_null.mean())
        median_val = float(non_null.median())
        std_val = float(non_null.std()) if len(non_null) > 1 else 0.0

    if semantic_type == SemanticType.CATEGORICAL and len(non_null) > 0:
        value_counts = non_null.value_counts().head(config.top_values_count)
        top_values = [
            {"value": str(val), "count": int(count)}
            for val, count in value_counts.items()
        ]

    if semantic_type == SemanticType.DATETIME and len(non_null) > 0:
        if not pd.api.types.is_datetime64_any_dtype(non_null):
            non_null = pd.to_datetime(non_null, errors="coerce").dropna()
        if len(non_null) > 0:
            date_min = non_null.min().isoformat()
            date_max = non_null.max().isoformat()
            date_granularity = _detect_date_granularity(non_null)

    return ColumnProfile(
        name=series.name,
        pandas_dtype=str(series.dtype),
        semantic_type=semantic_type,
        null_count=null_count,
        null_fraction=null_fraction,
        unique_count=unique_count,
        cardinality_fraction=cardinality_fraction,
        min=min_val,
        max=max_val,
        mean=mean_val,
        median=median_val,
        std=std_val,
        top_values=top_values,
        date_min=date_min,
        date_max=date_max,
        date_granularity=date_granularity,
    )


def profile_dataframe(
    df: pd.DataFrame,
    source_path: str,
    file_format: str,
    config: ProfileConfig,
) -> DataProfile:
    """Profile an entire DataFrame.

    Args:
        df: The loaded data.
        source_path: Original file path (for metadata).
        file_format: Format identifier from the loader.
        config: Profiling thresholds.
    """
    total_rows = len(df)

    columns = [
        profile_column(df[col_name], total_rows, config)
        for col_name in df.columns
    ]

    memory_bytes = int(df.memory_usage(deep=True).sum())

    sample_rows: Optional[list[dict[str, Any]]] = None
    sample_size = min(total_rows, config.sample_rows_count)
    if sample_size > 0:
        sample_rows = []
        for _, row in df.head(sample_size).iterrows():
            row_dict: dict[str, Any] = {}
            for col_name, value in row.items():
                if pd.isna(value):
                    row_dict[col_name] = None
                elif isinstance(value, (int, float, str, bool)):
                    row_dict[col_name] = value
                else:
                    row_dict[col_name] = str(value)
            sample_rows.append(row_dict)

    return DataProfile(
        source_path=source_path,
        row_count=total_rows,
        columns=columns,
        file_format=file_format,
        memory_bytes=memory_bytes,
        sample_rows=sample_rows,
    )


# =============================================================================
# Top-level entry point
# =============================================================================


def load_and_profile(
    path: str | Path,
    config: Optional[ProfileConfig] = None,
) -> DataProfile:
    """Load a file and return its profile. Main entry point for MVP.2.

    This is what the LangGraph pipeline node calls.

    Args:
        path: Path to the data file.
        config: Profiling configuration. Uses defaults if None.

    Raises:
        DataError: If loading or profiling fails.
    """
    if config is None:
        config = ProfileConfig()

    path = Path(path)

    if not path.exists():
        raise DataError(f"File not found: {path}")

    if not path.is_file():
        raise DataError(f"Path is not a file: {path}")

    df, file_format = load_dataframe(path)
    return profile_dataframe(df, str(path), file_format, config)
