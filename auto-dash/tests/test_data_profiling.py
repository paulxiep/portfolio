"""Tests for data profiling — semantic type detection, column/DF profiling."""

from pathlib import Path

import pandas as pd
import pytest

from autodash.config import ProfileConfig
from autodash.data import (
    _detect_date_granularity,
    detect_semantic_type,
    load_and_profile,
    profile_column,
    profile_dataframe,
)
from autodash.models import DataProfile, SemanticType
from plotlint.core.errors import DataError

TEST_DATA = Path(__file__).parent / "test_data"
SAMPLE_CSV = TEST_DATA / "sample.csv"

CONFIG = ProfileConfig()


# =============================================================================
# detect_semantic_type
# =============================================================================


class TestDetectSemanticType:
    def test_numeric_int(self):
        s = pd.Series([1, 2, 3, 4, 5], name="x")
        assert detect_semantic_type(s, 1.0, CONFIG) == SemanticType.NUMERIC

    def test_numeric_float(self):
        s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], name="x")
        assert detect_semantic_type(s, 1.0, CONFIG) == SemanticType.NUMERIC

    def test_datetime_native(self):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]), name="x")
        assert detect_semantic_type(s, 1.0, CONFIG) == SemanticType.DATETIME

    def test_datetime_string_parsed(self):
        s = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"], name="x")
        assert detect_semantic_type(s, 1.0, CONFIG) == SemanticType.DATETIME

    def test_boolean_native(self):
        s = pd.Series([True, False, True], name="x")
        assert detect_semantic_type(s, 2 / 3, CONFIG) == SemanticType.BOOLEAN

    def test_boolean_int_01(self):
        s = pd.Series([0, 1, 0, 1, 1], name="x")
        assert detect_semantic_type(s, 2 / 5, CONFIG) == SemanticType.BOOLEAN

    def test_boolean_string(self):
        s = pd.Series(["True", "False", "True", "False"], name="x")
        assert detect_semantic_type(s, 2 / 4, CONFIG) == SemanticType.BOOLEAN

    def test_boolean_yes_no(self):
        s = pd.Series(["yes", "no", "yes", "no", "yes"], name="x")
        assert detect_semantic_type(s, 2 / 5, CONFIG) == SemanticType.BOOLEAN

    def test_categorical_low_cardinality(self):
        s = pd.Series(["A", "B", "C", "A", "B"] * 10, name="x")
        assert detect_semantic_type(s, 3 / 50, CONFIG) == SemanticType.CATEGORICAL

    def test_identifier_high_cardinality(self):
        s = pd.Series([f"ID-{i}" for i in range(100)], name="x")
        assert detect_semantic_type(s, 1.0, CONFIG) == SemanticType.IDENTIFIER

    def test_text_medium_cardinality(self):
        # 30 unique in 50 rows = 0.6 ratio, 30 unique > 20 max_unique
        s = pd.Series([f"text_{i}" for i in range(30)] + [f"text_{i}" for i in range(20)], name="x")
        assert detect_semantic_type(s, 30 / 50, CONFIG) == SemanticType.TEXT

    def test_all_null_returns_text(self):
        s = pd.Series([None, None, None], name="x", dtype="object")
        assert detect_semantic_type(s, 0.0, CONFIG) == SemanticType.TEXT


# =============================================================================
# _detect_date_granularity
# =============================================================================


class TestDetectDateGranularity:
    def test_daily(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        assert _detect_date_granularity(dates) == "daily"

    def test_monthly(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=6, freq="MS"))
        assert _detect_date_granularity(dates) == "monthly"

    def test_yearly(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=5, freq="YS"))
        assert _detect_date_granularity(dates) == "yearly"

    def test_hourly(self):
        dates = pd.Series(pd.date_range("2024-01-01", periods=24, freq="h"))
        assert _detect_date_granularity(dates) == "hourly"

    def test_single_date(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01"]))
        assert _detect_date_granularity(dates) == "unknown"

    def test_empty(self):
        dates = pd.Series([], dtype="datetime64[ns]")
        assert _detect_date_granularity(dates) == "unknown"


# =============================================================================
# profile_column
# =============================================================================


class TestProfileColumn:
    def test_numeric_column(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="revenue")
        col = profile_column(s, 5, CONFIG)
        assert col.semantic_type == SemanticType.NUMERIC
        assert col.min == 10.0
        assert col.max == 50.0
        assert col.mean == 30.0
        assert col.median == 30.0
        assert col.std is not None
        assert col.null_count == 0
        assert col.null_fraction == 0.0

    def test_categorical_column(self):
        s = pd.Series(["A", "B", "C", "A", "B"] * 10, name="category")
        col = profile_column(s, 50, CONFIG)
        assert col.semantic_type == SemanticType.CATEGORICAL
        assert col.top_values is not None
        assert len(col.top_values) <= CONFIG.top_values_count
        assert col.top_values[0]["value"] in ("A", "B")

    def test_datetime_column(self):
        s = pd.Series(
            pd.date_range("2024-01-01", periods=10, freq="MS"), name="date"
        )
        col = profile_column(s, 10, CONFIG)
        assert col.semantic_type == SemanticType.DATETIME
        assert col.date_min is not None
        assert col.date_max is not None
        assert col.date_granularity == "monthly"

    def test_column_with_nulls(self):
        s = pd.Series([1.0, None, 3.0, None, 5.0], name="val")
        col = profile_column(s, 5, CONFIG)
        assert col.null_count == 2
        assert col.null_fraction == 0.4
        assert col.unique_count == 3

    def test_all_null_column(self):
        s = pd.Series([None, None, None], name="empty", dtype="object")
        col = profile_column(s, 3, CONFIG)
        assert col.null_count == 3
        assert col.null_fraction == 1.0
        assert col.unique_count == 0

    def test_frozen(self):
        s = pd.Series([1, 2, 3], name="x")
        col = profile_column(s, 3, CONFIG)
        with pytest.raises(AttributeError):
            col.name = "changed"


# =============================================================================
# profile_dataframe
# =============================================================================


class TestProfileDataframe:
    def test_profile_sample_csv(self):
        df = pd.read_csv(SAMPLE_CSV)
        profile = profile_dataframe(df, str(SAMPLE_CSV), "csv", CONFIG)
        assert profile.row_count == 20
        assert len(profile.columns) == 6
        assert profile.file_format == "csv"
        assert profile.memory_bytes is not None
        assert profile.memory_bytes > 0

    def test_sample_rows_populated(self):
        df = pd.read_csv(SAMPLE_CSV)
        profile = profile_dataframe(df, str(SAMPLE_CSV), "csv", CONFIG)
        assert profile.sample_rows is not None
        assert len(profile.sample_rows) == CONFIG.sample_rows_count
        assert "revenue" in profile.sample_rows[0]

    def test_column_names(self):
        df = pd.read_csv(SAMPLE_CSV)
        profile = profile_dataframe(df, str(SAMPLE_CSV), "csv", CONFIG)
        names = profile.column_names()
        assert "id" in names
        assert "category" in names
        assert "revenue" in names

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        profile = profile_dataframe(df, "empty.csv", "csv", CONFIG)
        assert profile.row_count == 0
        assert len(profile.columns) == 0
        assert profile.sample_rows is None

    def test_single_row(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        profile = profile_dataframe(df, "one.csv", "csv", CONFIG)
        assert profile.row_count == 1
        assert len(profile.columns) == 2


# =============================================================================
# JSON round-trip
# =============================================================================


class TestJsonRoundtrip:
    def test_roundtrip(self):
        df = pd.read_csv(SAMPLE_CSV)
        profile = profile_dataframe(df, str(SAMPLE_CSV), "csv", CONFIG)
        json_str = profile.to_json()
        restored = DataProfile.from_json(json_str)
        assert restored.source_path == profile.source_path
        assert restored.row_count == profile.row_count
        assert len(restored.columns) == len(profile.columns)
        for orig, rest in zip(profile.columns, restored.columns):
            assert orig.name == rest.name
            assert orig.semantic_type == rest.semantic_type
            assert orig.null_count == rest.null_count


# =============================================================================
# load_and_profile (integration)
# =============================================================================


class TestLoadAndProfile:
    def test_load_and_profile_csv(self):
        profile = load_and_profile(SAMPLE_CSV)
        assert profile.row_count == 20
        assert profile.file_format == "csv"
        assert len(profile.columns) == 6

    def test_load_and_profile_with_config(self):
        config = ProfileConfig(sample_rows_count=3)
        profile = load_and_profile(SAMPLE_CSV, config=config)
        assert profile.sample_rows is not None
        assert len(profile.sample_rows) == 3

    def test_file_not_found(self):
        with pytest.raises(DataError, match="File not found"):
            load_and_profile("nonexistent.csv")

    def test_path_is_directory(self):
        with pytest.raises(DataError, match="not a file"):
            load_and_profile(TEST_DATA)
