"""Tests for data loading — loader dispatch, registry, error handling."""

from pathlib import Path

import pandas as pd
import pytest

from autodash.data import (
    CsvLoader,
    DataLoader,
    ExcelLoader,
    ParquetLoader,
    load_dataframe,
    register_loader,
    _LOADERS,
)
from plotlint.core.errors import DataError

TEST_DATA = Path(__file__).parent / "test_data"
SAMPLE_CSV = TEST_DATA / "sample.csv"


class TestLoaderSupports:
    def test_csv_loader_supports_csv(self):
        assert CsvLoader().supports(Path("data.csv"))

    def test_csv_loader_rejects_xlsx(self):
        assert not CsvLoader().supports(Path("data.xlsx"))

    def test_excel_loader_supports_xlsx(self):
        assert ExcelLoader().supports(Path("data.xlsx"))

    def test_excel_loader_supports_xls(self):
        assert ExcelLoader().supports(Path("data.xls"))

    def test_excel_loader_rejects_csv(self):
        assert not ExcelLoader().supports(Path("data.csv"))

    def test_parquet_loader_supports_parquet(self):
        assert ParquetLoader().supports(Path("data.parquet"))

    def test_parquet_loader_rejects_csv(self):
        assert not ParquetLoader().supports(Path("data.csv"))

    def test_case_insensitive(self):
        assert CsvLoader().supports(Path("DATA.CSV"))
        assert ExcelLoader().supports(Path("file.XLSX"))


class TestLoaderProtocol:
    def test_csv_loader_is_data_loader(self):
        assert isinstance(CsvLoader(), DataLoader)

    def test_excel_loader_is_data_loader(self):
        assert isinstance(ExcelLoader(), DataLoader)

    def test_parquet_loader_is_data_loader(self):
        assert isinstance(ParquetLoader(), DataLoader)


class TestLoadCsv:
    def test_load_csv(self):
        df, fmt = load_dataframe(SAMPLE_CSV)
        assert isinstance(df, pd.DataFrame)
        assert fmt == "csv"
        assert len(df) == 20
        assert "revenue" in df.columns

    def test_load_csv_columns(self):
        df, _ = load_dataframe(SAMPLE_CSV)
        expected = {"id", "category", "revenue", "signup_date", "is_active", "notes"}
        assert set(df.columns) == expected


class TestLoadErrors:
    def test_unsupported_format(self):
        with pytest.raises(DataError, match="No loader"):
            load_dataframe(Path("data.txt"))

    def test_missing_file(self):
        with pytest.raises(DataError):
            load_dataframe(Path("nonexistent.csv"))


class TestLoaderRegistry:
    def test_register_custom_loader(self):
        class JsonLoader:
            format_name: str = "json"

            def supports(self, path: Path) -> bool:
                return path.suffix.lower() == ".json"

            def load(self, path: Path) -> pd.DataFrame:
                return pd.DataFrame()

        initial_count = len(_LOADERS)
        loader = JsonLoader()
        register_loader(loader)
        assert len(_LOADERS) == initial_count + 1
        assert _LOADERS[-1] is loader
        # Clean up to avoid leaking into other tests
        _LOADERS.pop()
