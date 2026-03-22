"""Shared test fixtures for invoice_parse integration tests."""

import os
import tempfile

import pytest

# Point config to local.yaml relative to project root
os.environ.setdefault("INVOICE_PARSE_CONFIG", "config/local.yaml")


@pytest.fixture
def tmp_blob_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d
