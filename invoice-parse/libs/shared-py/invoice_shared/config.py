"""Configuration loader — reads YAML config, returns typed AppConfig."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    url: str


class BlobStorageConfig(BaseModel):
    type: str  # "local_fs" or "s3"
    base_path: str | None = None  # local_fs
    bucket: str | None = None  # s3
    region: str | None = None  # s3


class QueueConfig(BaseModel):
    type: str  # "redis_stream" or "sqs"
    url: str | None = None  # redis_stream
    consumer_group: str = "invoice_workers"
    queue_a_url: str | None = None  # sqs
    queue_b_url: str | None = None  # sqs
    region: str | None = None  # sqs


class AppConfig(BaseModel):
    database: DatabaseConfig
    blob_storage: BlobStorageConfig
    queue: QueueConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML file.

    Resolution order:
    1. Explicit path argument
    2. INVOICE_PARSE_CONFIG env var
    3. config/local.yaml (relative to cwd)
    """
    if path is None:
        path = os.environ.get("INVOICE_PARSE_CONFIG", "config/local.yaml")
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
