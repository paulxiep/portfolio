"""Factory functions — create adapters from config."""

from __future__ import annotations

from ..config import AppConfig
from .blob_store import BlobStore, LocalFsBlobStore
from .queue import MessageQueue, RedisStreamQueue


def create_blob_store(config: AppConfig) -> BlobStore:
    match config.blob_storage.type:
        case "local_fs":
            if not config.blob_storage.base_path:
                raise ValueError("local_fs blob storage requires base_path")
            return LocalFsBlobStore(config.blob_storage.base_path)
        case "s3":
            raise NotImplementedError("S3BlobStore not yet implemented")
        case _:
            raise ValueError(f"Unknown blob storage type: {config.blob_storage.type}")


def create_queue(config: AppConfig) -> MessageQueue:
    match config.queue.type:
        case "redis_stream":
            if not config.queue.url:
                raise ValueError("redis_stream queue requires url")
            return RedisStreamQueue(config.queue.url, config.queue.consumer_group)
        case "sqs":
            raise NotImplementedError("SqsQueue not yet implemented")
        case _:
            raise ValueError(f"Unknown queue type: {config.queue.type}")
