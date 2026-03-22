"""Blob storage adapter — abstract interface + local filesystem implementation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


class BlobStore(ABC):
    @abstractmethod
    def put(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, path: str) -> bytes: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def delete(self, path: str) -> None: ...


class LocalFsBlobStore(BlobStore):
    """Local filesystem blob store with path safety validation."""

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, path: str) -> Path:
        if ".." in path:
            raise ValueError(f"Path traversal detected in blob path: {path}")

        # Validate UUID segments (tenant_id/job_id)
        parts = Path(path).parts
        # Strip leading separator
        clean_parts = [p for p in parts if p != "/"]
        if len(clean_parts) >= 2:
            tenant_id, job_id = clean_parts[0], clean_parts[1]
            if not _UUID_RE.match(tenant_id):
                raise ValueError(f"Invalid tenant_id in blob path: {tenant_id}")
            if not _UUID_RE.match(job_id):
                raise ValueError(f"Invalid job_id in blob path: {job_id}")

        resolved = (self._base / path.lstrip("/")).resolve()
        if not str(resolved).startswith(str(self._base)):
            raise ValueError(f"Path escapes base directory: {path}")
        return resolved

    def put(self, path: str, data: bytes) -> None:
        full = self._safe_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    def get(self, path: str) -> bytes:
        return self._safe_path(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._safe_path(path).exists()

    def delete(self, path: str) -> None:
        p = self._safe_path(path)
        if p.exists():
            p.unlink()
