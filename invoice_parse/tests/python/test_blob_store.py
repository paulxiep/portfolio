"""Tests for LocalFsBlobStore — path safety, CRUD operations."""

import pytest

from invoice_shared.adapters.blob_store import LocalFsBlobStore

TENANT = "a0000000-0000-0000-0000-000000000001"
JOB = "b0000000-0000-0000-0000-000000000002"


class TestLocalFsBlobStore:
    def test_put_get_roundtrip(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        path = f"{TENANT}/{JOB}/input.pdf"
        data = b"fake pdf content"

        store.put(path, data)
        assert store.get(path) == data

    def test_exists(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        path = f"{TENANT}/{JOB}/input.pdf"

        assert not store.exists(path)
        store.put(path, b"data")
        assert store.exists(path)

    def test_delete(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        path = f"{TENANT}/{JOB}/input.pdf"

        store.put(path, b"data")
        assert store.exists(path)
        store.delete(path)
        assert not store.exists(path)

    def test_delete_nonexistent_is_noop(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        store.delete(f"{TENANT}/{JOB}/nope.txt")  # should not raise

    def test_rejects_path_traversal(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        with pytest.raises(ValueError, match="Path traversal"):
            store.put("../escape/file.txt", b"bad")

    def test_rejects_non_uuid_tenant(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        with pytest.raises(ValueError, match="Invalid tenant_id"):
            store.put("not-a-uuid/also-bad/file.txt", b"data")

    def test_rejects_non_uuid_job(self, tmp_blob_dir):
        store = LocalFsBlobStore(tmp_blob_dir)
        with pytest.raises(ValueError, match="Invalid job_id"):
            store.put(f"{TENANT}/not-a-uuid/file.txt", b"data")
