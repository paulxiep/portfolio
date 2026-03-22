use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BlobError {
    #[error("Path traversal detected: {0}")]
    PathTraversal(String),
    #[error("Invalid UUID segment in path: {0}")]
    InvalidUuid(String),
    #[error("Path escapes base directory: {0}")]
    PathEscape(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Abstract blob storage interface.
pub trait BlobStore: Send + Sync {
    fn put(&self, path: &str, data: &[u8]) -> Result<(), BlobError>;
    fn get(&self, path: &str) -> Result<Vec<u8>, BlobError>;
    fn exists(&self, path: &str) -> Result<bool, BlobError>;
    fn delete(&self, path: &str) -> Result<(), BlobError>;
}

/// Local filesystem blob store with path safety validation.
pub struct LocalFsBlobStore {
    base: PathBuf,
}

impl LocalFsBlobStore {
    pub fn new(base_path: &str) -> Result<Self, BlobError> {
        let base = Path::new(base_path).canonicalize().or_else(|_| {
            std::fs::create_dir_all(base_path)?;
            Path::new(base_path).canonicalize()
        })?;
        Ok(Self { base })
    }

    fn safe_path(&self, path: &str) -> Result<PathBuf, BlobError> {
        if path.contains("..") {
            return Err(BlobError::PathTraversal(path.to_string()));
        }

        // Validate UUID segments (tenant_id/job_id)
        let clean = path.trim_start_matches('/');
        let parts: Vec<&str> = clean.split('/').collect();
        if parts.len() >= 2 {
            for segment in &parts[..2] {
                if uuid::Uuid::parse_str(segment).is_err() {
                    return Err(BlobError::InvalidUuid(segment.to_string()));
                }
            }
        }

        let full = self.base.join(clean);
        // Canonicalize parent to check prefix (file may not exist yet)
        let parent = full.parent().unwrap_or(&full);
        std::fs::create_dir_all(parent)?;
        let resolved_parent = parent.canonicalize()?;
        if !resolved_parent.starts_with(&self.base) {
            return Err(BlobError::PathEscape(path.to_string()));
        }
        Ok(full)
    }
}

impl BlobStore for LocalFsBlobStore {
    fn put(&self, path: &str, data: &[u8]) -> Result<(), BlobError> {
        let full = self.safe_path(path)?;
        std::fs::write(full, data)?;
        Ok(())
    }

    fn get(&self, path: &str) -> Result<Vec<u8>, BlobError> {
        let full = self.safe_path(path)?;
        Ok(std::fs::read(full)?)
    }

    fn exists(&self, path: &str) -> Result<bool, BlobError> {
        let full = self.safe_path(path)?;
        Ok(full.exists())
    }

    fn delete(&self, path: &str) -> Result<(), BlobError> {
        let full = self.safe_path(path)?;
        if full.exists() {
            std::fs::remove_file(full)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, LocalFsBlobStore) {
        let dir = TempDir::new().unwrap();
        let store = LocalFsBlobStore::new(dir.path().to_str().unwrap()).unwrap();
        (dir, store)
    }

    #[test]
    fn put_get_exists_delete_roundtrip() {
        let (_dir, store) = setup();
        let path = "a0000000-0000-0000-0000-000000000001/b0000000-0000-0000-0000-000000000002/input.pdf";
        let data = b"hello world";

        assert!(!store.exists(path).unwrap());
        store.put(path, data).unwrap();
        assert!(store.exists(path).unwrap());
        assert_eq!(store.get(path).unwrap(), data);
        store.delete(path).unwrap();
        assert!(!store.exists(path).unwrap());
    }

    #[test]
    fn rejects_path_traversal() {
        let (_dir, store) = setup();
        let result = store.put("../escape/file.txt", b"bad");
        assert!(matches!(result, Err(BlobError::PathTraversal(_))));
    }

    #[test]
    fn rejects_non_uuid_segments() {
        let (_dir, store) = setup();
        let result = store.put("not-a-uuid/also-bad/file.txt", b"data");
        assert!(matches!(result, Err(BlobError::InvalidUuid(_))));
    }
}
