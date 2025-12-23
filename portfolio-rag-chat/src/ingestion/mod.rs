pub mod parser;

use self::parser::{CodeAnalyzer, SupportedLanguage};
use crate::models::{CodeChunk, ReadmeChunk};
use std::path::{Path, PathBuf};
use tracing::warn;
use walkdir::{DirEntry, WalkDir};

/// Extract project name from file path based on directory structure
/// Returns the top-level directory name relative to the repo root
fn extract_project_name(path: &Path, repo_root: &Path) -> Option<String> {
    let relative = path.strip_prefix(repo_root).ok()?;
    let mut components = relative.components();
    let first = components.next()?;

    // If this is the only component, it's a root-level file
    components.next()?;

    first.as_os_str().to_str().map(|s| s.to_string())
}

/// Check if entry is a README file
fn is_readme(entry: &DirEntry) -> bool {
    entry
        .path()
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.eq_ignore_ascii_case("readme.md"))
        .unwrap_or(false)
}

/// Process a single README file
fn process_readme(entry: &DirEntry, repo_root: &Path) -> Option<ReadmeChunk> {
    let content = std::fs::read_to_string(entry.path()).ok()?;
    let project_name =
        extract_project_name(entry.path(), repo_root).unwrap_or_else(|| "root".to_string());

    Some(ReadmeChunk {
        file_path: entry.path().to_string_lossy().to_string(),
        project_name,
        content,
    })
}

/// Process a single code file
fn process_code_file(
    entry: &DirEntry,
    repo_root: &Path,
    analyzer: &mut CodeAnalyzer,
) -> Vec<CodeChunk> {
    let lang = match SupportedLanguage::from_path(entry.path()) {
        Some(l) => l,
        None => return Vec::new(),
    };

    let content = match std::fs::read_to_string(entry.path()) {
        Ok(c) => c,
        Err(e) => {
            warn!(path = %entry.path().display(), error = %e, "Failed to read file");
            return Vec::new();
        }
    };

    let path_str = entry.path().to_string_lossy().to_string();
    let project_name = extract_project_name(entry.path(), repo_root);

    analyzer
        .analyze_content(&content, lang)
        .into_iter()
        .map(|mut chunk| {
            chunk.file_path = path_str.clone();
            chunk.project_name = project_name.clone();
            chunk
        })
        .collect()
}

/// Directories to skip during ingestion
const IGNORED_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "dist",
    "build",
];

/// Check if path contains any ignored directory (relative to repo root)
fn should_skip(entry: &DirEntry, repo_root: &Path) -> bool {
    // Only check components within the repo, not the full system path
    let relative = match entry.path().strip_prefix(repo_root) {
        Ok(r) => r,
        Err(_) => return false,
    };

    relative.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map(|s| IGNORED_DIRS.contains(&s) || s.starts_with('.'))
            .unwrap_or(false)
    })
}

/// This orchestrates the flow from Disk -> Parser -> Data
pub fn run_ingestion(repo_path: &str) -> (Vec<CodeChunk>, Vec<ReadmeChunk>) {
    let repo_root = PathBuf::from(repo_path);
    let mut analyzer = CodeAnalyzer::new();

    let entries: Vec<_> = WalkDir::new(repo_path)
        .follow_links(false) // Don't follow symlinks to avoid cycles
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| !should_skip(e, &repo_root))
        .collect();

    let readme_chunks: Vec<ReadmeChunk> = entries
        .iter()
        .filter(|e| is_readme(e))
        .filter_map(|e| process_readme(e, &repo_root))
        .collect();

    let code_chunks: Vec<CodeChunk> = entries
        .iter()
        .flat_map(|e| process_code_file(e, &repo_root, &mut analyzer))
        .collect();

    (code_chunks, readme_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_workspace() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        // Create project1 with a Python file and README
        fs::create_dir(base.join("project1")).unwrap();
        fs::write(
            base.join("project1/main.py"),
            "def hello():\n    print('world')\n",
        )
        .unwrap();
        fs::write(
            base.join("project1/README.md"),
            "# Project 1\nA test project",
        )
        .unwrap();

        // Create project2 with a Rust file
        fs::create_dir(base.join("project2")).unwrap();
        fs::write(base.join("project2/lib.rs"), "fn test() {}\n").unwrap();

        temp_dir
    }

    #[test]
    fn test_extract_project_name() {
        let repo_root = Path::new("/home/user/portfolio");
        let file_path = Path::new("/home/user/portfolio/my_project/src/main.rs");

        let project_name = extract_project_name(file_path, repo_root);
        assert_eq!(project_name, Some("my_project".to_string()));
    }

    #[test]
    fn test_extract_project_name_root_file() {
        let repo_root = Path::new("/home/user/portfolio");
        let file_path = Path::new("/home/user/portfolio/README.md");

        let project_name = extract_project_name(file_path, repo_root);
        // Root level files have no project subdirectory
        assert!(project_name.is_none());
    }

    #[test]
    fn test_is_readme() {
        let temp_dir = TempDir::new().unwrap();
        let readme_path = temp_dir.path().join("README.md");
        fs::write(&readme_path, "test").unwrap();

        let entry = WalkDir::new(temp_dir.path())
            .into_iter()
            .find(|e| e.as_ref().unwrap().path() == readme_path)
            .unwrap()
            .unwrap();

        assert!(is_readme(&entry));
    }

    #[test]
    fn test_is_readme_case_insensitive() {
        let temp_dir = TempDir::new().unwrap();
        let readme_path = temp_dir.path().join("readme.MD");
        fs::write(&readme_path, "test").unwrap();

        let entry = WalkDir::new(temp_dir.path())
            .into_iter()
            .find(|e| e.as_ref().unwrap().path() == readme_path)
            .unwrap()
            .unwrap();

        assert!(is_readme(&entry));
    }

    #[test]
    fn test_run_ingestion_full_pipeline() {
        let temp_dir = create_test_workspace();
        let path = temp_dir.path().to_str().unwrap();

        let (code_chunks, readme_chunks) = run_ingestion(path);

        // Should find 2 code files (main.py and lib.rs)
        assert_eq!(code_chunks.len(), 2);

        // Should find 1 README
        assert_eq!(readme_chunks.len(), 1);

        // Check Python function was extracted
        assert!(
            code_chunks
                .iter()
                .any(|c| c.identifier == "hello" && c.language == "python")
        );

        // Check Rust function was extracted
        assert!(
            code_chunks
                .iter()
                .any(|c| c.identifier == "test" && c.language == "rust")
        );

        // Check project names were set
        assert!(
            code_chunks
                .iter()
                .any(|c| c.project_name.as_deref() == Some("project1"))
        );
        assert!(
            code_chunks
                .iter()
                .any(|c| c.project_name.as_deref() == Some("project2"))
        );

        // Check README content
        assert_eq!(readme_chunks[0].project_name, "project1");
        assert!(readme_chunks[0].content.contains("Project 1"));
    }

    #[test]
    fn test_process_code_file_enriches_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();

        fs::create_dir(base.join("myproj")).unwrap();
        let file_path = base.join("myproj/code.py");
        fs::write(&file_path, "def func(): pass\n").unwrap();

        let entry = WalkDir::new(base)
            .into_iter()
            .find(|e| e.as_ref().unwrap().path() == file_path)
            .unwrap()
            .unwrap();

        let mut analyzer = CodeAnalyzer::new();
        let chunks = process_code_file(&entry, base, &mut analyzer);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].file_path.contains("code.py"));
        assert_eq!(chunks[0].project_name.as_deref(), Some("myproj"));
    }
}
