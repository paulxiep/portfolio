use coderag_types::{CodeChunk, content_hash, new_chunk_id};
use std::path::Path;
use tracing::warn;
use tree_sitter::{Parser, Query, StreamingIterator};

use super::language::LanguageHandler;
use super::languages::{RustHandler, handler_for_path};

/// Default embedding model version (matches embedder.rs)
const DEFAULT_EMBEDDING_MODEL: &str = "BGESmallENV15_384";

/// The Parser Engine (Logic Layer)
pub struct CodeAnalyzer {
    parser: Parser,
}

impl CodeAnalyzer {
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
        }
    }

    /// Analyze a file, auto-detecting language from path
    pub fn analyze_file(&mut self, path: &Path, source: &str) -> Vec<CodeChunk> {
        let Some(handler) = handler_for_path(path) else {
            return Vec::new();
        };
        self.analyze_with_handler(source, handler)
    }

    /// Analyze source code with a specific handler
    pub fn analyze_with_handler(
        &mut self,
        source: &str,
        handler: &dyn LanguageHandler,
    ) -> Vec<CodeChunk> {
        let grammar = handler.grammar();
        if let Err(e) = self.parser.set_language(&grammar) {
            warn!(language = handler.name(), error = ?e, "Failed to set parser language");
            return Vec::new();
        }

        let Some(tree) = self.parser.parse(source, None) else {
            warn!(language = handler.name(), "Failed to parse source code");
            return Vec::new();
        };

        let query = match Query::new(&grammar, handler.query_string()) {
            Ok(q) => q,
            Err(e) => {
                warn!(language = handler.name(), error = ?e, "Failed to create tree-sitter query");
                return Vec::new();
            }
        };

        let mut cursor = tree_sitter::QueryCursor::new();
        let source_bytes = source.as_bytes();

        // Get capture indices by name for reliable lookup across multiple patterns
        let name_idx = query.capture_index_for_name("name");
        let body_idx = query.capture_index_for_name("body");

        // Use StreamingIterator::fold to collect matches
        let raw_matches: Vec<(String, String, String, usize)> = cursor
            .captures(&query, tree.root_node(), source_bytes)
            .fold(Vec::new(), |mut acc, (m, _)| {
                let body = m.captures.iter().find(|c| Some(c.index) == body_idx);
                let name = m.captures.iter().find(|c| Some(c.index) == name_idx);
                if let (Some(b), Some(n)) = (body, name) {
                    acc.push((
                        b.node.kind().to_string(),
                        n.node
                            .utf8_text(source_bytes)
                            .unwrap_or("unknown")
                            .to_string(),
                        b.node.utf8_text(source_bytes).unwrap_or("").to_string(),
                        b.node.start_position().row + 1,
                    ));
                }
                acc
            });

        // Transform to CodeChunks using functional style
        let mut chunks: Vec<CodeChunk> = raw_matches
            .into_iter()
            .map(|(node_type, identifier, code_content, start_line)| {
                let hash = content_hash(&code_content);
                CodeChunk {
                    file_path: "<set_by_caller>".to_string(),
                    language: handler.name().to_string(),
                    identifier,
                    node_type,
                    start_line,
                    project_name: String::new(),
                    docstring: None, // V1.4 will wire handler.extract_docstring() here
                    chunk_id: new_chunk_id(),
                    content_hash: hash,
                    embedding_model_version: DEFAULT_EMBEDDING_MODEL.to_string(),
                    code_content,
                }
            })
            .collect();

        // Deduplicate by (identifier, start_line) since impl blocks may capture methods multiple times
        chunks.sort_by(|a, b| (&a.identifier, a.start_line).cmp(&(&b.identifier, b.start_line)));
        chunks.dedup_by(|a, b| a.identifier == b.identifier && a.start_line == b.start_line);

        chunks
    }

    /// Extract module-level documentation comments (//! lines) from Rust source.
    /// Returns the concatenated doc content if found.
    pub fn extract_module_docs(&mut self, source: &str) -> Option<String> {
        let handler = RustHandler;
        let grammar = handler.grammar();
        if self.parser.set_language(&grammar).is_err() {
            return None;
        }

        let tree = self.parser.parse(source, None)?;
        let source_bytes = source.as_bytes();
        let mut cursor = tree.root_node().walk();

        let doc_lines: Vec<String> = tree
            .root_node()
            .children(&mut cursor)
            .map_while(|child| match child.kind() {
                "line_comment" => {
                    let text = child.utf8_text(source_bytes).ok()?;
                    text.starts_with("//!").then(|| {
                        text.strip_prefix("//!")
                            .unwrap_or(text)
                            .strip_prefix(' ')
                            .unwrap_or(text.strip_prefix("//!").unwrap_or(text))
                            .to_string()
                    })
                }
                "inner_line_doc" | "inner_line_doc_comment" => {
                    child.utf8_text(source_bytes).ok().map(String::from)
                }
                _ => None,
            })
            .collect();

        (!doc_lines.is_empty()).then(|| doc_lines.join("\n"))
    }
}

impl Default for CodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a Cargo.toml file and extract crate metadata.
/// Returns (crate_name, description, local_dependencies)
pub fn parse_cargo_toml(content: &str) -> Option<(String, Option<String>, Vec<String>)> {
    let parsed: toml::Value = content.parse().ok()?;

    let package = parsed.get("package")?;
    let crate_name = package.get("name")?.as_str()?.to_string();
    let description = package
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());

    // Extract workspace/path dependencies (local crates)
    let local_deps: Vec<String> = parsed
        .get("dependencies")
        .and_then(|d| d.as_table())
        .map(|deps_table| {
            deps_table
                .iter()
                .filter(|(_, value)| {
                    value
                        .as_table()
                        .map(|t| t.contains_key("path"))
                        .unwrap_or(false)
                })
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default();

    Some((crate_name, description, local_deps))
}

#[cfg(test)]
mod tests {
    use super::super::languages::{PythonHandler, handler_for_path};
    use super::*;

    #[test]
    fn test_handler_for_path() {
        assert!(handler_for_path(Path::new("test.rs")).is_some());
        assert!(handler_for_path(Path::new("test.py")).is_some());
        assert!(handler_for_path(Path::new("test.js")).is_some());
        assert!(handler_for_path(Path::new("test.go")).is_none());
        assert!(handler_for_path(Path::new("test")).is_none());
    }

    #[test]
    fn test_handler_names() {
        assert_eq!(RustHandler.name(), "rust");
        assert_eq!(PythonHandler.name(), "python");
    }

    #[test]
    fn test_parse_rust_function() {
        let source = r#"
            fn hello_world() {
                println!("Hello!");
            }
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &RustHandler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].identifier, "hello_world");
        assert_eq!(chunks[0].language, "rust");
        assert_eq!(chunks[0].node_type, "function_item");
        assert!(chunks[0].code_content.contains("println!"));
    }

    #[test]
    fn test_parse_python_function() {
        let source = r#"
def greet(name):
    print(f"Hello {name}")
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &PythonHandler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].identifier, "greet");
        assert_eq!(chunks[0].language, "python");
        assert_eq!(chunks[0].node_type, "function_definition");
    }

    #[test]
    fn test_parse_python_class() {
        let source = r#"
class MyClass:
    def __init__(self):
        pass
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &PythonHandler);

        assert!(
            chunks
                .iter()
                .any(|c| c.identifier == "MyClass" && c.node_type == "class_definition")
        );
    }

    #[test]
    fn test_parse_multiple_functions() {
        let source = r#"
fn first() {}
fn second() {}
fn third() {}
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &RustHandler);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].identifier, "first");
        assert_eq!(chunks[1].identifier, "second");
        assert_eq!(chunks[2].identifier, "third");
    }

    #[test]
    fn test_invalid_syntax_returns_empty() {
        let source = "fn invalid {{{";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &RustHandler);

        // Should handle gracefully - returns empty on parse errors
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_line_numbers() {
        let source = "fn test() {}";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &RustHandler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1); // 1-indexed, first line
    }

    #[test]
    fn test_rust_impl_method() {
        let source = r#"
impl MyStruct {
    fn method(&self) {
        println!("method");
    }
}
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &RustHandler);

        // Should capture method inside impl block
        assert!(chunks.iter().any(|c| c.identifier == "method"));
    }
}
