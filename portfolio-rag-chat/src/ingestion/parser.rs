use crate::models::CodeChunk;
use tracing::warn;
use tree_sitter::{Parser, Query, StreamingIterator};

#[derive(Debug, Clone, Copy)]
pub enum SupportedLanguage {
    Rust,
    Python,
}

impl SupportedLanguage {
    pub fn from_path(path: &std::path::Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            _ => None,
        }
    }

    pub fn get_grammar(&self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
        }
    }

    /// Returns the S-expression query for functions/methods/classes/structs/enums/traits
    pub fn query_string(&self) -> &'static str {
        match self {
            Self::Rust => {
                r#"(function_item name: (identifier) @name) @body
(struct_item name: (type_identifier) @name) @body
(enum_item name: (type_identifier) @name) @body
(trait_item name: (type_identifier) @name) @body
(impl_item type: (type_identifier) @name) @body
(type_item name: (type_identifier) @name) @body
(macro_definition name: (identifier) @name) @body"#
            }
            Self::Python => {
                r#"(function_definition name: (identifier) @name) @body
(class_definition name: (identifier) @name) @body"#
            }
        }
    }
}

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

    pub fn analyze_content(&mut self, source: &str, lang: SupportedLanguage) -> Vec<CodeChunk> {
        let grammar = lang.get_grammar();
        if let Err(e) = self.parser.set_language(&grammar) {
            warn!(language = lang.name(), error = ?e, "Failed to set parser language");
            return Vec::new();
        }

        let Some(tree) = self.parser.parse(source, None) else {
            warn!(language = lang.name(), "Failed to parse source code");
            return Vec::new();
        };

        let query = match Query::new(&grammar, lang.query_string()) {
            Ok(q) => q,
            Err(e) => {
                warn!(language = lang.name(), error = ?e, "Failed to create tree-sitter query");
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
            .map(
                |(node_type, identifier, code_content, start_line)| CodeChunk {
                    file_path: "<set_by_caller>".to_string(),
                    language: lang.name().to_string(),
                    identifier,
                    node_type,
                    code_content,
                    start_line,
                    project_name: None,
                    docstring: None,
                },
            )
            .collect();

        // Deduplicate by (identifier, start_line) since impl blocks may capture methods multiple times
        chunks.sort_by(|a, b| (&a.identifier, a.start_line).cmp(&(&b.identifier, b.start_line)));
        chunks.dedup_by(|a, b| a.identifier == b.identifier && a.start_line == b.start_line);

        chunks
    }

    /// Extract module-level documentation comments (//! lines) from Rust source.
    /// Returns the concatenated doc content if found.
    pub fn extract_module_docs(&mut self, source: &str) -> Option<String> {
        let grammar = SupportedLanguage::Rust.get_grammar();
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
    use super::*;
    use std::path::Path;

    #[test]
    fn test_language_detection_from_path() {
        assert!(matches!(
            SupportedLanguage::from_path(Path::new("test.rs")),
            Some(SupportedLanguage::Rust)
        ));
        assert!(matches!(
            SupportedLanguage::from_path(Path::new("test.py")),
            Some(SupportedLanguage::Python)
        ));
        assert!(SupportedLanguage::from_path(Path::new("test.js")).is_none());
        assert!(SupportedLanguage::from_path(Path::new("test")).is_none());
    }

    #[test]
    fn test_language_name() {
        assert_eq!(SupportedLanguage::Rust.name(), "rust");
        assert_eq!(SupportedLanguage::Python.name(), "python");
    }

    #[test]
    fn test_parse_rust_function() {
        let source = r#"
            fn hello_world() {
                println!("Hello!");
            }
        "#;

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Rust);

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
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Python);

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
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Python);

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
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Rust);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].identifier, "first");
        assert_eq!(chunks[1].identifier, "second");
        assert_eq!(chunks[2].identifier, "third");
    }

    #[test]
    fn test_invalid_syntax_returns_empty() {
        let source = "fn invalid {{{";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Rust);

        // Should handle gracefully - returns empty on parse errors
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_line_numbers() {
        let source = "fn test() {}";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Rust);

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
        let chunks = analyzer.analyze_content(source, SupportedLanguage::Rust);

        // Should capture method inside impl block
        assert!(chunks.iter().any(|c| c.identifier == "method"));
    }
}
