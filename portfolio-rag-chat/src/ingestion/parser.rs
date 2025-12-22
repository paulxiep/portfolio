use crate::models::CodeChunk;
use streaming_iterator::StreamingIterator;
use tracing::warn;
use tree_sitter::{Parser, Query};

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

    /// Returns the S-expression query for functions/methods/classes
    pub fn query_string(&self) -> &'static str {
        match self {
            Self::Rust => "(function_item name: (identifier) @name) @body",
            Self::Python => {
                r#"
                (function_definition name: (identifier) @name) @body
                (class_definition name: (identifier) @name) @body
            "#
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

        let Ok(query) = Query::new(&grammar, lang.query_string()) else {
            warn!(language = lang.name(), "Failed to create tree-sitter query");
            return Vec::new();
        };

        let mut cursor = tree_sitter::QueryCursor::new();
        let source_bytes = source.as_bytes();

        cursor
            .captures(&query, tree.root_node(), source_bytes)
            .filter(|(m, capture_idx)| m.captures.len() >= 2 && *capture_idx == 1)
            .fold(Vec::new(), |mut acc, (m, _)| {
                let body_node = m.captures[0].node;
                let name_node = m.captures[1].node;
                acc.push(CodeChunk {
                    file_path: "<set_by_caller>".to_string(),
                    language: lang.name().to_string(),
                    identifier: name_node
                        .utf8_text(source_bytes)
                        .unwrap_or("unknown")
                        .to_string(),
                    node_type: body_node.kind().to_string(),
                    code_content: body_node.utf8_text(source_bytes).unwrap_or("").to_string(),
                    start_line: body_node.start_position().row + 1,
                    project_name: None,
                    docstring: None,
                });
                acc
            })
    }
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
