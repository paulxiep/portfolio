use super::super::language::LanguageHandler;
use tree_sitter::{Language, Node};

pub struct PythonHandler;

impl LanguageHandler for PythonHandler {
    fn name(&self) -> &'static str {
        "python"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["py"]
    }

    fn grammar(&self) -> Language {
        tree_sitter_python::LANGUAGE.into()
    }

    fn query_string(&self) -> &'static str {
        r#"(function_definition name: (identifier) @name) @body
(class_definition name: (identifier) @name) @body"#
    }

    fn extract_docstring(&self, _source: &str, node: &Node, source_bytes: &[u8]) -> Option<String> {
        // Python docstrings are the first expression_statement in the body block
        let body = node.child_by_field_name("body")?;
        let first_stmt = body.named_child(0)?;

        if first_stmt.kind() != "expression_statement" {
            return None;
        }

        let expr = first_stmt.named_child(0)?;
        if expr.kind() != "string" {
            return None;
        }

        let raw = expr.utf8_text(source_bytes).ok()?;
        parse_python_docstring(raw)
    }
}

/// Parse a Python string literal into clean docstring text
fn parse_python_docstring(raw: &str) -> Option<String> {
    let trimmed = raw.trim();

    // Determine quote style and strip delimiters
    let content = if (trimmed.starts_with("\"\"\"") && trimmed.ends_with("\"\"\""))
        || (trimmed.starts_with("'''") && trimmed.ends_with("'''"))
    {
        &trimmed[3..trimmed.len() - 3]
    } else if (trimmed.starts_with('"') && trimmed.ends_with('"')
        || trimmed.starts_with('\'') && trimmed.ends_with('\''))
        && trimmed.len() >= 2
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return None;
    };

    let cleaned = dedent_docstring(content);

    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Remove common leading whitespace from multi-line docstring
fn dedent_docstring(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();

    if lines.is_empty() {
        return String::new();
    }

    let first_line = lines[0].trim();

    if lines.len() == 1 {
        return first_line.to_string();
    }

    // Find minimum indentation of non-empty lines (excluding first)
    let min_indent = lines[1..]
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.len() - line.trim_start().len())
        .min()
        .unwrap_or(0);

    let mut result = Vec::new();

    if !first_line.is_empty() {
        result.push(first_line.to_string());
    }

    for line in &lines[1..] {
        if line.trim().is_empty() {
            result.push(String::new());
        } else if line.len() >= min_indent {
            result.push(line[min_indent..].to_string());
        } else {
            result.push(line.trim().to_string());
        }
    }

    // Trim trailing empty lines
    while result.last().map(|s| s.is_empty()).unwrap_or(false) {
        result.pop();
    }

    result.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::parser::CodeAnalyzer;

    #[test]
    fn test_python_doc_simple() {
        let handler = PythonHandler;
        let source = "def foo():\n    \"\"\"Simple docstring.\"\"\"\n    pass";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].docstring, Some("Simple docstring.".to_string()));
    }

    #[test]
    fn test_python_doc_multiline() {
        let handler = PythonHandler;
        let source = "def foo():\n    \"\"\"\n    Line one.\n    Line two.\n    \"\"\"\n    pass";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].docstring,
            Some("Line one.\nLine two.".to_string())
        );
    }

    #[test]
    fn test_python_doc_class() {
        let handler = PythonHandler;
        let source = "class Foo:\n    \"\"\"Class docstring.\"\"\"\n    pass";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert!(
            chunks
                .iter()
                .any(|c| c.identifier == "Foo"
                    && c.docstring == Some("Class docstring.".to_string()))
        );
    }

    #[test]
    fn test_python_doc_single_quotes() {
        let handler = PythonHandler;
        let source = "def foo():\n    '''Single quote docs.'''\n    pass";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].docstring, Some("Single quote docs.".to_string()));
    }

    #[test]
    fn test_python_no_doc() {
        let handler = PythonHandler;
        let source = "def foo():\n    x = 1\n    return x";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].docstring, None);
    }

    #[test]
    fn test_python_doc_pipeline() {
        let handler = PythonHandler;
        let source =
            "def greet(name):\n    \"\"\"Return a greeting.\"\"\"\n    return f\"Hello {name}\"";

        let mut analyzer = CodeAnalyzer::new();
        let chunks = analyzer.analyze_with_handler(source, &handler);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].identifier, "greet");
        assert_eq!(chunks[0].docstring, Some("Return a greeting.".to_string()));
    }
}
