use tree_sitter::{Language, Node};

/// Trait for language-specific parsing behavior.
///
/// Implement this trait to add support for a new programming language.
/// Each implementation handles grammar loading and query patterns for its language.
/// Docstring extraction (V1.4) will override the default `None` return.
pub trait LanguageHandler: Send + Sync {
    /// Language identifier (e.g., "rust", "python")
    fn name(&self) -> &'static str;

    /// File extensions this handler supports (e.g., &["rs"] for Rust)
    fn extensions(&self) -> &'static [&'static str];

    /// Get the tree-sitter grammar for this language
    fn grammar(&self) -> Language;

    /// Tree-sitter S-expression query for extracting code elements.
    ///
    /// Must capture:
    /// - `@name` - the identifier of the element
    /// - `@body` - the full element node
    fn query_string(&self) -> &'static str;

    /// Extract docstring from a code element.
    ///
    /// Default returns None. V1.4 will add per-language implementations.
    fn extract_docstring(
        &self,
        _source: &str,
        _node: &Node,
        _source_bytes: &[u8],
    ) -> Option<String> {
        None
    }
}
