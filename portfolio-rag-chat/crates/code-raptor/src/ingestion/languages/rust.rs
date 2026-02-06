use super::super::language::LanguageHandler;
use tree_sitter::Language;

pub struct RustHandler;

impl LanguageHandler for RustHandler {
    fn name(&self) -> &'static str {
        "rust"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["rs"]
    }

    fn grammar(&self) -> Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn query_string(&self) -> &'static str {
        r#"(function_item name: (identifier) @name) @body
(struct_item name: (type_identifier) @name) @body
(enum_item name: (type_identifier) @name) @body
(trait_item name: (type_identifier) @name) @body
(impl_item type: (type_identifier) @name) @body
(type_item name: (type_identifier) @name) @body
(macro_definition name: (identifier) @name) @body"#
    }

    // extract_docstring: uses default (None) until V1.4
}
