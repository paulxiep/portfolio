use super::super::language::LanguageHandler;
use tree_sitter::Language;

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

    // extract_docstring: uses default (None) until V1.4
}
