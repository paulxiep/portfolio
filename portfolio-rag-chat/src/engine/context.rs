use super::retriever::RetrievalResult;

/// System prompt - instructs the LLM how to behave
pub const SYSTEM_PROMPT: &str = r#"You are a helpful assistant answering questions about a developer's portfolio of coding projects.

Guidelines:
- Use the provided code snippets and documentation to give accurate, specific answers
- Reference project names, file paths, and function names when relevant
- If the context doesn't contain enough information, say so honestly
- Be concise but thorough
- When explaining code, focus on what it does and why, not line-by-line narration"#;

/// Format retrieved chunks into context for the LLM
pub fn build_context(result: &RetrievalResult) -> String {
    let mut sections = Vec::new();

    // Code chunks
    if !result.code_chunks.is_empty() {
        sections.push(format_code_section(&result.code_chunks));
    }

    // README chunks
    if !result.readme_chunks.is_empty() {
        sections.push(format_readme_section(&result.readme_chunks));
    }

    if sections.is_empty() {
        return "No relevant code or documentation found.".into();
    }

    sections.join("\n\n")
}

fn format_code_section(chunks: &[crate::models::CodeChunk]) -> String {
    let mut out = String::from("## Relevant Code\n");

    for chunk in chunks {
        let project = chunk.project_name.as_deref().unwrap_or("(root)");
        out.push_str(&format!(
            "\n### `{}` in {} ({}:{})\n```{}\n{}\n```\n",
            chunk.identifier,
            project,
            chunk.file_path,
            chunk.start_line,
            chunk.language,
            chunk.code_content.trim()
        ));
    }

    out
}

fn format_readme_section(chunks: &[crate::models::ReadmeChunk]) -> String {
    let mut out = String::from("## Project Documentation\n");

    for chunk in chunks {
        out.push_str(&format!(
            "\n### {}\n{}\n",
            chunk.project_name,
            truncate(&chunk.content, 800)
        ));
    }

    out
}

/// Build the complete prompt sent to the LLM
pub fn build_prompt(query: &str, context: &str) -> String {
    format!(
        "{system}\n\n---\n\n{context}\n\n---\n\n**Question:** {query}",
        system = SYSTEM_PROMPT,
        context = context,
        query = query
    )
}

/// Truncate string to max characters, respecting word boundaries
fn truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        return s;
    }

    // Find last space before limit
    s[..max_chars]
        .rfind(' ')
        .map(|idx| &s[..idx])
        .unwrap_or(&s[..max_chars])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{CodeChunk, ReadmeChunk};

    fn sample_code_chunk() -> CodeChunk {
        CodeChunk {
            file_path: "my_project/src/lib.rs".into(),
            language: "rust".into(),
            identifier: "process_data".into(),
            node_type: "function_item".into(),
            code_content: "fn process_data(input: &str) -> Result<Output, Error> {\n    // ...\n}"
                .into(),
            start_line: 42,
            project_name: Some("my_project".into()),
            docstring: None,
        }
    }

    fn sample_readme_chunk() -> ReadmeChunk {
        ReadmeChunk {
            file_path: "my_project/README.md".into(),
            project_name: "my_project".into(),
            content: "# My Project\n\nA data processing library.".into(),
        }
    }

    #[test]
    fn test_build_context_with_code() {
        let result = RetrievalResult {
            code_chunks: vec![sample_code_chunk()],
            readme_chunks: vec![],
        };

        let context = build_context(&result);

        assert!(context.contains("## Relevant Code"));
        assert!(context.contains("process_data"));
        assert!(context.contains("my_project"));
        assert!(context.contains("```rust"));
    }

    #[test]
    fn test_build_context_with_readme() {
        let result = RetrievalResult {
            code_chunks: vec![],
            readme_chunks: vec![sample_readme_chunk()],
        };

        let context = build_context(&result);

        assert!(context.contains("## Project Documentation"));
        assert!(context.contains("My Project"));
    }

    #[test]
    fn test_build_context_empty() {
        let result = RetrievalResult {
            code_chunks: vec![],
            readme_chunks: vec![],
        };

        let context = build_context(&result);

        assert!(context.contains("No relevant"));
    }

    #[test]
    fn test_build_prompt_structure() {
        let prompt = build_prompt("What does process_data do?", "## Code\n...");

        assert!(prompt.contains(SYSTEM_PROMPT));
        assert!(prompt.contains("## Code"));
        assert!(prompt.contains("What does process_data do?"));
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_at_word_boundary() {
        let result = truncate("hello world foo bar", 12);
        assert_eq!(result, "hello world");
    }
}
