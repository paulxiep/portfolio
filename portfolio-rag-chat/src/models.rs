use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CodeChunk {
    pub file_path: String,
    pub language: String,     // "rust" or "python"
    pub identifier: String,   // Function/Class name
    pub node_type: String,    // "function_definition", "class_definition"
    pub code_content: String, // The actual snippet for the LLM
    pub start_line: usize,
    pub project_name: Option<String>, // e.g., "7_wonders", "catan"
    pub docstring: Option<String>,    // Extracted documentation
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReadmeChunk {
    pub file_path: String,
    pub project_name: String,
    pub content: String,
}
