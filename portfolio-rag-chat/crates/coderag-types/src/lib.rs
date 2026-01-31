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

/// Represents a Rust crate extracted from Cargo.toml
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrateChunk {
    pub crate_name: String,
    pub crate_path: String,           // Path to the crate directory
    pub description: Option<String>,  // From [package].description
    pub dependencies: Vec<String>,    // Workspace/local dependencies
    pub project_name: Option<String>, // Parent project (e.g., "quant-trading-gym")
}

/// Represents module-level documentation (//! comments at top of lib.rs)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModuleDocChunk {
    pub file_path: String,
    pub module_name: String, // Derived from file/crate name
    pub doc_content: String, // The //! doc comments
    pub project_name: Option<String>,
}
