use serde::{Deserialize, Serialize};

/// POST /chat request
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub query: String,
}

/// POST /chat response
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub answer: String,
    pub sources: Vec<SourceInfo>,
}

/// Source reference in response
#[derive(Debug, Serialize)]
pub struct SourceInfo {
    pub file: String,
    pub function: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub line: usize,
}

/// POST /ingest request
#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    pub repo_path: String,
}

/// POST /ingest response
#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub code_chunks: usize,
    pub readme_chunks: usize,
    pub crate_chunks: usize,
    pub module_doc_chunks: usize,
    pub message: String,
}

/// GET /health response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

/// GET /projects response
#[derive(Debug, Serialize)]
pub struct ProjectsResponse {
    pub projects: Vec<String>,
    pub count: usize,
}
