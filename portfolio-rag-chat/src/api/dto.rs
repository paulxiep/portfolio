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
    pub project: String,
    pub line: usize,
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
