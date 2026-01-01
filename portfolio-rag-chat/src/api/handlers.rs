use axum::{Json, extract::State};
use std::sync::Arc;

use super::dto::*;
use super::error::ApiError;
use super::state::AppState;
use crate::engine::{context, generator, retriever};
use crate::store::ingest_repository;

/// POST /chat - Ask a question about the portfolio
pub async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let query = req.query.trim();
    if query.is_empty() {
        return Err(ApiError::BadRequest("Query cannot be empty".into()));
    }

    // Retrieve relevant chunks (handles embedding + search)
    let result = {
        let mut embedder = state.embedder.lock().await;
        retriever::retrieve(query, &mut embedder, &state.store, &state.config.retrieval).await?
    };

    // Build context (pure function)
    let context = context::build_context(&result);
    let prompt = context::build_prompt(query, &context);

    // LLM call runs without any lock (slow: 2-5 seconds)
    let answer = generator::generate(&prompt, &state.llm).await?;

    // Build response
    let sources = result
        .code_chunks
        .into_iter()
        .map(|c| SourceInfo {
            file: c.file_path,
            function: c.identifier,
            project: c.project_name,
            line: c.start_line,
        })
        .collect();

    Ok(Json(ChatResponse { answer, sources }))
}

/// POST /ingest - Ingest a repository
pub async fn ingest(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IngestRequest>,
) -> Result<Json<IngestResponse>, ApiError> {
    // Validate path exists
    let path = req.repo_path.trim();
    if !std::path::Path::new(path).exists() {
        return Err(ApiError::BadRequest(format!(
            "Path does not exist: {}",
            path
        )));
    }

    // Run ingestion pipeline
    let mut embedder = state.embedder.lock().await;
    let result = ingest_repository(path, &state.store, &mut embedder).await?;

    Ok(Json(IngestResponse {
        code_chunks: result.code_chunks,
        readme_chunks: result.readme_chunks,
        message: format!(
            "Successfully ingested {} code chunks and {} readme chunks",
            result.code_chunks, result.readme_chunks
        ),
    }))
}

pub async fn list_projects(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ProjectsResponse>, ApiError> {
    let projects = state.store.list_projects().await?;
    let count = projects.len();

    Ok(Json(ProjectsResponse { projects, count }))
}

/// GET /health - Health check
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}
