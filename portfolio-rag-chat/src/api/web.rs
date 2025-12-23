use askama::Template;
use axum::{
    Form,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
};
use std::sync::Arc;

use crate::api::dto::SourceInfo;
use crate::api::state::AppState;
use crate::engine::{context, generator, retriever};
use crate::store::ingest_repository;

/// Helper to render templates into axum responses
fn render_template<T: Template>(template: &T) -> Response {
    match template.render() {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            tracing::error!("Template render error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Template error: {}", e),
            )
                .into_response()
        }
    }
}

/// GET / - Main chat page
#[derive(Template)]
#[template(path = "chat.html")]
pub struct ChatPage;

pub async fn index() -> Response {
    render_template(&ChatPage)
}

/// POST /api/chat - Chat form submission (returns HTML fragment)
#[derive(serde::Deserialize)]
pub struct ChatForm {
    pub query: String,
}

#[derive(Template)]
#[template(path = "partials/message.html")]
pub struct MessageFragment {
    pub query: String,
    pub answer: String,
    pub sources: Vec<SourceInfo>,
}

pub async fn chat_html(State(state): State<Arc<AppState>>, Form(form): Form<ChatForm>) -> Response {
    let query = form.query.trim().to_string();

    if query.is_empty() {
        return Html("<div class=\"message assistant\"><div class=\"message-content\"><strong>Assistant:</strong><p>Please enter a question.</p></div></div>".to_string()).into_response();
    }

    // Retrieve relevant chunks
    let result = {
        let mut embedder = state.embedder.lock().await;
        match retriever::retrieve(&query, &mut embedder, &state.store, &state.config.retrieval)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Retrieval error: {}", e);
                return Html(format!(
                    "<div class=\"message user\"><div class=\"message-content\"><strong>You:</strong><p>{}</p></div></div>\
                     <div class=\"message assistant\"><div class=\"message-content\"><strong>Assistant:</strong><p>Sorry, I encountered an error retrieving context.</p></div></div>",
                    query
                )).into_response();
            }
        }
    };

    // Build context and generate response
    let context = context::build_context(&result);
    let prompt = context::build_prompt(&query, &context);

    let answer = match generator::generate(&prompt, &state.llm).await {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("Generation error: {}", e);
            format!("Sorry, I encountered an error: {}", e)
        }
    };

    let sources: Vec<SourceInfo> = result
        .code_chunks
        .into_iter()
        .map(|c| SourceInfo {
            file: c.file_path,
            function: c.identifier,
            project: c.project_name,
            line: c.start_line,
        })
        .collect();

    render_template(&MessageFragment {
        query,
        answer,
        sources,
    })
}

/// GET /api/projects-list - Returns HTML fragment of projects
#[derive(Template)]
#[template(path = "partials/projects_list.html")]
pub struct ProjectsListFragment {
    pub projects: Vec<String>,
    pub count: usize,
}

pub async fn projects_list_html(State(state): State<Arc<AppState>>) -> Response {
    let projects = state.store.list_projects().await.unwrap_or_default();
    let count = projects.len();

    render_template(&ProjectsListFragment { projects, count })
}

/// POST /api/ingest - Ingest form submission (returns HTML fragment)
#[derive(serde::Deserialize)]
pub struct IngestForm {
    pub repo_path: String,
}

#[derive(Template)]
#[template(path = "partials/ingest_result.html")]
pub struct IngestResultFragment {
    pub success: bool,
    pub message: String,
    pub code_chunks: usize,
    pub readme_chunks: usize,
}

pub async fn ingest_html(
    State(state): State<Arc<AppState>>,
    Form(form): Form<IngestForm>,
) -> Response {
    let path = form.repo_path.trim();

    if path.is_empty() {
        return render_template(&IngestResultFragment {
            success: false,
            message: "Please enter a repository path".to_string(),
            code_chunks: 0,
            readme_chunks: 0,
        });
    }

    if !std::path::Path::new(path).exists() {
        return render_template(&IngestResultFragment {
            success: false,
            message: format!("Path does not exist: {}", path),
            code_chunks: 0,
            readme_chunks: 0,
        });
    }

    // Run ingestion
    let result = {
        let mut embedder = state.embedder.lock().await;
        match ingest_repository(path, &state.store, &mut embedder).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Ingestion error: {}", e);
                return render_template(&IngestResultFragment {
                    success: false,
                    message: format!("Ingestion failed: {}", e),
                    code_chunks: 0,
                    readme_chunks: 0,
                });
            }
        }
    };

    render_template(&IngestResultFragment {
        success: true,
        message: "Repository indexed successfully!".to_string(),
        code_chunks: result.code_chunks,
        readme_chunks: result.readme_chunks,
    })
}
