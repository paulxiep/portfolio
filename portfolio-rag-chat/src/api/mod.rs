mod dto;
mod error;
mod handlers;
mod state;

pub use state::AppState;

use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Build the application router
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/chat", post(handlers::chat))
        .route("/ingest", post(handlers::ingest))
        .route("/projects", get(handlers::list_projects)) // New
        .route("/health", get(handlers::health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
