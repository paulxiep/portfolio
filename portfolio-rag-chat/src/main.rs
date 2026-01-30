mod api;
mod engine;
mod ingestion;
mod models;
mod store;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Warmup mode - just download model and exit
    if std::env::args().any(|a| a == "--warmup") {
        println!("Warming up embedding model...");
        let _ = crate::store::Embedder::new();
        println!("Warmup complete");
        return Ok(());
    }

    // Health check mode
    if std::env::args().any(|a| a == "--health") {
        // Simple health check - just exit 0
        return Ok(());
    }

    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "portfolio_rag_chat=info".into()),
        )
        .init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Configuration
    let db_path = std::env::var("DB_PATH").unwrap_or_else(|_| "./data/portfolio.lance".into());

    // Ingest mode - run ingestion and exit
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--ingest") {
        let repo_path = args.get(pos + 1).expect("--ingest requires a path argument");
        tracing::info!(repo_path, db_path, "Starting ingestion");

        let mut embedder = store::Embedder::new()?;
        let vector_store = store::VectorStore::new(&db_path, embedder.dimension()).await?;

        let stats = store::ingest_repository(repo_path, &vector_store, &mut embedder).await?;
        tracing::info!(
            code = stats.code_chunks,
            readme = stats.readme_chunks,
            crates = stats.crate_chunks,
            module_docs = stats.module_doc_chunks,
            "Ingestion complete"
        );
        return Ok(());
    }

    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".into());
    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);

    tracing::info!(db_path, model, "Initializing application");

    // Build application state
    let state = api::AppState::from_config(&db_path, &model).await?;

    // Build router
    let app = api::router(state);

    // Start server
    let addr = format!("{}:{}", host, port);
    tracing::info!("Starting server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
