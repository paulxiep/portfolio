//! code-raptor CLI - Code knowledge graph construction tool

use clap::{Parser, Subcommand};
use code_raptor::{
    Embedder, VectorStore, format_code_for_embedding, format_crate_for_embedding,
    format_module_doc_for_embedding, format_readme_for_embedding, run_ingestion,
};
use coderag_types::{CodeChunk, CrateChunk, ModuleDocChunk, ReadmeChunk};
use tracing::info;

/// Batch size for embedding processing to reduce peak memory usage
const EMBEDDING_BATCH_SIZE: usize = 25;

#[derive(Parser)]
#[command(name = "code-raptor")]
#[command(about = "Build code knowledge graphs for RAG applications")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a repository and build the knowledge graph
    Ingest {
        /// Path to the repository to ingest
        #[arg(value_name = "PATH")]
        repo_path: String,

        /// Path to the LanceDB database
        #[arg(short, long, default_value = "data/portfolio.lance")]
        db_path: String,
    },
    /// Show status of indexed repositories
    Status {
        /// Path to the LanceDB database
        #[arg(short, long, default_value = "data/portfolio.lance")]
        db_path: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Ingest { repo_path, db_path } => {
            info!("Ingesting repository: {}", repo_path);
            info!("Database path: {}", db_path);

            // Step 1: Parse code into chunks
            let result = run_ingestion(&repo_path);

            info!(
                "Parsed: {} code chunks, {} readmes, {} crates, {} module docs",
                result.code_chunks.len(),
                result.readme_chunks.len(),
                result.crate_chunks.len(),
                result.module_doc_chunks.len()
            );

            // Step 2: Initialize embedder
            info!("Initializing embedding model...");
            let mut embedder = Embedder::new()?;
            let dimension = embedder.dimension();

            // Step 3: Initialize vector store
            info!("Connecting to LanceDB at: {}", db_path);
            let store = VectorStore::new(&db_path, dimension).await?;

            // Step 4: Embed and store code chunks (batched for memory efficiency)
            let code_count =
                embed_and_store_code(&result.code_chunks, &store, &mut embedder).await?;
            if code_count > 0 {
                info!("Stored {} code chunks", code_count);
            }

            // Step 5: Embed and store readme chunks
            let readme_count =
                embed_and_store_readme(&result.readme_chunks, &store, &mut embedder).await?;
            if readme_count > 0 {
                info!("Stored {} readme chunks", readme_count);
            }

            // Step 6: Embed and store crate chunks
            let crate_count =
                embed_and_store_crates(&result.crate_chunks, &store, &mut embedder).await?;
            if crate_count > 0 {
                info!("Stored {} crate chunks", crate_count);
            }

            // Step 7: Embed and store module doc chunks
            let module_doc_count =
                embed_and_store_module_docs(&result.module_doc_chunks, &store, &mut embedder)
                    .await?;
            if module_doc_count > 0 {
                info!("Stored {} module doc chunks", module_doc_count);
            }

            info!("Ingestion complete!");
        }
        Commands::Status { db_path } => {
            info!("Checking status of: {}", db_path);
            let store = VectorStore::new(&db_path, 384).await?;
            let projects = store.list_projects().await?;
            info!("Projects indexed: {:?}", projects);
        }
    }

    Ok(())
}

async fn embed_and_store_code(
    chunks: &[CodeChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> anyhow::Result<usize> {
    if chunks.is_empty() {
        return Ok(0);
    }

    info!("Embedding {} code chunks...", chunks.len());
    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| {
                format_code_for_embedding(
                    &c.identifier,
                    &c.language,
                    c.docstring.as_deref(),
                    &c.code_content,
                )
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;
        total += store.upsert_code_chunks(batch, embeddings).await?;
    }
    Ok(total)
}

async fn embed_and_store_readme(
    chunks: &[ReadmeChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> anyhow::Result<usize> {
    if chunks.is_empty() {
        return Ok(0);
    }

    info!("Embedding {} readme chunks...", chunks.len());
    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| format_readme_for_embedding(&c.project_name, &c.content))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;
        total += store.upsert_readme_chunks(batch, embeddings).await?;
    }
    Ok(total)
}

async fn embed_and_store_crates(
    chunks: &[CrateChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> anyhow::Result<usize> {
    if chunks.is_empty() {
        return Ok(0);
    }

    info!("Embedding {} crate chunks...", chunks.len());
    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| {
                format_crate_for_embedding(&c.crate_name, c.description.as_deref(), &c.dependencies)
            })
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;
        total += store.upsert_crate_chunks(batch, embeddings).await?;
    }
    Ok(total)
}

async fn embed_and_store_module_docs(
    chunks: &[ModuleDocChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> anyhow::Result<usize> {
    if chunks.is_empty() {
        return Ok(0);
    }

    info!("Embedding {} module doc chunks...", chunks.len());
    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch
            .iter()
            .map(|c| format_module_doc_for_embedding(&c.module_name, &c.doc_content))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;
        total += store.upsert_module_doc_chunks(batch, embeddings).await?;
    }
    Ok(total)
}
