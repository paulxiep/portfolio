#[allow(dead_code)]
pub mod embedder;
#[allow(dead_code)]
pub mod vector_store;

pub use embedder::Embedder;
pub use vector_store::VectorStore;

use crate::ingestion::run_ingestion;
use crate::models::{CodeChunk, ReadmeChunk};
use embedder::{format_code_for_embedding, format_readme_for_embedding};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("embedding error: {0}")]
    Embed(#[from] embedder::EmbedError),

    #[error("store error: {0}")]
    Store(#[from] vector_store::StoreError),
}

/// Result of running the full ingestion pipeline.
#[derive(Debug)]
pub struct IngestionResult {
    pub code_chunks: usize,
    pub readme_chunks: usize,
}

/// Full pipeline: Disk → Parse → Embed → Store
pub async fn ingest_repository(
    repo_path: &str,
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<IngestionResult, PipelineError> {
    let (code_chunks, readme_chunks) = run_ingestion(repo_path);

    let code_count = embed_and_store_code(&code_chunks, store, embedder).await?;
    let readme_count = embed_and_store_readme(&readme_chunks, store, embedder).await?;

    Ok(IngestionResult {
        code_chunks: code_count,
        readme_chunks: readme_count,
    })
}

async fn embed_and_store_code(
    chunks: &[CodeChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<usize, PipelineError> {
    if chunks.is_empty() {
        return Ok(0);
    }

    // Format chunks for embedding
    let texts: Vec<String> = chunks
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

    let count = store.upsert_code_chunks(chunks, embeddings).await?;
    Ok(count)
}

async fn embed_and_store_readme(
    chunks: &[ReadmeChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<usize, PipelineError> {
    if chunks.is_empty() {
        return Ok(0);
    }

    let texts: Vec<String> = chunks
        .iter()
        .map(|c| format_readme_for_embedding(&c.project_name, &c.content))
        .collect();

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = embedder.embed_batch(&text_refs)?;

    let count = store.upsert_readme_chunks(chunks, embeddings).await?;
    Ok(count)
}
