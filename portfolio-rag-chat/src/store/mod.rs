pub mod embedder;
pub mod vector_store;

pub use embedder::Embedder;
pub use vector_store::VectorStore;

use crate::ingestion::run_ingestion;
use crate::models::{CodeChunk, CrateChunk, ModuleDocChunk, ReadmeChunk};
use embedder::{format_code_for_embedding, format_readme_for_embedding};
use thiserror::Error;

/// Batch size for embedding processing to reduce peak memory usage
const EMBEDDING_BATCH_SIZE: usize = 50;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("embedding error: {0}")]
    Embed(#[from] embedder::EmbedError),

    #[error("store error: {0}")]
    Store(#[from] vector_store::StoreError),
}

/// Result of running the full ingestion pipeline.
#[derive(Debug)]
pub struct IngestionStats {
    pub code_chunks: usize,
    pub readme_chunks: usize,
    pub crate_chunks: usize,
    pub module_doc_chunks: usize,
}

/// Full pipeline: Disk → Parse → Embed → Store
pub async fn ingest_repository(
    repo_path: &str,
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<IngestionStats, PipelineError> {
    // Destructure to allow early drop of each chunk type after processing
    let crate::ingestion::IngestionResult {
        code_chunks,
        readme_chunks,
        crate_chunks,
        module_doc_chunks,
    } = run_ingestion(repo_path);

    // Process code chunks (largest memory consumer), then drop
    let code_count = embed_and_store_code(&code_chunks, store, embedder).await?;
    drop(code_chunks);

    let readme_count = embed_and_store_readme(&readme_chunks, store, embedder).await?;
    drop(readme_chunks);

    let crate_count = embed_and_store_crates(&crate_chunks, store, embedder).await?;
    drop(crate_chunks);

    let module_doc_count = embed_and_store_module_docs(&module_doc_chunks, store, embedder).await?;
    drop(module_doc_chunks);

    Ok(IngestionStats {
        code_chunks: code_count,
        readme_chunks: readme_count,
        crate_chunks: crate_count,
        module_doc_chunks: module_doc_count,
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

    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        // Format chunks for embedding
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
        // texts & embeddings dropped here, freeing memory before next batch
    }
    Ok(total)
}

async fn embed_and_store_readme(
    chunks: &[ReadmeChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<usize, PipelineError> {
    if chunks.is_empty() {
        return Ok(0);
    }

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
/// Format a crate chunk for embedding
fn format_crate_for_embedding(chunk: &CrateChunk) -> String {
    let mut parts = vec![format!("Crate: {}", chunk.crate_name)];

    if let Some(desc) = &chunk.description {
        parts.push(desc.clone());
    }

    if !chunk.dependencies.is_empty() {
        parts.push(format!("Dependencies: {}", chunk.dependencies.join(", ")));
    }

    if let Some(project) = &chunk.project_name {
        parts.push(format!("Project: {}", project));
    }

    parts.join("\n")
}

async fn embed_and_store_crates(
    chunks: &[CrateChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<usize, PipelineError> {
    if chunks.is_empty() {
        return Ok(0);
    }

    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch.iter().map(format_crate_for_embedding).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;

        total += store.upsert_crate_chunks(batch, embeddings).await?;
    }
    Ok(total)
}

/// Format a module doc chunk for embedding
fn format_module_doc_for_embedding(chunk: &ModuleDocChunk) -> String {
    let mut parts = vec![format!("Module: {}", chunk.module_name)];
    parts.push(chunk.doc_content.clone());

    if let Some(project) = &chunk.project_name {
        parts.push(format!("Project: {}", project));
    }

    parts.join("\n")
}

async fn embed_and_store_module_docs(
    chunks: &[ModuleDocChunk],
    store: &VectorStore,
    embedder: &mut Embedder,
) -> Result<usize, PipelineError> {
    if chunks.is_empty() {
        return Ok(0);
    }

    let mut total = 0;
    for batch in chunks.chunks(EMBEDDING_BATCH_SIZE) {
        let texts: Vec<String> = batch.iter().map(format_module_doc_for_embedding).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = embedder.embed_batch(&text_refs)?;

        total += store.upsert_module_doc_chunks(batch, embeddings).await?;
    }
    Ok(total)
}
