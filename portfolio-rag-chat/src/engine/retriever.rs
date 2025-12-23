use crate::models::{CodeChunk, ReadmeChunk};
use crate::store::{Embedder, VectorStore};

use super::EngineError;
use super::config::RetrievalConfig;

/// Retrieved context from vector search
#[derive(Debug)]
pub struct RetrievalResult {
    pub code_chunks: Vec<CodeChunk>,
    pub readme_chunks: Vec<ReadmeChunk>,
}

/// 1. Embed the query text
/// 2. Search vector store for similar chunks
/// 3. Return structured results
pub async fn retrieve(
    query: &str,
    embedder: &mut Embedder,
    store: &VectorStore,
    config: &RetrievalConfig,
) -> Result<RetrievalResult, EngineError> {
    // Embed the query
    let query_embedding = embedder.embed_one(query)?;

    // Search both tables
    let (code_chunks, readme_chunks) = store
        .search_all(&query_embedding, config.code_limit, config.readme_limit)
        .await?;

    Ok(RetrievalResult {
        code_chunks,
        readme_chunks,
    })
}
