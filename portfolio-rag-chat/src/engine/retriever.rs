use crate::models::{CodeChunk, CrateChunk, ModuleDocChunk, ReadmeChunk};
use crate::store::VectorStore;

use super::EngineError;
use super::config::RetrievalConfig;

/// Retrieved context from vector search
#[derive(Debug)]
pub struct RetrievalResult {
    pub code_chunks: Vec<CodeChunk>,
    pub readme_chunks: Vec<ReadmeChunk>,
    pub crate_chunks: Vec<CrateChunk>,
    pub module_doc_chunks: Vec<ModuleDocChunk>,
}

/// Search vector store for similar chunks using a pre-computed query embedding.
pub async fn retrieve(
    query_embedding: &[f32],
    store: &VectorStore,
    config: &RetrievalConfig,
) -> Result<RetrievalResult, EngineError> {
    let (code_chunks, readme_chunks, crate_chunks, module_doc_chunks) = store
        .search_all(
            query_embedding,
            config.code_limit,
            config.readme_limit,
            config.crate_limit,
            config.module_doc_limit,
        )
        .await?;

    Ok(RetrievalResult {
        code_chunks,
        readme_chunks,
        crate_chunks,
        module_doc_chunks,
    })
}
