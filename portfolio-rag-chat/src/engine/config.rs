/// RAG pipeline configuration
#[derive(Clone, Debug, Default)]
pub struct EngineConfig {
    pub retrieval: RetrievalConfig,
}

/// How many chunks to retrieve
#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    pub code_limit: usize,
    pub readme_limit: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            code_limit: 5,
            readme_limit: 2,
        }
    }
}
