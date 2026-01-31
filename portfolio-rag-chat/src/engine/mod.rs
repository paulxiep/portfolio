mod config;
pub mod context;
pub mod generator;
pub mod retriever;

pub use config::EngineConfig;
pub use generator::LlmClient;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] coderag_store::EmbedError),

    #[error("store error: {0}")]
    Store(#[from] coderag_store::StoreError),

    #[error("generation failed: {0}")]
    Generation(String),
}
