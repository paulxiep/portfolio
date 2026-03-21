use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub database: DatabaseConfig,
    pub blob_storage: BlobStorageConfig,
    pub queue: QueueConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlobStorageConfig {
    #[serde(rename = "type")]
    pub storage_type: String,
    pub base_path: Option<String>,
    pub bucket: Option<String>,
    pub region: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueueConfig {
    #[serde(rename = "type")]
    pub queue_type: String,
    pub url: Option<String>,
    pub consumer_group: Option<String>,
    pub queue_a_url: Option<String>,
    pub queue_b_url: Option<String>,
    pub region: Option<String>,
}

/// Load config from a YAML file.
///
/// Resolution order:
/// 1. Explicit path argument
/// 2. `INVOICE_PARSE_CONFIG` env var
/// 3. `config/local.yaml`
pub fn load_config(path: Option<&Path>) -> Result<AppConfig, Box<dyn std::error::Error>> {
    let path = match path {
        Some(p) => p.to_path_buf(),
        None => {
            let env_path = std::env::var("INVOICE_PARSE_CONFIG").ok();
            env_path
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| "config/local.yaml".into())
        }
    };
    let contents = std::fs::read_to_string(&path)?;
    let config: AppConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}
