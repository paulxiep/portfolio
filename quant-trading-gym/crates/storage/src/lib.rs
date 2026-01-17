//! Storage layer for quant-trading-gym
//!
//! **Philosophy:** Declarative, Modular, SoC
//! - Declarative: Schema defined upfront, behavior driven by config
//! - Modular: Storage is swappable via SimulationHook trait
//! - SoC: This crate ONLY handles persistence, no simulation logic
//!
//! **V3.9 Scope:** Minimal storage for V4 bifurcation
//! - Trade history (append-only event log)
//! - Candle aggregation (time-series OLAP)
//! - Portfolio snapshots (analysis checkpoints)
//!
//! **V5.3 Scope:** ML training data recording
//! - Feature extraction (price, technical, fundamental, sentiment, agent state)
//! - Parquet output for Python ML consumption
//! - RecordingHook for capturing training data

mod candles;
mod hook;
mod schema;

// V5.3: ML training data recording
pub mod comprehensive_features;
pub mod features;
pub mod parquet_writer;
pub mod price_history;
pub mod recording_hook;

pub use hook::StorageHook;
pub use schema::StorageConfig;

// V5.3: Re-export recording types
pub use comprehensive_features::ComprehensiveFeatures;
pub use features::{FeatureContext, FeatureExtractor, MlModel};
pub use parquet_writer::{FeatureRecord, ParquetWriter, ParquetWriterError};
pub use price_history::PriceHistory;
pub use recording_hook::{RecordingConfig, RecordingHook};

#[cfg(test)]
mod tests;
