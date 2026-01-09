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

mod candles;
mod hook;
mod schema;

pub use hook::StorageHook;
pub use schema::StorageConfig;

#[cfg(test)]
mod tests;
