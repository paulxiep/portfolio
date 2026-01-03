//! Trading strategy implementations.
//!
//! This module contains concrete `Agent` implementations for various
//! trading strategies. These are "Tier 1" agents that run every tick.
//!
//! # Available Strategies
//! - [`NoiseTrader`] - Random orders near mid price to generate activity
//! - [`MarketMaker`] - Provides liquidity with bid/ask spread

mod market_maker;
mod noise_trader;

pub use market_maker::{MarketMaker, MarketMakerConfig};
pub use noise_trader::{NoiseTrader, NoiseTraderConfig};
