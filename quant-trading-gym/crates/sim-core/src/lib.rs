//! Sim-core: Market mechanics for the Quant Trading Gym.
//!
//! This crate provides the core market simulation components:
//! - Order book management with price-time priority
//! - Matching engine for executing trades
//! - Slippage and market impact calculation (V2.2)
//! - Error handling for market operations

mod error;
mod matching;
mod order_book;
mod slippage;

pub use error::{Result, SimCoreError};
pub use matching::{MatchResult, MatchingEngine};
pub use order_book::{OrderBook, PriceLevel};
pub use slippage::{ImpactEstimate, SlippageCalculator};
