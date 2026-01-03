//! Trading strategy implementations.
//!
//! This module contains concrete `Agent` implementations for various
//! trading strategies. These are "Tier 1" agents that run every tick.
//!
//! # Available Strategies
//!
//! ## Market Infrastructure (Phase 5)
//! - [`NoiseTrader`] - Random orders near mid price to generate activity
//! - [`MarketMaker`] - Provides liquidity with bid/ask spread
//!
//! ## Technical Strategies (Phase 7)
//! - [`MomentumTrader`] - RSI-based momentum strategy (buy oversold, sell overbought)
//! - [`TrendFollower`] - SMA crossover trend following (golden/death cross)
//! - [`MacdCrossover`] - MACD signal line crossover strategy
//! - [`BollingerReversion`] - Mean reversion using Bollinger Bands
//!
//! ## Execution Algorithms (Phase 8)
//! - [`VwapExecutor`] - VWAP-targeting order execution algorithm

mod bollinger_reversion;
mod macd_crossover;
mod market_maker;
mod momentum;
mod noise_trader;
mod trend_follower;
mod vwap_executor;

pub use bollinger_reversion::{BollingerReversion, BollingerReversionConfig};
pub use macd_crossover::{MacdCrossover, MacdCrossoverConfig};
pub use market_maker::{MarketMaker, MarketMakerConfig};
pub use momentum::{MomentumConfig, MomentumTrader};
pub use noise_trader::{NoiseTrader, NoiseTraderConfig};
pub use trend_follower::{TrendFollower, TrendFollowerConfig};
pub use vwap_executor::{VwapExecutor, VwapExecutorConfig};
