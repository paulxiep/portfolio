//! Agents crate: Trading agents for the Quant Trading Gym.
//!
//! This crate provides:
//! - The `Agent` trait that all trading agents must implement
//! - `MarketData` context passed to agents each tick
//! - `AgentAction` for returning agent decisions
//! - `AgentState` for common state tracking (position, cash, metrics)
//! - Concrete strategy implementations (`strategies` module)
//!
//! # Architecture
//! Agents receive a `MarketData` snapshot each tick and return an `AgentAction`
//! containing any orders they wish to submit. The simulation handles order
//! routing, matching, and notifying agents of fills.
//!
//! # Available Strategies
//!
//! ## Market Infrastructure (Phase 5)
//! - [`strategies::NoiseTrader`] - Random orders to generate market activity
//! - [`strategies::MarketMaker`] - Two-sided liquidity provider
//!
//! ## Technical Strategies (Phase 7)
//! - [`strategies::MomentumTrader`] - RSI-based momentum (buy oversold, sell overbought)
//! - [`strategies::TrendFollower`] - SMA crossover trend following
//! - [`strategies::MacdCrossover`] - MACD signal line crossover
//! - [`strategies::BollingerReversion`] - Bollinger Bands mean reversion
//!
//! ## Execution Algorithms (Phase 8)
//! - [`strategies::VwapExecutor`] - VWAP-targeting order execution
//!
//! # Example
//! ```ignore
//! use agents::{Agent, AgentAction, MarketData};
//! use types::AgentId;
//!
//! struct MyAgent {
//!     id: AgentId,
//! }
//!
//! impl Agent for MyAgent {
//!     fn id(&self) -> AgentId { self.id }
//!
//!     fn on_tick(&mut self, market: &MarketData) -> AgentAction {
//!         AgentAction::none()
//!     }
//! }
//! ```

mod state;
pub mod strategies;
mod traits;

pub use state::AgentState;
pub use strategies::{
    BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig, MarketMaker,
    MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    TrendFollower, TrendFollowerConfig, VwapExecutor, VwapExecutorConfig,
};
pub use traits::{Agent, AgentAction, MarketData};
