//! Agents crate: Trading agents for the Quant Trading Gym.
//!
//! This crate provides:
//! - The `Agent` trait that all trading agents must implement
//! - `MarketData` context passed to agents each tick
//! - `AgentAction` for returning agent decisions
//!
//! # Architecture
//! Agents receive a `MarketData` snapshot each tick and return an `AgentAction`
//! containing any orders they wish to submit. The simulation handles order
//! routing, matching, and notifying agents of fills.
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

mod traits;

pub use traits::{Agent, AgentAction, MarketData};
