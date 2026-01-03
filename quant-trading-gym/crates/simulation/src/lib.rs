//! Simulation crate: The event loop for the Quant Trading Gym.
//!
//! This crate provides the simulation runner that coordinates:
//! - Tick-based event loop
//! - Agent execution
//! - Order processing through the matching engine
//! - Market data distribution
//!
//! # Architecture
//!
//! The simulation runs in discrete ticks:
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              Simulation.step()          │
//! │                                         │
//! │  1. Build MarketData snapshot           │
//! │  2. Call agent.on_tick() for each agent │
//! │  3. Collect AgentActions (orders)       │
//! │  4. Process orders through MatchingEngine│
//! │  5. Notify agents of fills via on_fill()│
//! │  6. Advance tick counter                │
//! │                                         │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use simulation::{Simulation, SimulationConfig};
//! use agents::{Agent, AgentAction, MarketData};
//! use types::AgentId;
//!
//! // Create a simple agent
//! struct MyAgent { id: AgentId }
//! impl Agent for MyAgent {
//!     fn id(&self) -> AgentId { self.id }
//!     fn on_tick(&mut self, _: &MarketData) -> AgentAction { AgentAction::none() }
//! }
//!
//! // Set up and run simulation
//! let mut sim = Simulation::new(SimulationConfig::new("AAPL"));
//! sim.add_agent(Box::new(MyAgent { id: AgentId(1) }));
//! let trades = sim.run(1000);
//! println!("Executed {} trades over 1000 ticks", trades.len());
//! ```

mod config;
mod runner;

pub use config::SimulationConfig;
pub use runner::{Simulation, SimulationStats};
