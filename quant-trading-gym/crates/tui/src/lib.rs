//! TUI crate: Terminal User Interface for the Quant Trading Gym.
//!
//! This crate provides real-time visualization of the trading simulation:
//! - Live price chart (line graph)
//! - Order book depth visualization
//! - Agent P&L summary table
//! - Simulation statistics
//!
//! # Architecture
//!
//! The TUI runs in a separate thread from the simulation, communicating via channels:
//!
//! ```text
//! ┌────────────────┐     SimUpdate      ┌────────────────┐
//! │   Simulation   │ ────────────────►  │      TUI       │
//! │   (Thread A)   │    (channel)       │   (Thread B)   │
//! └────────────────┘                    └────────────────┘
//! ```
//!
//! This prevents slow terminal rendering from blocking the matching engine.
//!
//! # Usage
//!
//! ```ignore
//! use tui::{TuiApp, SimUpdate};
//! use crossbeam_channel::unbounded;
//!
//! // Create channel
//! let (tx, rx) = unbounded();
//!
//! // Start TUI in separate thread
//! let tui_handle = std::thread::spawn(move || {
//!     let mut app = TuiApp::new(rx.iter());
//!     app.run();
//! });
//!
//! // Send updates from simulation
//! tx.send(SimUpdate { ... });
//! ```

mod app;
mod widgets;

pub use app::{SimpleTui, TuiApp, check_quit};
pub use widgets::{AgentInfo, RiskInfo, SimUpdate};
