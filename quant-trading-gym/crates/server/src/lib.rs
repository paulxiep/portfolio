//! Server crate: Axum-based async web services for Quant Trading Gym (V4.2).
//!
//! Provides the bridge between the synchronous simulation engine and async web clients.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────┐         ┌────────────────────────┐
//! │  Simulation Thread      │         │  Axum Server Thread    │
//! │  (sync loop)            │         │  (async/await)         │
//! │                         │         │                        │
//! │  sim.step()             │────────>│ receive tick event     │
//! │  hook.on_tick_end()     │ channel │ broadcast to WS        │
//! │                         │         │ handle REST requests   │
//! └─────────────────────────┘         └────────────────────────┘
//! ```
//!
//! # Design Principles
//!
//! - **Declarative**: Routes and handlers declared via Axum's type-safe routing
//! - **Modular**: Each feature (health, WebSocket, API) in separate module
//! - **SoC**: Simulation owns state; server observes and broadcasts
//!
//! # Modules
//!
//! - [`app`]: Axum application builder and router setup
//! - [`state`]: Shared server state (channels, metrics)
//! - [`error`]: Unified error handling with HTTP status codes
//! - [`routes`]: HTTP route handlers (health, ws, api)
//! - [`bridge`]: Channel types for simulation ↔ server communication
//! - [`hooks`]: SimulationHook implementations for broadcasting updates

pub mod app;
pub mod bridge;
pub mod error;
pub mod hooks;
pub mod routes;
pub mod state;

// Re-exports for convenience
pub use app::create_app;
pub use bridge::{SimCommand, SimUpdate, TickData};
pub use error::AppError;
pub use hooks::BroadcastHook;
pub use state::ServerState;
