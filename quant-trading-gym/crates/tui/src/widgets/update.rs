//! SimUpdate message type for TUI updates.
//!
//! This module defines the data structure sent from the simulation thread
//! to the TUI thread via channels.

use types::{BookLevel, Cash, Price, Tick, Trade};

use crate::widgets::RiskInfo;

/// Agent state summary for TUI display.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent display name.
    pub name: String,
    /// Current position (positive = long, negative = short).
    pub position: i64,
    /// Total realized P&L.
    pub realized_pnl: Cash,
    /// Current cash balance.
    pub cash: Cash,
    /// Whether this is a market maker (for sorting to bottom).
    pub is_market_maker: bool,
    /// Current equity (cash + position value) for sorting.
    pub equity: f64,
}

/// Update message sent from simulation to TUI.
///
/// Contains all data needed to render a single frame.
/// Designed for efficient channel transmission without
/// requiring the TUI to understand simulation internals.
#[derive(Debug, Clone, Default)]
pub struct SimUpdate {
    /// Current simulation tick.
    pub tick: Tick,
    /// Latest trades this tick (may be empty).
    pub trades: Vec<Trade>,
    /// Last trade price (None if no trades yet).
    pub last_price: Option<Price>,
    /// Price history for charting (most recent last).
    pub price_history: Vec<f64>,
    /// Bid levels (highest first).
    pub bids: Vec<BookLevel>,
    /// Ask levels (lowest first).
    pub asks: Vec<BookLevel>,
    /// Agent summaries for P&L table.
    pub agents: Vec<AgentInfo>,
    /// Total trades executed.
    pub total_trades: u64,
    /// Total orders submitted.
    pub total_orders: u64,
    /// Simulation is complete.
    pub finished: bool,
    /// Per-agent risk metrics.
    pub risk_metrics: Vec<RiskInfo>,
}

impl SimUpdate {
    /// Create a "simulation finished" message.
    pub fn finished(tick: Tick, total_trades: u64) -> Self {
        Self {
            tick,
            total_trades,
            finished: true,
            ..Default::default()
        }
    }
}
