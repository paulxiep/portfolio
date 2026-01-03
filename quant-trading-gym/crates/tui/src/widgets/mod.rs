//! TUI widgets for displaying simulation state.
//!
//! Each widget has a single responsibility (SoC):
//! - `PriceChart`: Renders price history as a line graph
//! - `BookDepth`: Renders bid/ask depth visualization
//! - `AgentTable`: Renders agent P&L summary
//! - `StatsPanel`: Renders simulation statistics
//! - `RiskPanel`: Renders per-agent risk metrics

mod agent_table;
mod book_depth;
mod price_chart;
mod risk_panel;
mod stats_panel;
mod update;

pub use agent_table::AgentTable;
pub use book_depth::BookDepth;
pub use price_chart::PriceChart;
pub use risk_panel::{RiskInfo, RiskPanel};
pub use stats_panel::StatsPanel;
pub use update::{AgentInfo, SimUpdate};
