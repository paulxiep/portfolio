//! TUI widgets for displaying simulation state.
//!
//! Each widget has a single responsibility (SoC):
//! - `PriceChart`: Renders price history as a line graph
//! - `BookDepth`: Renders bid/ask depth visualization
//! - `AgentTable`: Renders agent P&L summary
//! - `StatsPanel`: Renders simulation statistics

mod agent_table;
mod book_depth;
mod price_chart;
mod stats_panel;
mod update;

pub use agent_table::AgentTable;
pub use book_depth::BookDepth;
pub use price_chart::PriceChart;
pub use stats_panel::StatsPanel;
pub use update::{AgentInfo, SimUpdate};
