//! Common agent state tracking.
//!
//! This module provides shared state management for agents that track
//! position, cash, and trading metrics. Using a shared struct reduces
//! code duplication across agent implementations.
//!
//! ## P&L Tracking
//!
//! The state tracks realized P&L using weighted average cost basis:
//! - On buy: `new_avg_cost = (old_qty * old_avg + buy_qty * buy_price) / (old_qty + buy_qty)`
//! - On sell: `realized_pnl += (sell_price - avg_cost) * sell_qty`

use types::Cash;

/// Common state shared across agent implementations.
///
/// Agents that track position and cash should embed this struct
/// rather than duplicating the fields. This ensures consistent
/// behavior and makes it easier to add new metrics.
#[derive(Debug, Clone)]
pub struct AgentState {
    /// Current position in shares (positive = long, negative = short).
    position: i64,
    /// Current cash balance.
    cash: Cash,
    /// Weighted average cost basis per share (for P&L calculation).
    avg_cost: f64,
    /// Accumulated realized P&L from closed positions.
    realized_pnl: Cash,
    /// Total number of orders placed.
    orders_placed: u64,
    /// Total number of fills received.
    fills_received: u64,
}

impl AgentState {
    /// Create a new agent state with initial cash.
    pub fn new(initial_cash: Cash) -> Self {
        Self {
            position: 0,
            cash: initial_cash,
            avg_cost: 0.0,
            realized_pnl: Cash::ZERO,
            orders_placed: 0,
            fills_received: 0,
        }
    }

    /// Get current position.
    pub fn position(&self) -> i64 {
        self.position
    }

    /// Set position directly (for initial allocation from float).
    pub fn set_position(&mut self, position: i64) {
        self.position = position;
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.cash
    }

    /// Get total orders placed.
    pub fn orders_placed(&self) -> u64 {
        self.orders_placed
    }

    /// Get total fills received.
    pub fn fills_received(&self) -> u64 {
        self.fills_received
    }

    /// Get realized P&L.
    pub fn realized_pnl(&self) -> Cash {
        self.realized_pnl
    }

    /// Get average cost basis per share.
    pub fn avg_cost(&self) -> f64 {
        self.avg_cost
    }

    /// Update state after a buy fill.
    /// Uses weighted average cost basis calculation.
    pub fn on_buy(&mut self, quantity: u64, value: Cash) {
        let buy_price = value.to_float() / quantity as f64;
        let old_qty = self.position.max(0) as f64;
        let new_qty = old_qty + quantity as f64;

        // Update weighted average cost (only for long positions)
        if new_qty > 0.0 {
            self.avg_cost = (old_qty * self.avg_cost + quantity as f64 * buy_price) / new_qty;
        }

        self.position += quantity as i64;
        self.cash -= value;
        self.fills_received += 1;
    }

    /// Update state after a sell fill.
    /// Computes realized P&L as (sell_price - avg_cost) * quantity.
    pub fn on_sell(&mut self, quantity: u64, value: Cash) {
        let sell_price = value.to_float() / quantity as f64;

        // Only compute realized P&L if we had a long position with cost basis
        if self.position > 0 && self.avg_cost > 0.0 {
            let qty_to_close = (quantity as i64).min(self.position) as f64;
            let pnl = (sell_price - self.avg_cost) * qty_to_close;
            self.realized_pnl += Cash::from_float(pnl);
        }

        self.position -= quantity as i64;
        self.cash += value;
        self.fills_received += 1;
    }

    /// Increment orders placed counter.
    pub fn record_order(&mut self) {
        self.orders_placed += 1;
    }

    /// Increment orders placed counter by count.
    pub fn record_orders(&mut self, count: u64) {
        self.orders_placed += count;
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            position: 0,
            cash: Cash::ZERO,
            avg_cost: 0.0,
            realized_pnl: Cash::ZERO,
            orders_placed: 0,
            fills_received: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state_new() {
        let state = AgentState::new(Cash::from_float(10_000.0));
        assert_eq!(state.position(), 0);
        assert_eq!(state.cash(), Cash::from_float(10_000.0));
        assert_eq!(state.orders_placed(), 0);
        assert_eq!(state.fills_received(), 0);
        assert_eq!(state.realized_pnl(), Cash::ZERO);
        assert_eq!(state.avg_cost(), 0.0);
    }

    #[test]
    fn test_agent_state_on_buy() {
        let mut state = AgentState::new(Cash::from_float(10_000.0));
        state.on_buy(100, Cash::from_float(1_000.0));
        assert_eq!(state.position(), 100);
        assert_eq!(state.cash(), Cash::from_float(9_000.0));
        assert_eq!(state.fills_received(), 1);
        // avg_cost = 1000 / 100 = $10 per share
        assert!((state.avg_cost() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_agent_state_on_sell() {
        let mut state = AgentState::new(Cash::from_float(10_000.0));
        // Buy first to establish cost basis
        state.on_buy(100, Cash::from_float(1_000.0)); // $10/share
        state.on_sell(50, Cash::from_float(600.0)); // Sell at $12/share
        assert_eq!(state.position(), 50);
        assert_eq!(state.cash(), Cash::from_float(9_600.0)); // 9000 + 600
        assert_eq!(state.fills_received(), 2);
        // Realized P&L = (12 - 10) * 50 = $100
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_agent_state_record_orders() {
        let mut state = AgentState::default();
        state.record_order();
        assert_eq!(state.orders_placed(), 1);
        state.record_orders(3);
        assert_eq!(state.orders_placed(), 4);
    }

    #[test]
    fn test_weighted_average_cost() {
        let mut state = AgentState::new(Cash::from_float(100_000.0));
        // Buy 100 shares at $10
        state.on_buy(100, Cash::from_float(1_000.0));
        assert!((state.avg_cost() - 10.0).abs() < 0.001);

        // Buy 100 more shares at $20
        state.on_buy(100, Cash::from_float(2_000.0));
        // New avg = (100 * 10 + 100 * 20) / 200 = 3000 / 200 = $15
        assert!((state.avg_cost() - 15.0).abs() < 0.001);
        assert_eq!(state.position(), 200);
    }

    #[test]
    fn test_realized_pnl_profit() {
        let mut state = AgentState::new(Cash::from_float(100_000.0));
        // Buy 100 at $50
        state.on_buy(100, Cash::from_float(5_000.0));
        // Sell 100 at $60 (profit of $10/share)
        state.on_sell(100, Cash::from_float(6_000.0));
        // Realized P&L = (60 - 50) * 100 = $1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 1000.0).abs() < 0.01);
        assert_eq!(state.position(), 0);
    }

    #[test]
    fn test_realized_pnl_loss() {
        let mut state = AgentState::new(Cash::from_float(100_000.0));
        // Buy 100 at $50
        state.on_buy(100, Cash::from_float(5_000.0));
        // Sell 100 at $40 (loss of $10/share)
        state.on_sell(100, Cash::from_float(4_000.0));
        // Realized P&L = (40 - 50) * 100 = -$1000
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - (-1000.0)).abs() < 0.01);
    }

    #[test]
    fn test_partial_sell_pnl() {
        let mut state = AgentState::new(Cash::from_float(100_000.0));
        // Buy 100 at $10
        state.on_buy(100, Cash::from_float(1_000.0));
        // Sell 30 at $15 (profit of $5/share on 30 shares)
        state.on_sell(30, Cash::from_float(450.0));
        // Realized P&L = (15 - 10) * 30 = $150
        let pnl = state.realized_pnl().to_float();
        assert!((pnl - 150.0).abs() < 0.01);
        assert_eq!(state.position(), 70);
        // avg_cost should remain $10
        assert!((state.avg_cost() - 10.0).abs() < 0.001);
    }
}
