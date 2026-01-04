//! Common agent state tracking.
//!
//! This module provides shared state management for agents that track
//! position, cash, and trading metrics. Using a shared struct reduces
//! code duplication across agent implementations.

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

    /// Update state after a buy fill.
    pub fn on_buy(&mut self, quantity: u64, value: Cash) {
        self.position += quantity as i64;
        self.cash -= value;
        self.fills_received += 1;
    }

    /// Update state after a sell fill.
    pub fn on_sell(&mut self, quantity: u64, value: Cash) {
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
        Self::new(Cash::ZERO)
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
    }

    #[test]
    fn test_agent_state_on_buy() {
        let mut state = AgentState::new(Cash::from_float(10_000.0));
        state.on_buy(100, Cash::from_float(1_000.0));
        assert_eq!(state.position(), 100);
        assert_eq!(state.cash, Cash::from_float(9_000.0));
        assert_eq!(state.fills_received, 1);
    }

    #[test]
    fn test_agent_state_on_sell() {
        let mut state = AgentState::new(Cash::from_float(10_000.0));
        state.position = 100; // Start with position
        state.on_sell(50, Cash::from_float(500.0));
        assert_eq!(state.position, 50);
        assert_eq!(state.cash, Cash::from_float(10_500.0));
        assert_eq!(state.fills_received, 1);
    }

    #[test]
    fn test_agent_state_record_orders() {
        let mut state = AgentState::default();
        state.record_order();
        assert_eq!(state.orders_placed, 1);
        state.record_orders(3);
        assert_eq!(state.orders_placed, 4);
    }
}
