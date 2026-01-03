//! Agent trait and type definitions for the trading simulation.
//!
//! This module defines the core `Agent` trait that all trading agents must implement,
//! as well as the `MarketData` context they receive each tick.

use quant::IndicatorSnapshot;
use types::{AgentId, Candle, IndicatorType, Order};

// Re-export context from simulation crate once available
// For now, we define a minimal trait that the simulation crate will use

/// Result of an agent's decision each tick.
///
/// An agent may submit zero, one, or multiple orders each tick.
#[derive(Debug, Clone, Default)]
pub struct AgentAction {
    /// Orders to submit this tick.
    pub orders: Vec<Order>,
}

impl AgentAction {
    /// Create an empty action (no orders).
    pub fn none() -> Self {
        Self { orders: vec![] }
    }

    /// Create an action with a single order.
    pub fn single(order: Order) -> Self {
        Self {
            orders: vec![order],
        }
    }

    /// Create an action with multiple orders.
    pub fn multiple(orders: Vec<Order>) -> Self {
        Self { orders }
    }
}

/// The core trait that all trading agents must implement.
///
/// Agents are called once per tick with a snapshot of the market state.
/// They return an optional order to submit to the market.
///
/// # Lifetimes
/// The `MarketData` parameter borrows from the simulation state, so agents
/// cannot store references to it. They should extract any needed data during
/// the `on_tick` call.
///
/// # Example
/// ```ignore
/// struct SimpleAgent {
///     id: AgentId,
/// }
///
/// impl Agent for SimpleAgent {
///     fn id(&self) -> AgentId { self.id }
///
///     fn on_tick(&mut self, market: &MarketData) -> AgentAction {
///         // Decide whether to place an order based on market state
///         AgentAction::none()
///     }
/// }
/// ```
pub trait Agent: Send {
    /// Get the unique identifier for this agent.
    fn id(&self) -> AgentId;

    /// Called each simulation tick with the current market state.
    ///
    /// The agent should analyze the market data and return any orders
    /// it wishes to submit.
    ///
    /// # Arguments
    /// * `market` - Read-only snapshot of the current market state
    ///
    /// # Returns
    /// An `AgentAction` containing zero or more orders to submit
    fn on_tick(&mut self, market: &MarketData) -> AgentAction;

    /// Called when one of this agent's orders is filled (fully or partially).
    ///
    /// This allows agents to update their internal state when trades occur.
    /// Default implementation does nothing.
    ///
    /// # Arguments
    /// * `trade` - The trade that occurred involving this agent's order
    fn on_fill(&mut self, _trade: &types::Trade) {
        // Default: no-op
    }

    /// Get a human-readable name for this agent (for logging/debugging).
    fn name(&self) -> &str {
        "Agent"
    }

    /// Get the agent's current position (shares held).
    /// Positive = long, negative = short, zero = flat.
    /// Default returns 0 (unknown/not tracked).
    fn position(&self) -> i64 {
        0
    }

    /// Get the agent's current cash balance.
    /// Default returns zero (unknown/not tracked).
    fn cash(&self) -> types::Cash {
        types::Cash::ZERO
    }

    /// Get the agent's realized P&L.
    /// Default returns zero (unknown/not tracked).
    fn realized_pnl(&self) -> types::Cash {
        types::Cash::ZERO
    }
}

/// Market data snapshot passed to agents each tick.
///
/// Contains all the information an agent needs to make trading decisions.
/// This is a read-only view of the current market state.
#[derive(Debug, Clone)]
pub struct MarketData {
    /// Current simulation tick.
    pub tick: types::Tick,

    /// Current timestamp (wall clock).
    pub timestamp: types::Timestamp,

    /// Snapshot of the order book.
    pub book_snapshot: types::BookSnapshot,

    /// Recent trades (most recent first).
    pub recent_trades: Vec<types::Trade>,

    /// Last trade price (None if no trades yet).
    pub last_price: Option<types::Price>,

    /// Historical candles (oldest to newest, if available).
    pub candles: Vec<Candle>,

    /// Pre-computed indicator values (if available).
    pub indicators: Option<IndicatorSnapshot>,
}

impl MarketData {
    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<types::Price> {
        self.book_snapshot.best_bid()
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<types::Price> {
        self.book_snapshot.best_ask()
    }

    /// Get the mid price.
    pub fn mid_price(&self) -> Option<types::Price> {
        self.book_snapshot.mid_price()
    }

    /// Get the spread.
    pub fn spread(&self) -> Option<types::Price> {
        self.book_snapshot.spread()
    }

    /// Get a specific indicator value.
    pub fn get_indicator(&self, indicator_type: IndicatorType) -> Option<f64> {
        self.indicators
            .as_ref()
            .and_then(|s| s.get(&self.book_snapshot.symbol, indicator_type))
    }

    /// Get the most recent candle.
    pub fn last_candle(&self) -> Option<&Candle> {
        self.candles.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{BookSnapshot, Price};

    #[test]
    fn test_agent_action_none() {
        let action = AgentAction::none();
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_agent_action_single() {
        let order = Order::market(
            AgentId(1),
            "AAPL",
            types::OrderSide::Buy,
            types::Quantity(100),
        );
        let action = AgentAction::single(order.clone());
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].agent_id, AgentId(1));
    }

    #[test]
    fn test_market_data_accessors() {
        let market = MarketData {
            tick: 100,
            timestamp: 1000,
            book_snapshot: BookSnapshot {
                symbol: "AAPL".to_string(),
                bids: vec![types::BookLevel {
                    price: Price::from_float(99.0),
                    quantity: types::Quantity(100),
                    order_count: 1,
                }],
                asks: vec![types::BookLevel {
                    price: Price::from_float(101.0),
                    quantity: types::Quantity(100),
                    order_count: 1,
                }],
                timestamp: 1000,
                tick: 100,
            },
            recent_trades: vec![],
            last_price: Some(Price::from_float(100.0)),
            candles: vec![],
            indicators: None,
        };

        assert_eq!(market.best_bid(), Some(Price::from_float(99.0)));
        assert_eq!(market.best_ask(), Some(Price::from_float(101.0)));
        assert_eq!(market.mid_price(), Some(Price::from_float(100.0)));
        assert_eq!(market.spread(), Some(Price::from_float(2.0)));
    }
}
