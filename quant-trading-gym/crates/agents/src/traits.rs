//! Agent trait and type definitions for the trading simulation.
//!
//! This module defines the core `Agent` trait that all trading agents must implement,
//! as well as the `StrategyContext` they receive each tick.
//!
//! # V2.3 Changes
//!
//! The `on_tick` method now receives `StrategyContext<'_>` instead of `MarketData`.
//! `StrategyContext` provides multi-symbol access via the `MarketView` trait.
//!
//! `MarketData` is kept for backwards compatibility but should be considered deprecated.

use quant::IndicatorSnapshot;
use types::{AgentId, Candle, IndicatorType, Order};

use crate::StrategyContext;

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
/// Agents are called once per tick with a `StrategyContext` providing
/// access to market state, indicators, and historical data.
///
/// # V2.3 Changes
///
/// The `on_tick` method now receives `StrategyContext<'_>` instead of `MarketData`.
/// This enables multi-symbol access through the `MarketView` trait.
///
/// # Lifetimes
///
/// The `StrategyContext` parameter borrows from the simulation state, so agents
/// cannot store references to it. They should extract any needed data during
/// the `on_tick` call.
///
/// # Example
/// ```ignore
/// struct SimpleAgent {
///     id: AgentId,
///     symbol: String,
/// }
///
/// impl Agent for SimpleAgent {
///     fn id(&self) -> AgentId { self.id }
///
///     fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
///         // Access market data for our symbol
///         let mid = ctx.mid_price(&self.symbol);
///         let rsi = ctx.get_indicator(&self.symbol, IndicatorType::Rsi(14));
///         AgentAction::none()
///     }
/// }
/// ```
pub trait Agent: Send {
    /// Get the unique identifier for this agent.
    fn id(&self) -> AgentId;

    /// Called each simulation tick with the current market state.
    ///
    /// The agent should analyze the context and return any orders
    /// it wishes to submit.
    ///
    /// # Arguments
    /// * `ctx` - Read-only context with market state, indicators, and historical data
    ///
    /// # Returns
    /// An `AgentAction` containing zero or more orders to submit
    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction;

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

    /// Whether this agent is a market maker (exempt from short position limits).
    /// Market makers need flexibility to provide two-sided liquidity.
    fn is_market_maker(&self) -> bool {
        false
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

    /// Compute the agent's total equity at the given price.
    /// Equity = cash + (position * price)
    fn equity(&self, price: types::Price) -> types::Cash {
        let position = self.position();
        let position_value = if position >= 0 {
            price * types::Quantity(position as u64)
        } else {
            -(price * types::Quantity((-position) as u64))
        };
        self.cash() + position_value
    }
}

// =============================================================================
// Legacy MarketData (Deprecated)
// =============================================================================

/// Market data snapshot passed to agents each tick.
///
/// **DEPRECATED:** Use `StrategyContext` instead. This struct is kept for
/// backwards compatibility but will be removed in a future version.
///
/// Contains all the information an agent needs to make trading decisions.
/// This is a read-only view of the current market state.
#[derive(Debug, Clone)]
#[deprecated(since = "2.3.0", note = "Use StrategyContext instead")]
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

#[allow(deprecated)]
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
    use quant::IndicatorSnapshot;
    use sim_core::SingleSymbolMarket;
    use std::collections::HashMap;
    use types::{OrderId, Price, Quantity};

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
    fn test_strategy_context_integration() {
        // Create a minimal order book
        let mut book = sim_core::OrderBook::new("TEST");
        let mut bid = Order::limit(
            AgentId(1),
            "TEST",
            types::OrderSide::Buy,
            Price::from_float(99.0),
            Quantity(100),
        );
        bid.id = OrderId(1);
        let mut ask = Order::limit(
            AgentId(2),
            "TEST",
            types::OrderSide::Sell,
            Price::from_float(101.0),
            Quantity(100),
        );
        ask.id = OrderId(2);
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();

        // Create context
        let market = SingleSymbolMarket::new(&book);
        let candles = HashMap::new();
        let indicators = IndicatorSnapshot::new(100);
        let recent_trades = HashMap::new();
        let ctx = StrategyContext::new(100, 1000, &market, &candles, &indicators, &recent_trades);

        // Test access
        let symbol = "TEST".to_string();
        assert_eq!(ctx.mid_price(&symbol), Some(Price::from_float(100.0)));
        assert_eq!(ctx.best_bid(&symbol), Some(Price::from_float(99.0)));
        assert_eq!(ctx.best_ask(&symbol), Some(Price::from_float(101.0)));
    }
}
