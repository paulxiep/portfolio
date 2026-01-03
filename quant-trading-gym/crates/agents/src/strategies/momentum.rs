//! RSI Momentum Trader - buys oversold, sells overbought.
//!
//! A momentum strategy that uses the Relative Strength Index (RSI) to identify
//! overbought and oversold conditions and trade accordingly.
//!
//! # Strategy Logic
//! - **Buy signal**: RSI < oversold_threshold (default 30)
//! - **Sell signal**: RSI > overbought_threshold (default 70)
//! - Flat (exit position) when RSI returns to neutral zone
//!
//! # Configuration
//! The strategy is fully declarative via [`MomentumConfig`]. Indicators are
//! requested through the `MarketData` snapshot, not computed internally.

use crate::state::AgentState;
use crate::{Agent, AgentAction, MarketData};
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a Momentum (RSI) trader.
#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// RSI period for calculations.
    pub rsi_period: usize,
    /// RSI threshold for oversold (buy signal).
    pub oversold_threshold: f64,
    /// RSI threshold for overbought (sell signal).
    pub overbought_threshold: f64,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            rsi_period: 14,
            oversold_threshold: 30.0,
            overbought_threshold: 70.0,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
        }
    }
}

/// RSI Momentum trader that buys oversold and sells overbought conditions.
///
/// This strategy demonstrates the indicator pipeline by requesting RSI
/// values from the market data and making trading decisions based on
/// standard momentum signals.
pub struct MomentumTrader {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MomentumConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
}

impl MomentumTrader {
    /// Create a new MomentumTrader with the given configuration.
    pub fn new(id: AgentId, config: MomentumConfig) -> Self {
        let initial_cash = config.initial_cash;
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
        }
    }

    /// Create a MomentumTrader with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, MomentumConfig::default())
    }

    /// Get the IndicatorType for RSI that this strategy uses.
    pub fn required_indicator(&self) -> IndicatorType {
        IndicatorType::Rsi(self.config.rsi_period)
    }

    /// Determine the reference price for orders.
    fn get_reference_price(&self, market: &MarketData) -> Price {
        market
            .mid_price()
            .or(market.last_price)
            .unwrap_or(self.config.initial_price)
    }

    /// Check if we can take more long positions.
    fn can_buy(&self) -> bool {
        self.state.position() < self.config.max_position
    }

    /// Check if we can take more short positions.
    fn can_sell(&self) -> bool {
        self.state.position() > -self.config.max_position
    }

    /// Generate a buy order at the current reference price.
    fn generate_buy_order(&self, market: &MarketData) -> Order {
        let price = self.get_reference_price(market);
        // Slightly below mid to increase fill probability
        let order_price = Price::from_float(price.to_float() * 0.999);
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Buy,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Generate a sell order at the current reference price.
    fn generate_sell_order(&self, market: &MarketData) -> Order {
        let price = self.get_reference_price(market);
        // Slightly above mid to increase fill probability
        let order_price = Price::from_float(price.to_float() * 1.001);
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }
}

impl Agent for MomentumTrader {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Get RSI from pre-computed indicators
        let rsi = match market.get_indicator(IndicatorType::Rsi(self.config.rsi_period)) {
            Some(rsi) => rsi,
            None => return AgentAction::none(), // Not enough data yet
        };

        // RSI < oversold_threshold: buy signal
        if rsi < self.config.oversold_threshold && self.can_buy() {
            let order = self.generate_buy_order(market);
            self.state.record_order();
            return AgentAction::single(order);
        }

        // RSI > overbought_threshold: sell signal
        if rsi > self.config.overbought_threshold && self.can_sell() {
            let order = self.generate_sell_order(market);
            self.state.record_order();
            return AgentAction::single(order);
        }

        AgentAction::none()
    }

    fn on_fill(&mut self, trade: &Trade) {
        if trade.buyer_id == self.id {
            self.state.on_buy(trade.quantity.raw(), trade.value());
        } else if trade.seller_id == self.id {
            self.state.on_sell(trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        "Momentum"
    }

    fn position(&self) -> i64 {
        self.state.position()
    }

    fn cash(&self) -> Cash {
        self.state.cash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use types::BookSnapshot;

    fn make_market_data(rsi: Option<f64>) -> MarketData {
        use quant::IndicatorSnapshot;

        let indicators = rsi.map(|r| {
            let mut snap = IndicatorSnapshot::new(100);
            let mut indicators = HashMap::new();
            indicators.insert(IndicatorType::Rsi(14), r);
            snap.insert("ACME".to_string(), indicators);
            snap
        });

        MarketData {
            tick: 100,
            timestamp: 1000,
            book_snapshot: BookSnapshot {
                symbol: "ACME".to_string(),
                bids: vec![types::BookLevel {
                    price: Price::from_float(99.0),
                    quantity: Quantity(100),
                    order_count: 1,
                }],
                asks: vec![types::BookLevel {
                    price: Price::from_float(101.0),
                    quantity: Quantity(100),
                    order_count: 1,
                }],
                timestamp: 1000,
                tick: 100,
            },
            recent_trades: vec![],
            last_price: Some(Price::from_float(100.0)),
            candles: vec![],
            indicators,
        }
    }

    #[test]
    fn test_momentum_buys_on_oversold() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let market = make_market_data(Some(25.0)); // Oversold

        let action = trader.on_tick(&market);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_momentum_sells_on_overbought() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let market = make_market_data(Some(75.0)); // Overbought

        let action = trader.on_tick(&market);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }

    #[test]
    fn test_momentum_no_action_neutral() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let market = make_market_data(Some(50.0)); // Neutral

        let action = trader.on_tick(&market);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_momentum_no_action_without_indicator() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let market = make_market_data(None);

        let action = trader.on_tick(&market);
        assert!(action.orders.is_empty());
    }
}
