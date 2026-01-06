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
//! requested through the `StrategyContext` snapshot, not computed internally.

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext};
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
            config: config.clone(),
            state: AgentState::new(initial_cash, &[&config.symbol]),
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
    fn get_reference_price(&self, ctx: &StrategyContext<'_>) -> Price {
        ctx.mid_price(&self.config.symbol)
            .or(ctx.last_price(&self.config.symbol))
            .unwrap_or(self.config.initial_price)
    }

    /// Check if we can take more long positions.
    fn can_buy(&self) -> bool {
        self.state.position_for(&self.config.symbol) < self.config.max_position
    }

    /// Check if we can take more short positions.
    fn can_sell(&self) -> bool {
        self.state.position_for(&self.config.symbol) > -self.config.max_position
    }

    /// Generate a buy order at the current reference price.
    fn generate_buy_order(&self, ctx: &StrategyContext<'_>) -> Order {
        let price = self.get_reference_price(ctx);
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
    fn generate_sell_order(&self, ctx: &StrategyContext<'_>) -> Order {
        let price = self.get_reference_price(ctx);
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

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        // Get RSI from pre-computed indicators
        let rsi = match ctx.get_indicator(
            &self.config.symbol,
            IndicatorType::Rsi(self.config.rsi_period),
        ) {
            Some(rsi) => rsi,
            None => return AgentAction::none(), // Not enough data yet
        };

        // RSI < oversold_threshold: buy signal
        if rsi < self.config.oversold_threshold && self.can_buy() {
            let order = self.generate_buy_order(ctx);
            self.state.record_order();
            return AgentAction::single(order);
        }

        // RSI > overbought_threshold: sell signal
        if rsi > self.config.overbought_threshold && self.can_sell() {
            let order = self.generate_sell_order(ctx);
            self.state.record_order();
            return AgentAction::single(order);
        }

        AgentAction::none()
    }

    fn on_fill(&mut self, trade: &Trade) {
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        } else if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        "Momentum"
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StrategyContext;
    use quant::IndicatorSnapshot;
    use sim_core::SingleSymbolMarket;
    use std::collections::HashMap;
    use types::{Candle, Order, OrderId, Symbol};

    fn setup_test_context(
        rsi: Option<f64>,
    ) -> (
        sim_core::OrderBook,
        HashMap<Symbol, Vec<Candle>>,
        IndicatorSnapshot,
        HashMap<Symbol, Vec<Trade>>,
    ) {
        // Create order book with bids and asks
        let mut book = sim_core::OrderBook::new("ACME");
        let mut bid = Order::limit(
            AgentId(99),
            "ACME",
            OrderSide::Buy,
            Price::from_float(99.0),
            Quantity(100),
        );
        bid.id = OrderId(1);
        let mut ask = Order::limit(
            AgentId(99),
            "ACME",
            OrderSide::Sell,
            Price::from_float(101.0),
            Quantity(100),
        );
        ask.id = OrderId(2);
        book.add_order(bid).unwrap();
        book.add_order(ask).unwrap();

        let candles = HashMap::new();
        let recent_trades = HashMap::new();

        let mut indicators = IndicatorSnapshot::new(100);
        if let Some(rsi_value) = rsi {
            let mut symbol_indicators = HashMap::new();
            symbol_indicators.insert(IndicatorType::Rsi(14), rsi_value);
            indicators.insert("ACME".to_string(), symbol_indicators);
        }

        (book, candles, indicators, recent_trades)
    }

    #[test]
    fn test_momentum_buys_on_oversold() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let (book, candles, indicators, recent_trades) = setup_test_context(Some(25.0)); // Oversold
        let market = SingleSymbolMarket::new(&book);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = trader.on_tick(&ctx);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_momentum_sells_on_overbought() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let (book, candles, indicators, recent_trades) = setup_test_context(Some(75.0)); // Overbought
        let market = SingleSymbolMarket::new(&book);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = trader.on_tick(&ctx);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }

    #[test]
    fn test_momentum_no_action_neutral() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let (book, candles, indicators, recent_trades) = setup_test_context(Some(50.0)); // Neutral
        let market = SingleSymbolMarket::new(&book);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = trader.on_tick(&ctx);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_momentum_no_action_without_indicator() {
        let mut trader = MomentumTrader::with_defaults(AgentId(1));
        let (book, candles, indicators, recent_trades) = setup_test_context(None);
        let market = SingleSymbolMarket::new(&book);
        let events = vec![];
        let fundamentals = news::SymbolFundamentals::default();
        let ctx = StrategyContext::new(
            100,
            1000,
            &market,
            &candles,
            &indicators,
            &recent_trades,
            &events,
            &fundamentals,
        );

        let action = trader.on_tick(&ctx);
        assert!(action.orders.is_empty());
    }
}
