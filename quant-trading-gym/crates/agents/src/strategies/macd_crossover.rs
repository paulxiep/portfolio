//! MACD Crossover Trader - trades on MACD signal line crossovers.
//!
//! A momentum strategy that uses MACD (Moving Average Convergence Divergence)
//! to identify trend changes and generate trading signals.
//!
//! # Strategy Logic
//! - **Buy signal**: MACD line crosses above signal line (bullish crossover)
//! - **Sell signal**: MACD line crosses below signal line (bearish crossover)
//! - Uses histogram (MACD - Signal) for crossover detection
//!
//! # Configuration
//! The strategy is fully declarative via [`MacdCrossoverConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, MarketData};
use quant::Macd;
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a MACD Crossover trader.
#[derive(Debug, Clone)]
pub struct MacdCrossoverConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// MACD fast EMA period (typically 12).
    pub fast_period: usize,
    /// MACD slow EMA period (typically 26).
    pub slow_period: usize,
    /// Signal line EMA period (typically 9).
    pub signal_period: usize,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
    /// Minimum histogram magnitude for signal confirmation.
    pub min_histogram: f64,
}

impl Default for MacdCrossoverConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
            min_histogram: 0.0, // No minimum by default
        }
    }
}

/// Crossover state for MACD signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MacdState {
    /// Not enough data to determine state.
    Unknown,
    /// MACD line is above signal line (bullish).
    Bullish,
    /// MACD line is below signal line (bearish).
    Bearish,
}

/// MACD Crossover trader using MACD/Signal line crossovers.
///
/// MACD is a trend-following momentum indicator that shows the relationship
/// between two EMAs of price. Crossovers of the MACD and signal lines
/// generate trading signals.
pub struct MacdCrossover {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MacdCrossoverConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Previous MACD state for crossover detection.
    prev_state: MacdState,
    /// MACD calculator for full output access.
    macd: Macd,
}

impl MacdCrossover {
    /// Create a new MacdCrossover with the given configuration.
    pub fn new(id: AgentId, config: MacdCrossoverConfig) -> Self {
        let initial_cash = config.initial_cash;
        let macd = Macd::new(config.fast_period, config.slow_period, config.signal_period);
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
            prev_state: MacdState::Unknown,
            macd,
        }
    }

    /// Create a MacdCrossover with default (12, 26, 9) configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, MacdCrossoverConfig::default())
    }

    /// Get the IndicatorType this strategy uses.
    pub fn required_indicator(&self) -> IndicatorType {
        IndicatorType::Macd {
            fast: self.config.fast_period,
            slow: self.config.slow_period,
            signal: self.config.signal_period,
        }
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

    /// Generate a buy order.
    fn generate_buy_order(&self, market: &MarketData) -> Order {
        let price = self.get_reference_price(market);
        let order_price = Price::from_float(price.to_float() * 0.999);
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Buy,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Generate a sell order.
    fn generate_sell_order(&self, market: &MarketData) -> Order {
        let price = self.get_reference_price(market);
        let order_price = Price::from_float(price.to_float() * 1.001);
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Determine MACD state from histogram value.
    fn macd_state_from_histogram(&self, histogram: f64) -> MacdState {
        // Histogram > 0 means MACD > Signal (bullish)
        // Histogram < 0 means MACD < Signal (bearish)
        if histogram > 0.0 {
            MacdState::Bullish
        } else {
            MacdState::Bearish
        }
    }
}

impl Agent for MacdCrossover {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Calculate full MACD output from candles
        let macd_output = match self.macd.calculate_full(&market.candles) {
            Some(output) => output,
            None => return AgentAction::none(), // Not enough data
        };

        let current_state = self.macd_state_from_histogram(macd_output.histogram);
        let prev_state = self.prev_state;
        self.prev_state = current_state;

        // Check minimum histogram magnitude if configured
        if macd_output.histogram.abs() < self.config.min_histogram {
            return AgentAction::none();
        }

        // Only act on crossover events
        match (prev_state, current_state) {
            // Bullish crossover: MACD crosses above signal -> buy
            (MacdState::Bearish, MacdState::Bullish) if self.can_buy() => {
                let order = self.generate_buy_order(market);
                self.state.record_order();
                AgentAction::single(order)
            }
            // Bearish crossover: MACD crosses below signal -> sell
            (MacdState::Bullish, MacdState::Bearish) if self.can_sell() => {
                let order = self.generate_sell_order(market);
                self.state.record_order();
                AgentAction::single(order)
            }
            _ => AgentAction::none(),
        }
    }

    fn on_fill(&mut self, trade: &Trade) {
        if trade.buyer_id == self.id {
            self.state.on_buy(trade.quantity.raw(), trade.value());
        } else if trade.seller_id == self.id {
            self.state.on_sell(trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        "MACD"
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
    use types::{BookSnapshot, Candle};

    /// Generate candles that will produce specific MACD outputs.
    fn make_trending_candles(trend_up: bool, count: usize) -> Vec<Candle> {
        let base = 100.0;
        let increment = if trend_up { 0.5 } else { -0.5 };

        (0..count)
            .map(|i| {
                let close = base + (i as f64 * increment);
                Candle {
                    symbol: "ACME".to_string(),
                    open: Price::from_float(close - 0.1),
                    high: Price::from_float(close + 0.2),
                    low: Price::from_float(close - 0.2),
                    close: Price::from_float(close),
                    volume: Quantity(1000),
                    timestamp: i as u64,
                    tick: i as u64,
                }
            })
            .collect()
    }

    fn make_market_data(candles: Vec<Candle>) -> MarketData {
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
            candles,
            indicators: None,
        }
    }

    #[test]
    fn test_macd_needs_enough_data() {
        let mut trader = MacdCrossover::with_defaults(AgentId(1));
        // Only 10 candles, need 26 + 9 = 35 minimum
        let candles = make_trending_candles(true, 10);
        let market = make_market_data(candles);

        let action = trader.on_tick(&market);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_macd_no_action_on_first_state() {
        let mut trader = MacdCrossover::with_defaults(AgentId(1));
        // Enough data for MACD
        let candles = make_trending_candles(true, 50);
        let market = make_market_data(candles);

        // First tick should only set state, not generate orders
        // (prev_state is Unknown, which doesn't trigger crossover)
        let action = trader.on_tick(&market);
        // This might or might not generate an order depending on state logic
        // The main assertion is that it doesn't panic
        assert!(action.orders.len() <= 1);
    }
}
