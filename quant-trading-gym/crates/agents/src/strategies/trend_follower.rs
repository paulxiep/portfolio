//! Trend Following Trader - follows SMA crossover signals.
//!
//! A classic trend-following strategy that uses two Simple Moving Averages
//! (fast and slow) to identify trend direction and generate trading signals.
//!
//! # Strategy Logic
//! - **Buy signal**: Fast SMA crosses above Slow SMA (golden cross)
//! - **Sell signal**: Fast SMA crosses below Slow SMA (death cross)
//! - Tracks crossover state to avoid repeated signals
//!
//! # Configuration
//! The strategy is fully declarative via [`TrendFollowerConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, MarketData};
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a Trend Following trader.
#[derive(Debug, Clone)]
pub struct TrendFollowerConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Fast SMA period (shorter period, more responsive).
    pub fast_period: usize,
    /// Slow SMA period (longer period, smoother).
    pub slow_period: usize,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
}

impl Default for TrendFollowerConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            fast_period: 10,
            slow_period: 50,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
        }
    }
}

/// Crossover state to track signal changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossoverState {
    /// Not enough data to determine state.
    Unknown,
    /// Fast SMA is above slow SMA (bullish).
    Above,
    /// Fast SMA is below slow SMA (bearish).
    Below,
}

/// Trend Following trader using SMA crossover strategy.
///
/// This is a classic trend-following approach that aims to capture
/// sustained price movements by identifying trend direction through
/// moving average relationships.
pub struct TrendFollower {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: TrendFollowerConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Previous crossover state for signal detection.
    prev_state: CrossoverState,
}

impl TrendFollower {
    /// Create a new TrendFollower with the given configuration.
    pub fn new(id: AgentId, config: TrendFollowerConfig) -> Self {
        assert!(
            config.fast_period < config.slow_period,
            "Fast period must be less than slow period"
        );
        let initial_cash = config.initial_cash;
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
            prev_state: CrossoverState::Unknown,
        }
    }

    /// Create a TrendFollower with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, TrendFollowerConfig::default())
    }

    /// Get the IndicatorTypes this strategy requires.
    pub fn required_indicators(&self) -> Vec<IndicatorType> {
        vec![
            IndicatorType::Sma(self.config.fast_period),
            IndicatorType::Sma(self.config.slow_period),
        ]
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

    /// Determine the current crossover state.
    fn current_crossover_state(&self, fast_sma: f64, slow_sma: f64) -> CrossoverState {
        if fast_sma > slow_sma {
            CrossoverState::Above
        } else {
            CrossoverState::Below
        }
    }
}

impl Agent for TrendFollower {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Get both SMAs from pre-computed indicators
        let fast_sma = match market.get_indicator(IndicatorType::Sma(self.config.fast_period)) {
            Some(v) => v,
            None => return AgentAction::none(),
        };

        let slow_sma = match market.get_indicator(IndicatorType::Sma(self.config.slow_period)) {
            Some(v) => v,
            None => return AgentAction::none(),
        };

        let current_state = self.current_crossover_state(fast_sma, slow_sma);
        let prev_state = self.prev_state;
        self.prev_state = current_state;

        // Only act on crossover events (state changes)
        match (prev_state, current_state) {
            // Golden cross: fast crosses above slow -> buy
            (CrossoverState::Below, CrossoverState::Above) if self.can_buy() => {
                let order = self.generate_buy_order(market);
                self.state.record_order();
                AgentAction::single(order)
            }
            // Death cross: fast crosses below slow -> sell
            (CrossoverState::Above, CrossoverState::Below) if self.can_sell() => {
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
        "TrendFollow"
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

    fn make_market_data(fast_sma: Option<f64>, slow_sma: Option<f64>) -> MarketData {
        use quant::IndicatorSnapshot;

        let indicators = match (fast_sma, slow_sma) {
            (Some(fast), Some(slow)) => {
                let mut snap = IndicatorSnapshot::new(100);
                let mut indicators = HashMap::new();
                indicators.insert(IndicatorType::Sma(10), fast);
                indicators.insert(IndicatorType::Sma(50), slow);
                snap.insert("ACME".to_string(), indicators);
                Some(snap)
            }
            _ => None,
        };

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
    fn test_trend_golden_cross_buys() {
        let mut trader = TrendFollower::with_defaults(AgentId(1));

        // First tick: fast below slow (set prev_state)
        let market1 = make_market_data(Some(49.0), Some(50.0));
        let _ = trader.on_tick(&market1);

        // Second tick: fast above slow (golden cross!)
        let market2 = make_market_data(Some(51.0), Some(50.0));
        let action = trader.on_tick(&market2);

        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_trend_death_cross_sells() {
        let mut trader = TrendFollower::with_defaults(AgentId(1));

        // First tick: fast above slow
        let market1 = make_market_data(Some(51.0), Some(50.0));
        let _ = trader.on_tick(&market1);

        // Second tick: fast below slow (death cross!)
        let market2 = make_market_data(Some(49.0), Some(50.0));
        let action = trader.on_tick(&market2);

        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }

    #[test]
    fn test_trend_no_action_without_crossover() {
        let mut trader = TrendFollower::with_defaults(AgentId(1));

        // Both ticks: fast above slow (no crossover)
        let market1 = make_market_data(Some(51.0), Some(50.0));
        let _ = trader.on_tick(&market1);

        let market2 = make_market_data(Some(52.0), Some(50.0));
        let action = trader.on_tick(&market2);

        assert!(action.orders.is_empty());
    }
}
