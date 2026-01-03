//! Bollinger Bands Mean Reversion Trader.
//!
//! A mean reversion strategy that uses Bollinger Bands to identify when
//! price has moved too far from its mean and is likely to revert.
//!
//! # Strategy Logic
//! - **Buy signal**: Price touches or crosses below the lower band (oversold)
//! - **Sell signal**: Price touches or crosses above the upper band (overbought)
//! - Uses %B indicator: (Price - Lower) / (Upper - Lower)
//!   - %B < 0: Below lower band -> buy
//!   - %B > 1: Above upper band -> sell
//!
//! # Configuration
//! The strategy is fully declarative via [`BollingerReversionConfig`].

use crate::state::AgentState;
use crate::{Agent, AgentAction, MarketData};
use quant::BollingerBands;
use types::{AgentId, Cash, IndicatorType, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a Bollinger Bands Reversion trader.
#[derive(Debug, Clone)]
pub struct BollingerReversionConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Bollinger Bands period (typically 20).
    pub period: usize,
    /// Standard deviation multiplier (typically 2.0).
    pub std_dev_multiplier: f64,
    /// Order size for each trade.
    pub order_size: u64,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Maximum position size (absolute value).
    pub max_position: i64,
    /// Lower %B threshold for buy signal (typically 0.0 or slightly above).
    pub lower_threshold: f64,
    /// Upper %B threshold for sell signal (typically 1.0 or slightly below).
    pub upper_threshold: f64,
}

impl Default for BollingerReversionConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            period: 20,
            std_dev_multiplier: 2.0,
            order_size: 50,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
            max_position: 500,
            lower_threshold: 0.0, // Buy when at or below lower band
            upper_threshold: 1.0, // Sell when at or above upper band
        }
    }
}

/// Bollinger Bands Mean Reversion trader.
///
/// This strategy is based on the statistical concept that prices tend to
/// return to their mean. When price moves beyond the bands (2 std devs
/// by default), it's considered overextended and likely to revert.
pub struct BollingerReversion {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: BollingerReversionConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Bollinger Bands calculator.
    bollinger: BollingerBands,
}

impl BollingerReversion {
    /// Create a new BollingerReversion with the given configuration.
    pub fn new(id: AgentId, config: BollingerReversionConfig) -> Self {
        let initial_cash = config.initial_cash;
        let bollinger = BollingerBands::new(config.period, config.std_dev_multiplier);
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
            bollinger,
        }
    }

    /// Create a BollingerReversion with default (20, 2.0) configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, BollingerReversionConfig::default())
    }

    /// Get the IndicatorType this strategy uses.
    pub fn required_indicator(&self) -> IndicatorType {
        IndicatorType::BollingerBands {
            period: self.config.period,
            std_dev_bp: (self.config.std_dev_multiplier * 100.0) as u32,
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
        // For mean reversion, we want to buy at or below the lower band
        // Use a limit slightly above current price to improve fill
        let order_price = Price::from_float(price.to_float() * 1.001);
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
        // For mean reversion, we want to sell at or above the upper band
        // Use a limit slightly below current price to improve fill
        let order_price = Price::from_float(price.to_float() * 0.999);
        Order::limit(
            self.id,
            &self.config.symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }
}

impl Agent for BollingerReversion {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Calculate full Bollinger Bands output from candles
        let bb_output = match self.bollinger.calculate_full(&market.candles) {
            Some(output) => output,
            None => return AgentAction::none(), // Not enough data
        };

        // %B indicates where current price is relative to the bands
        // %B < lower_threshold: price at/below lower band -> buy
        // %B > upper_threshold: price at/above upper band -> sell
        let percent_b = bb_output.percent_b;

        if percent_b <= self.config.lower_threshold && self.can_buy() {
            let order = self.generate_buy_order(market);
            self.state.record_order();
            return AgentAction::single(order);
        }

        if percent_b >= self.config.upper_threshold && self.can_sell() {
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
        "Bollinger"
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

    /// Generate candles with high volatility that will push price beyond bands.
    fn make_volatile_candles(spike_up: bool, count: usize) -> Vec<Candle> {
        let base = 100.0;
        let mut candles = Vec::with_capacity(count);

        for i in 0..count {
            // Create mostly stable prices, then a spike at the end
            let close = if i == count - 1 {
                if spike_up {
                    base + 10.0 // 10% spike up (beyond 2 std devs)
                } else {
                    base - 10.0 // 10% spike down
                }
            } else {
                // Add small random-ish variation to make bands meaningful
                base + ((i % 3) as f64 - 1.0) * 0.5
            };

            candles.push(Candle {
                symbol: "ACME".to_string(),
                open: Price::from_float(close - 0.1),
                high: Price::from_float(close + 0.2),
                low: Price::from_float(close - 0.2),
                close: Price::from_float(close),
                volume: Quantity(1000),
                timestamp: i as u64,
                tick: i as u64,
            });
        }

        candles
    }

    fn make_market_data(candles: Vec<Candle>) -> MarketData {
        let last_price = candles.last().map(|c| c.close);

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
            last_price,
            candles,
            indicators: None,
        }
    }

    #[test]
    fn test_bollinger_needs_enough_data() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        // Only 10 candles, need 20 minimum
        let candles = make_volatile_candles(true, 10);
        let market = make_market_data(candles);

        let action = trader.on_tick(&market);
        assert!(action.orders.is_empty());
    }

    #[test]
    fn test_bollinger_buys_on_lower_band() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        // Price spikes down beyond lower band
        let candles = make_volatile_candles(false, 25);
        let market = make_market_data(candles);

        let action = trader.on_tick(&market);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Buy);
    }

    #[test]
    fn test_bollinger_sells_on_upper_band() {
        let mut trader = BollingerReversion::with_defaults(AgentId(1));
        // Price spikes up beyond upper band
        let candles = make_volatile_candles(true, 25);
        let market = make_market_data(candles);

        let action = trader.on_tick(&market);
        assert_eq!(action.orders.len(), 1);
        assert_eq!(action.orders[0].side, OrderSide::Sell);
    }
}
