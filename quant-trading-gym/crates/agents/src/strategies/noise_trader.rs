//! Noise Trader - generates random market activity.
//!
//! A simple agent that places random orders near the current mid price.
//! This provides liquidity and price discovery by generating trades.
//!
//! # Zombie Risk Prevention
//! NoiseTrader orders use the current mid price as reference. If there's no
//! mid price (empty book), it uses the last trade price. If neither exists,
//! it falls back to a configured initial price.

use crate::state::AgentState;
use crate::{Agent, AgentAction, MarketData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a NoiseTrader agent.
#[derive(Debug, Clone)]
pub struct NoiseTraderConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Probability of placing an order each tick (0.0 to 1.0).
    pub order_probability: f64,
    /// Maximum price deviation from mid as a fraction (e.g., 0.02 = 2%).
    pub price_deviation: f64,
    /// Minimum order size.
    pub min_quantity: u64,
    /// Maximum order size.
    pub max_quantity: u64,
    /// Initial price reference when market is empty.
    pub initial_price: Price,
    /// Starting cash balance.
    pub initial_cash: Cash,
}

impl Default for NoiseTraderConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            order_probability: 0.3,
            price_deviation: 0.02,
            min_quantity: 10,
            max_quantity: 100,
            initial_price: Price::from_float(100.0),
            initial_cash: Cash::from_float(100_000.0),
        }
    }
}

/// A random trader that generates market activity.
///
/// NoiseTraders provide essential liquidity and price movement in the
/// simulation. They place limit orders randomly near the current mid price.
pub struct NoiseTrader {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: NoiseTraderConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// Random number generator (Send-compatible).
    rng: StdRng,
}

impl NoiseTrader {
    /// Create a new NoiseTrader with the given configuration.
    pub fn new(id: AgentId, config: NoiseTraderConfig) -> Self {
        let initial_cash = config.initial_cash;
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
            rng: StdRng::from_os_rng(),
        }
    }

    /// Create a new NoiseTrader with a specific seed (for reproducible testing).
    pub fn with_seed(id: AgentId, config: NoiseTraderConfig, seed: u64) -> Self {
        let initial_cash = config.initial_cash;
        Self {
            id,
            config,
            state: AgentState::new(initial_cash),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a NoiseTrader with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, NoiseTraderConfig::default())
    }

    /// Get current position.
    pub fn position(&self) -> i64 {
        self.state.position()
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.state.cash()
    }

    /// Determine the reference price for order generation.
    fn get_reference_price(&self, market: &MarketData) -> Price {
        // Priority: mid price > last trade > initial price
        market
            .mid_price()
            .or(market.last_price)
            .unwrap_or(self.config.initial_price)
    }

    /// Generate a random order around the reference price.
    fn generate_order(&mut self, reference_price: Price) -> Order {
        // Random side
        let side = if self.rng.random_bool(0.5) {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };

        // Random price within deviation range
        let deviation_range = self.config.price_deviation;
        let deviation = self.rng.random_range(-deviation_range..deviation_range);
        let price_float = reference_price.to_float() * (1.0 + deviation);
        let price = Price::from_float(price_float.max(0.01)); // Ensure positive

        // Random quantity
        let quantity = Quantity(
            self.rng
                .random_range(self.config.min_quantity..=self.config.max_quantity),
        );

        Order::limit(self.id, &self.config.symbol, side, price, quantity)
    }
}

impl Agent for NoiseTrader {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Randomly decide whether to place an order
        if !self.rng.random_bool(self.config.order_probability) {
            return AgentAction::none();
        }

        let reference_price = self.get_reference_price(market);
        let order = self.generate_order(reference_price);

        self.state.record_order();
        AgentAction::single(order)
    }

    fn on_fill(&mut self, trade: &Trade) {
        let trade_value = trade.value();

        if trade.buyer_id == self.id {
            self.state.on_buy(trade.quantity.raw(), trade_value);
        } else if trade.seller_id == self.id {
            self.state.on_sell(trade.quantity.raw(), trade_value);
        }
    }

    fn name(&self) -> &str {
        "NoiseTrader"
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
    use types::BookSnapshot;

    fn mock_market_data(mid_price: Option<Price>) -> MarketData {
        let book = if let Some(mid) = mid_price {
            let bid = Price::from_float(mid.to_float() - 0.5);
            let ask = Price::from_float(mid.to_float() + 0.5);
            BookSnapshot {
                symbol: "ACME".to_string(),
                bids: vec![types::BookLevel {
                    price: bid,
                    quantity: Quantity(100),
                    order_count: 1,
                }],
                asks: vec![types::BookLevel {
                    price: ask,
                    quantity: Quantity(100),
                    order_count: 1,
                }],
                ..Default::default()
            }
        } else {
            BookSnapshot::default()
        };

        MarketData {
            tick: 1,
            timestamp: 0,
            book_snapshot: book,
            recent_trades: vec![],
            last_price: mid_price,
        }
    }

    #[test]
    fn test_noise_trader_creation() {
        let trader = NoiseTrader::with_defaults(AgentId(1));
        assert_eq!(trader.id(), AgentId(1));
        assert_eq!(trader.position(), 0);
        assert_eq!(trader.cash(), Cash::from_float(100_000.0));
    }

    #[test]
    fn test_reference_price_priority() {
        let trader = NoiseTrader::with_defaults(AgentId(1));

        // Empty market: use initial price
        let market = mock_market_data(None);
        let ref_price = trader.get_reference_price(&market);
        assert_eq!(ref_price, Price::from_float(100.0));

        // With mid price: use mid price
        let market = mock_market_data(Some(Price::from_float(150.0)));
        let ref_price = trader.get_reference_price(&market);
        assert_eq!(ref_price, Price::from_float(150.0));
    }

    #[test]
    fn test_on_fill_updates_state() {
        let mut trader = NoiseTrader::with_defaults(AgentId(1));

        // Simulate a buy fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: types::OrderId(1),
            seller_order_id: types::OrderId(2),
            price: Price::from_float(100.0),
            quantity: Quantity(10),
            timestamp: 0,
            tick: 1,
        };

        trader.on_fill(&trade);

        assert_eq!(trader.position(), 10);
        // Cash decreased by 10 * 100 = 1000
        assert_eq!(trader.cash(), Cash::from_float(99_000.0));
    }

    #[test]
    fn test_on_fill_sell_updates_state() {
        let mut trader = NoiseTrader::with_defaults(AgentId(1));

        // Simulate a sell fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(2),
            seller_id: AgentId(1),
            buyer_order_id: types::OrderId(2),
            seller_order_id: types::OrderId(1),
            price: Price::from_float(100.0),
            quantity: Quantity(10),
            timestamp: 0,
            tick: 1,
        };

        trader.on_fill(&trade);

        assert_eq!(trader.position(), -10);
        // Cash increased by 10 * 100 = 1000
        assert_eq!(trader.cash(), Cash::from_float(101_000.0));
    }
}
