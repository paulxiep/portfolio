//! Market Maker - provides liquidity with bid/ask spread.
//!
//! A market maker continuously quotes two-sided markets by placing
//! both bid and ask orders. It seeds liquidity and helps prevent
//! the "zombie simulation" problem where no trades occur.
//!
//! # Strategy
//! - Maintains bid/ask orders around the current mid price
//! - Adjusts quotes based on inventory (skew away from large positions)
//! - Cancels stale orders when prices move significantly

use crate::{Agent, AgentAction, MarketData};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Trade};

/// Configuration for a MarketMaker agent.
#[derive(Debug, Clone)]
pub struct MarketMakerConfig {
    /// Symbol to trade.
    pub symbol: String,
    /// Half-spread as a fraction of mid price (e.g., 0.005 = 0.5%).
    pub half_spread: f64,
    /// Order size to quote on each side.
    pub quote_size: u64,
    /// Initial price to seed the market (used when book is empty).
    pub initial_price: Price,
    /// Starting cash balance.
    pub initial_cash: Cash,
    /// Maximum inventory before skewing quotes (in shares).
    pub max_inventory: i64,
    /// Inventory skew factor (how much to adjust price per unit of inventory).
    pub inventory_skew: f64,
    /// Ticks between quote refreshes.
    pub refresh_interval: u64,
}

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            symbol: "ACME".to_string(),
            half_spread: 0.005, // 0.5% half spread = 1% total spread
            quote_size: 100,
            initial_price: Price::from_float(100.0),
            initial_cash: Cash::from_float(1_000_000.0),
            max_inventory: 1000,
            inventory_skew: 0.0001, // Adjust price 0.01% per share of inventory
            refresh_interval: 5,
        }
    }
}

/// Internal state tracking for MarketMaker.
#[derive(Debug, Clone, Default)]
struct MarketMakerState {
    /// Current position in shares (positive = long, negative = short).
    position: i64,
    /// Current cash balance.
    cash: Cash,
    /// Last tick when quotes were placed.
    last_quote_tick: u64,
    /// Total number of orders placed.
    orders_placed: u64,
    /// Total number of fills received.
    fills_received: u64,
    /// Total trading volume.
    total_volume: u64,
}

/// A market maker that provides liquidity.
///
/// MarketMakers are essential for a functioning market. They seed the
/// order book with initial quotes and continuously provide two-sided
/// liquidity, enabling price discovery and trade execution.
pub struct MarketMaker {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: MarketMakerConfig,
    /// Internal state.
    state: MarketMakerState,
}

impl MarketMaker {
    /// Create a new MarketMaker with the given configuration.
    pub fn new(id: AgentId, config: MarketMakerConfig) -> Self {
        let initial_cash = config.initial_cash;
        Self {
            id,
            config,
            state: MarketMakerState {
                cash: initial_cash,
                ..Default::default()
            },
        }
    }

    /// Create a MarketMaker with default configuration.
    pub fn with_defaults(id: AgentId) -> Self {
        Self::new(id, MarketMakerConfig::default())
    }

    /// Get current position.
    pub fn position(&self) -> i64 {
        self.state.position
    }

    /// Get current cash balance.
    pub fn cash(&self) -> Cash {
        self.state.cash
    }

    /// Determine the reference price for quoting.
    fn get_reference_price(&self, market: &MarketData) -> Price {
        // Priority: mid price > last trade > initial price
        market
            .mid_price()
            .or(market.last_price)
            .unwrap_or(self.config.initial_price)
    }

    /// Calculate inventory-adjusted skew.
    ///
    /// When we have positive inventory (long), we want to:
    /// - Lower our ask to sell more easily
    /// - Lower our bid to reduce buying
    ///
    /// When we have negative inventory (short), we do the opposite.
    fn calculate_skew(&self) -> f64 {
        // Clamp inventory to avoid extreme skews
        let clamped_inventory = self
            .state
            .position
            .clamp(-self.config.max_inventory, self.config.max_inventory);

        // Negative skew means lower prices (to sell inventory)
        // Positive skew means higher prices (to buy back)
        -self.config.inventory_skew * clamped_inventory as f64
    }

    /// Generate bid and ask orders around reference price.
    fn generate_quotes(&self, reference_price: Price) -> Vec<Order> {
        let ref_float = reference_price.to_float();
        let half_spread = self.config.half_spread;
        let skew = self.calculate_skew();

        // Calculate bid and ask prices with inventory skew
        let bid_price = Price::from_float(ref_float * (1.0 - half_spread + skew));
        let ask_price = Price::from_float(ref_float * (1.0 + half_spread + skew));

        let quote_size = Quantity(self.config.quote_size);

        vec![
            Order::limit(
                self.id,
                &self.config.symbol,
                OrderSide::Buy,
                bid_price,
                quote_size,
            ),
            Order::limit(
                self.id,
                &self.config.symbol,
                OrderSide::Sell,
                ask_price,
                quote_size,
            ),
        ]
    }

    /// Check if we should refresh quotes this tick.
    fn should_refresh(&self, current_tick: u64) -> bool {
        current_tick == 0
            || current_tick >= self.state.last_quote_tick + self.config.refresh_interval
    }
}

impl Agent for MarketMaker {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, market: &MarketData) -> AgentAction {
        // Only refresh quotes periodically
        if !self.should_refresh(market.tick) {
            return AgentAction::none();
        }

        let reference_price = self.get_reference_price(market);
        let orders = self.generate_quotes(reference_price);

        self.state.orders_placed += orders.len() as u64;
        self.state.last_quote_tick = market.tick;

        AgentAction::multiple(orders)
    }

    fn on_fill(&mut self, trade: &Trade) {
        self.state.fills_received += 1;
        self.state.total_volume += trade.quantity.raw();

        let trade_value = trade.value();

        if trade.buyer_id == self.id {
            // We bought: increase position, decrease cash
            self.state.position += trade.quantity.raw() as i64;
            self.state.cash -= trade_value;
        } else if trade.seller_id == self.id {
            // We sold: decrease position, increase cash
            self.state.position -= trade.quantity.raw() as i64;
            self.state.cash += trade_value;
        }
    }

    fn name(&self) -> &str {
        "MarketMaker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::BookSnapshot;

    fn mock_market_data(mid_price: Option<Price>, tick: u64) -> MarketData {
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
                tick,
                ..Default::default()
            }
        } else {
            BookSnapshot::default()
        };

        MarketData {
            tick,
            timestamp: 0,
            book_snapshot: book,
            recent_trades: vec![],
            last_price: mid_price,
        }
    }

    #[test]
    fn test_market_maker_creation() {
        let mm = MarketMaker::with_defaults(AgentId(1));
        assert_eq!(mm.id(), AgentId(1));
        assert_eq!(mm.position(), 0);
        assert_eq!(mm.cash(), Cash::from_float(1_000_000.0));
    }

    #[test]
    fn test_market_maker_generates_two_sided_quotes() {
        let mut mm = MarketMaker::with_defaults(AgentId(1));

        let market = mock_market_data(Some(Price::from_float(100.0)), 0);
        let action = mm.on_tick(&market);

        assert_eq!(action.orders.len(), 2);

        // Find bid and ask
        let bid = action.orders.iter().find(|o| o.side == OrderSide::Buy);
        let ask = action.orders.iter().find(|o| o.side == OrderSide::Sell);

        assert!(bid.is_some());
        assert!(ask.is_some());

        // Bid should be below reference, ask should be above
        let bid_price = bid.unwrap().limit_price().unwrap();
        let ask_price = ask.unwrap().limit_price().unwrap();

        assert!(bid_price < Price::from_float(100.0));
        assert!(ask_price > Price::from_float(100.0));
    }

    #[test]
    fn test_market_maker_respects_refresh_interval() {
        let config = MarketMakerConfig {
            refresh_interval: 10,
            ..Default::default()
        };
        let mut mm = MarketMaker::new(AgentId(1), config);

        // Tick 0: should place orders
        let market = mock_market_data(Some(Price::from_float(100.0)), 0);
        let action = mm.on_tick(&market);
        assert_eq!(action.orders.len(), 2);

        // Tick 5: should NOT place orders
        let market = mock_market_data(Some(Price::from_float(100.0)), 5);
        let action = mm.on_tick(&market);
        assert!(action.orders.is_empty());

        // Tick 10: should place orders again
        let market = mock_market_data(Some(Price::from_float(100.0)), 10);
        let action = mm.on_tick(&market);
        assert_eq!(action.orders.len(), 2);
    }

    #[test]
    fn test_inventory_skew() {
        let mut mm = MarketMaker::with_defaults(AgentId(1));

        // Simulate large long position
        mm.state.position = 500;

        // Skew should be negative (lower prices to sell)
        let skew = mm.calculate_skew();
        assert!(skew < 0.0);

        // Simulate large short position
        mm.state.position = -500;

        // Skew should be positive (higher prices to buy)
        let skew = mm.calculate_skew();
        assert!(skew > 0.0);
    }

    #[test]
    fn test_on_fill_updates_state() {
        let mut mm = MarketMaker::with_defaults(AgentId(1));

        // Simulate a buy fill
        let trade = Trade {
            id: types::TradeId(1),
            symbol: "ACME".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: types::OrderId(1),
            seller_order_id: types::OrderId(2),
            price: Price::from_float(100.0),
            quantity: Quantity(50),
            timestamp: 0,
            tick: 1,
        };

        mm.on_fill(&trade);

        assert_eq!(mm.position(), 50);
        assert_eq!(mm.state.total_volume, 50);
        // Cash decreased by 50 * 100 = 5000
        assert_eq!(mm.cash(), Cash::from_float(995_000.0));
    }
}
