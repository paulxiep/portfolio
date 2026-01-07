//! Batch auction matching engine for parallel order processing.
//!
//! Unlike continuous matching where orders are processed sequentially and can
//! match against each other within a batch, batch auctions:
//!
//! 1. Collect all orders for a symbol
//! 2. Compute a single clearing price
//! 3. Match all crossing orders at that price
//!
//! This enables full parallelization: each symbol can be processed independently.
//!
//! # Clearing Price Algorithm
//!
//! The clearing price maximizes executed volume:
//! - Sort bids descending by price, asks ascending
//! - Build cumulative supply/demand curves
//! - Find price where supply crosses demand
//! - In case of multiple valid prices, use midpoint
//!
//! # Trade Generation
//!
//! All trades execute at the clearing price. Orders are matched in price-time priority:
//! - Bids above clearing price matched first (descending price, then time)
//! - Asks below clearing price matched first (ascending price, then time)

use std::cmp::Reverse;
use std::collections::HashMap;

use types::{
    Order, OrderId, OrderSide, OrderType, Price, Quantity, Tick, Timestamp, Trade, TradeId,
};

/// Result of a batch auction for a single symbol.
#[derive(Debug, Clone, Default)]
pub struct BatchAuctionResult {
    /// The clearing price (None if no trades occurred).
    pub clearing_price: Option<Price>,
    /// All trades executed at the clearing price.
    pub trades: Vec<Trade>,
    /// Orders that were fully filled.
    pub filled_orders: Vec<OrderId>,
    /// Orders that were partially filled (order_id, filled_qty).
    pub partial_fills: Vec<(OrderId, Quantity)>,
    /// Orders that didn't participate (price didn't cross).
    pub unfilled_orders: Vec<OrderId>,
    /// Total buy volume that could have executed at clearing price.
    pub total_bid_volume: Quantity,
    /// Total sell volume that could have executed at clearing price.
    pub total_ask_volume: Quantity,
}

impl BatchAuctionResult {
    /// Check if any trades occurred.
    pub fn has_trades(&self) -> bool {
        !self.trades.is_empty()
    }

    /// Total quantity traded.
    pub fn traded_volume(&self) -> Quantity {
        self.trades.iter().map(|t| t.quantity).sum()
    }
}

/// Batch auction engine for a single symbol.
///
/// Processes all orders for a symbol in one batch, computing a single
/// clearing price and matching all crossing orders.
pub struct BatchAuction {
    /// Counter for generating unique trade IDs.
    next_trade_id: u64,
}

impl Default for BatchAuction {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchAuction {
    /// Create a new batch auction engine.
    pub fn new() -> Self {
        Self { next_trade_id: 1 }
    }

    /// Create with a starting trade ID (for coordinating across symbols).
    pub fn with_starting_id(start_id: u64) -> Self {
        Self {
            next_trade_id: start_id,
        }
    }

    /// Get the next trade ID (for external coordination).
    pub fn peek_next_id(&self) -> u64 {
        self.next_trade_id
    }

    /// Generate the next trade ID.
    fn next_trade_id(&mut self) -> TradeId {
        let id = TradeId(self.next_trade_id);
        self.next_trade_id += 1;
        id
    }

    /// Run a batch auction on the provided orders.
    ///
    /// # Arguments
    /// * `symbol` - The symbol being auctioned
    /// * `orders` - All orders for this symbol (will be partitioned into bids/asks)
    /// * `timestamp` - Current timestamp for trades
    /// * `tick` - Current tick for trades
    /// * `reference_price` - Last traded price (anchor for market orders)
    ///
    /// # Returns
    /// Auction result with clearing price, trades, and order status updates.
    pub fn run(
        &mut self,
        symbol: &str,
        orders: Vec<Order>,
        timestamp: Timestamp,
        tick: Tick,
        reference_price: Option<Price>,
    ) -> BatchAuctionResult {
        if orders.is_empty() {
            return BatchAuctionResult::default();
        }

        // Partition into bids and asks
        let (bids, asks): (Vec<_>, Vec<_>) =
            orders.into_iter().partition(|o| o.side == OrderSide::Buy);

        if bids.is_empty() || asks.is_empty() {
            // No matching possible without both sides
            let unfilled: Vec<_> = bids.iter().chain(asks.iter()).map(|o| o.id).collect();
            return BatchAuctionResult {
                unfilled_orders: unfilled,
                ..Default::default()
            };
        }

        // Find clearing price (use reference price as anchor for market orders)
        let clearing_price = self.compute_clearing_price(&bids, &asks, reference_price);

        let Some(price) = clearing_price else {
            // No crossing - all orders unfilled
            let unfilled: Vec<_> = bids.iter().chain(asks.iter()).map(|o| o.id).collect();
            return BatchAuctionResult {
                unfilled_orders: unfilled,
                ..Default::default()
            };
        };

        // Match orders at clearing price
        self.match_at_price(symbol, bids, asks, price, timestamp, tick)
    }

    /// Compute the clearing price that maximizes volume.
    ///
    /// Algorithm:
    /// 1. Collect all unique prices from bids and asks
    /// 2. Include reference price as a candidate (anchors market orders)
    /// 3. For each candidate price, compute executable volume
    /// 4. Return price with maximum volume (prefer reference price on ties)
    fn compute_clearing_price(
        &self,
        bids: &[Order],
        asks: &[Order],
        reference_price: Option<Price>,
    ) -> Option<Price> {
        // Collect all limit prices as candidates
        let mut candidates: Vec<Price> = bids
            .iter()
            .chain(asks.iter())
            .filter_map(|o| o.limit_price())
            .collect();

        // Include reference price as a candidate - this is crucial for stability!
        // Without this, market orders can cause clearing price to jump to extreme values.
        if let Some(ref_price) = reference_price {
            candidates.push(ref_price);
        }

        if candidates.is_empty() {
            // All market orders with no reference - can't determine price
            return None;
        }

        candidates.sort();
        candidates.dedup();

        // For each candidate, compute executable volume
        let mut best_volume = Quantity::ZERO;
        let mut best_prices: Vec<Price> = Vec::new();

        for &price in &candidates {
            // Demand at price: sum of bids >= price (willing to pay at least this much)
            let demand: Quantity = bids
                .iter()
                .filter(|b| self.bid_crosses_price(b, price))
                .map(|b| b.remaining_quantity)
                .sum();

            // Supply at price: sum of asks <= price (willing to sell at this price or lower)
            let supply: Quantity = asks
                .iter()
                .filter(|a| self.ask_crosses_price(a, price))
                .map(|a| a.remaining_quantity)
                .sum();

            // Executable volume is min of supply and demand
            let volume = demand.min(supply);

            if volume > best_volume {
                best_volume = volume;
                best_prices.clear();
                best_prices.push(price);
            } else if volume == best_volume && !volume.is_zero() {
                best_prices.push(price);
            }
        }

        if best_volume.is_zero() || best_prices.is_empty() {
            return None;
        }

        // If multiple prices have same volume, prefer reference price for stability
        if best_prices.len() == 1 {
            Some(best_prices[0])
        } else if let Some(ref_price) = reference_price {
            // If reference price is among the best, use it for stability
            if best_prices.contains(&ref_price) {
                Some(ref_price)
            } else {
                // Otherwise use midpoint of range
                let low = best_prices.first().unwrap();
                let high = best_prices.last().unwrap();
                Some(Price((low.raw() + high.raw()) / 2))
            }
        } else {
            // No reference price - use midpoint of range (average of low and high)
            let low = best_prices.first().unwrap();
            let high = best_prices.last().unwrap();
            Some(Price((low.raw() + high.raw()) / 2))
        }
    }

    /// Check if a bid order crosses (is willing to trade at) a given price.
    fn bid_crosses_price(&self, bid: &Order, price: Price) -> bool {
        match bid.order_type {
            OrderType::Market => true,
            OrderType::Limit { price: limit } => limit >= price,
        }
    }

    /// Check if an ask order crosses (is willing to trade at) a given price.
    fn ask_crosses_price(&self, ask: &Order, price: Price) -> bool {
        match ask.order_type {
            OrderType::Market => true,
            OrderType::Limit { price: limit } => limit <= price,
        }
    }

    /// Match orders at the clearing price.
    ///
    /// Uses price-time priority within each side.
    fn match_at_price(
        &mut self,
        symbol: &str,
        bids: Vec<Order>,
        asks: Vec<Order>,
        clearing_price: Price,
        timestamp: Timestamp,
        tick: Tick,
    ) -> BatchAuctionResult {
        // Filter to orders that cross at clearing price
        let mut crossing_bids: Vec<_> = bids
            .iter()
            .filter(|b| self.bid_crosses_price(b, clearing_price))
            .cloned()
            .collect();

        let mut crossing_asks: Vec<_> = asks
            .iter()
            .filter(|a| self.ask_crosses_price(a, clearing_price))
            .cloned()
            .collect();

        // Sort by price-time priority
        // Bids: highest price first (most aggressive), then by order ID (time proxy)
        // For market orders, use a very high price to sort them first
        crossing_bids.sort_by_key(|b| {
            let price = b.limit_price().unwrap_or(Price(i64::MAX));
            (Reverse(price), b.id)
        });

        // Asks: lowest price first (most aggressive), then by order ID
        // For market orders, use zero price to sort them first
        crossing_asks.sort_by_key(|a| {
            let price = a.limit_price().unwrap_or(Price::ZERO);
            (price, a.id)
        });

        // Track volumes
        let total_bid_volume: Quantity = crossing_bids.iter().map(|b| b.remaining_quantity).sum();
        let total_ask_volume: Quantity = crossing_asks.iter().map(|a| a.remaining_quantity).sum();

        // Match orders
        let mut trades = Vec::new();
        let mut filled_orders = Vec::new();
        let mut partial_fills: HashMap<OrderId, Quantity> = HashMap::new();

        let mut bid_idx = 0;
        let mut ask_idx = 0;
        let mut bid_remaining: Vec<Quantity> =
            crossing_bids.iter().map(|b| b.remaining_quantity).collect();
        let mut ask_remaining: Vec<Quantity> =
            crossing_asks.iter().map(|a| a.remaining_quantity).collect();

        while bid_idx < crossing_bids.len() && ask_idx < crossing_asks.len() {
            let bid = &crossing_bids[bid_idx];
            let ask = &crossing_asks[ask_idx];

            let trade_qty = bid_remaining[bid_idx].min(ask_remaining[ask_idx]);

            if trade_qty.is_zero() {
                // Shouldn't happen, but safety check
                if bid_remaining[bid_idx].is_zero() {
                    bid_idx += 1;
                }
                if ask_idx < ask_remaining.len() && ask_remaining[ask_idx].is_zero() {
                    ask_idx += 1;
                }
                continue;
            }

            // Create trade
            let trade = Trade {
                id: self.next_trade_id(),
                symbol: symbol.to_string(),
                buyer_id: bid.agent_id,
                seller_id: ask.agent_id,
                buyer_order_id: bid.id,
                seller_order_id: ask.id,
                price: clearing_price,
                quantity: trade_qty,
                timestamp,
                tick,
            };
            trades.push(trade);

            // Update remaining quantities
            bid_remaining[bid_idx] -= trade_qty;
            ask_remaining[ask_idx] -= trade_qty;

            // Track fills
            *partial_fills.entry(bid.id).or_insert(Quantity::ZERO) += trade_qty;
            *partial_fills.entry(ask.id).or_insert(Quantity::ZERO) += trade_qty;

            // Advance indices if fully filled
            if bid_remaining[bid_idx].is_zero() {
                filled_orders.push(bid.id);
                bid_idx += 1;
            }
            if ask_idx < ask_remaining.len() && ask_remaining[ask_idx].is_zero() {
                filled_orders.push(ask.id);
                ask_idx += 1;
            }
        }

        // Collect unfilled orders (didn't cross at all)
        let unfilled_orders: Vec<_> = bids
            .iter()
            .chain(asks.iter())
            .filter(|o| !partial_fills.contains_key(&o.id))
            .map(|o| o.id)
            .collect();

        // Convert partial fills, excluding fully filled
        let partial_fills: Vec<_> = partial_fills
            .into_iter()
            .filter(|(id, _)| !filled_orders.contains(id))
            .collect();

        BatchAuctionResult {
            clearing_price: Some(clearing_price),
            trades,
            filled_orders,
            partial_fills,
            unfilled_orders,
            total_bid_volume,
            total_ask_volume,
        }
    }
}

/// Run batch auctions for multiple symbols in parallel.
///
/// Each symbol is processed independently with its own orders.
/// Returns results keyed by symbol.
///
/// # Arguments
/// * `orders_by_symbol` - Orders grouped by symbol
/// * `reference_prices` - Last traded price per symbol (for price stability)
/// * `timestamp` - Current timestamp
/// * `tick` - Current tick
/// * `starting_trade_id` - Starting ID for trades
///
/// # Note on Trade IDs
///
/// Each symbol gets a separate BatchAuction instance with sequential IDs.
/// For globally unique IDs, the caller should offset based on symbol index
/// or use a different ID scheme.
#[cfg(feature = "parallel")]
pub fn run_parallel_auctions(
    orders_by_symbol: HashMap<String, Vec<Order>>,
    reference_prices: &HashMap<String, Price>,
    timestamp: Timestamp,
    tick: Tick,
    starting_trade_id: u64,
) -> HashMap<String, BatchAuctionResult> {
    use rayon::prelude::*;

    // Estimate max trades per symbol for ID spacing
    let symbols: Vec<_> = orders_by_symbol.keys().cloned().collect();
    let max_orders_per_symbol = orders_by_symbol
        .values()
        .map(|v| v.len() as u64)
        .max()
        .unwrap_or(0);

    symbols
        .into_par_iter()
        .enumerate()
        .map(|(idx, symbol)| {
            let orders = orders_by_symbol.get(&symbol).cloned().unwrap_or_default();
            let ref_price = reference_prices.get(&symbol).copied();
            // Space out trade IDs so they don't collide across parallel executions
            let symbol_start_id = starting_trade_id + (idx as u64 * max_orders_per_symbol);
            let mut auction = BatchAuction::with_starting_id(symbol_start_id);
            let result = auction.run(&symbol, orders, timestamp, tick, ref_price);
            (symbol, result)
        })
        .collect()
}

/// Sequential version of multi-symbol batch auctions.
#[cfg(not(feature = "parallel"))]
pub fn run_parallel_auctions(
    orders_by_symbol: HashMap<String, Vec<Order>>,
    reference_prices: &HashMap<String, Price>,
    timestamp: Timestamp,
    tick: Tick,
    starting_trade_id: u64,
) -> HashMap<String, BatchAuctionResult> {
    let mut results = HashMap::new();
    let mut auction = BatchAuction::with_starting_id(starting_trade_id);

    for (symbol, orders) in orders_by_symbol {
        let ref_price = reference_prices.get(&symbol).copied();
        let result = auction.run(&symbol, orders, timestamp, tick, ref_price);
        results.insert(symbol, result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{AgentId, OrderId};

    fn make_bid(id: u64, agent: u64, price: f64, qty: u64) -> Order {
        let mut order = Order::limit(
            AgentId(agent),
            "TEST",
            OrderSide::Buy,
            Price::from_float(price),
            Quantity(qty),
        );
        order.id = OrderId(id);
        order
    }

    fn make_ask(id: u64, agent: u64, price: f64, qty: u64) -> Order {
        let mut order = Order::limit(
            AgentId(agent),
            "TEST",
            OrderSide::Sell,
            Price::from_float(price),
            Quantity(qty),
        );
        order.id = OrderId(id);
        order
    }

    #[test]
    fn test_no_crossing() {
        let mut auction = BatchAuction::new();

        // Bids at 99, asks at 101 - no crossing
        let orders = vec![make_bid(1, 1, 99.0, 100), make_ask(2, 2, 101.0, 100)];

        let result = auction.run("TEST", orders, 0, 0, None);

        assert!(result.clearing_price.is_none());
        assert!(result.trades.is_empty());
        assert_eq!(result.unfilled_orders.len(), 2);
    }

    #[test]
    fn test_simple_crossing() {
        let mut auction = BatchAuction::new();

        // Bid at 100, ask at 100 - exact crossing
        let orders = vec![make_bid(1, 1, 100.0, 50), make_ask(2, 2, 100.0, 50)];

        let result = auction.run("TEST", orders, 0, 0, None);

        assert_eq!(result.clearing_price, Some(Price::from_float(100.0)));
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].quantity, Quantity(50));
        assert_eq!(result.trades[0].price, Price::from_float(100.0));
        assert_eq!(result.filled_orders.len(), 2);
    }

    #[test]
    fn test_multiple_price_levels() {
        let mut auction = BatchAuction::new();

        // Multiple bids and asks that cross
        let orders = vec![
            make_bid(1, 1, 102.0, 100), // Aggressive buyer
            make_bid(2, 2, 101.0, 100), // Less aggressive
            make_bid(3, 3, 100.0, 100), // At clearing
            make_ask(4, 4, 98.0, 100),  // Aggressive seller
            make_ask(5, 5, 99.0, 100),  // Less aggressive
            make_ask(6, 6, 100.0, 100), // At clearing
        ];

        let result = auction.run("TEST", orders, 0, 0, None);

        // Should find a clearing price where supply meets demand
        assert!(result.clearing_price.is_some());
        assert!(!result.trades.is_empty());

        // All trades should be at the clearing price
        let cp = result.clearing_price.unwrap();
        for trade in &result.trades {
            assert_eq!(trade.price, cp);
        }
    }

    #[test]
    fn test_partial_fill() {
        let mut auction = BatchAuction::new();

        // Bid for 100, ask for only 30
        let orders = vec![make_bid(1, 1, 100.0, 100), make_ask(2, 2, 100.0, 30)];

        let result = auction.run("TEST", orders, 0, 0, None);

        assert_eq!(result.clearing_price, Some(Price::from_float(100.0)));
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].quantity, Quantity(30));

        // Ask should be fully filled, bid partially filled
        assert!(result.filled_orders.contains(&OrderId(2)));
        assert!(
            result
                .partial_fills
                .iter()
                .any(|(id, qty)| *id == OrderId(1) && *qty == Quantity(30))
        );
    }

    #[test]
    fn test_price_time_priority() {
        let mut auction = BatchAuction::new();

        // Two bids at same price - earlier one (lower ID) should match first
        let orders = vec![
            make_bid(1, 1, 100.0, 50), // Earlier
            make_bid(2, 2, 100.0, 50), // Later
            make_ask(3, 3, 100.0, 50), // Only enough for one bid
        ];

        let result = auction.run("TEST", orders, 0, 0, None);

        assert_eq!(result.trades.len(), 1);
        // Buyer should be agent 1 (earlier order)
        assert_eq!(result.trades[0].buyer_id, AgentId(1));
        assert!(result.filled_orders.contains(&OrderId(1)));
    }
}
