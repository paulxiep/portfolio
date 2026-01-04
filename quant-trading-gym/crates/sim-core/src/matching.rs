//! Matching engine implementing price-time priority.
//!
//! The matching engine processes incoming orders against the order book,
//! executing trades at the best available prices. V2.2 adds Fill tracking
//! for per-level execution details and slippage measurement.

use types::{
    Fill, FillId, Order, OrderSide, OrderStatus, OrderType, Price, Quantity, SlippageMetrics, Tick,
    Timestamp, Trade, TradeId,
};

use crate::order_book::OrderBook;

/// Result of attempting to match an order.
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Trades that occurred during matching (aggregated view).
    pub trades: Vec<Trade>,
    /// Individual fills at each price level (V2.2).
    pub fills: Vec<Fill>,
    /// Slippage metrics aggregated across all fills (V2.2).
    pub slippage_metrics: SlippageMetrics,
    /// Updated status of the incoming order.
    pub status: OrderStatus,
    /// Remaining quantity of the incoming order (if any).
    pub remaining_quantity: Quantity,
}

impl MatchResult {
    /// Check if any trades occurred.
    pub fn has_trades(&self) -> bool {
        !self.trades.is_empty()
    }

    /// Check if any fills occurred.
    pub fn has_fills(&self) -> bool {
        !self.fills.is_empty()
    }

    /// Total quantity filled.
    pub fn filled_quantity(&self) -> Quantity {
        self.trades.iter().map(|t| t.quantity).sum()
    }

    /// Number of price levels crossed.
    pub fn levels_crossed(&self) -> usize {
        self.fills.len()
    }

    /// Volume-weighted average price of all fills.
    pub fn vwap(&self) -> Option<Price> {
        self.slippage_metrics.vwap()
    }
}

/// Context for match execution, bundling common parameters.
///
/// Used internally to reduce argument count in matching functions.
struct MatchContext<'a> {
    symbol: &'a str,
    timestamp: Timestamp,
    tick: Tick,
    reference_price: Option<Price>,
}

/// Matching engine for executing orders against an order book.
///
/// Implements price-time priority matching:
/// - Buy orders match against the lowest ask prices first
/// - Sell orders match against the highest bid prices first
/// - Within a price level, orders are matched in FIFO order
#[derive(Debug, Clone)]
pub struct MatchingEngine {
    /// Counter for generating unique trade IDs.
    next_trade_id: u64,
    /// Counter for generating unique fill IDs (V2.2).
    next_fill_id: u64,
}

impl Default for MatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchingEngine {
    /// Create a new matching engine.
    pub fn new() -> Self {
        Self {
            next_trade_id: 1,
            next_fill_id: 1,
        }
    }

    /// Generate the next trade ID.
    fn next_trade_id(&mut self) -> TradeId {
        let id = TradeId(self.next_trade_id);
        self.next_trade_id += 1;
        id
    }

    /// Generate the next fill ID (V2.2).
    fn next_fill_id(&mut self) -> FillId {
        let id = FillId(self.next_fill_id);
        self.next_fill_id += 1;
        id
    }

    /// Process an incoming order against the order book.
    ///
    /// This method will:
    /// 1. Try to match the order against existing orders
    /// 2. Execute trades for any matches found
    /// 3. Generate fills for each execution at a price level
    /// 4. Track slippage metrics
    /// 5. Return remaining quantity if not fully filled
    ///
    /// Note: This does NOT add unfilled limit orders to the book.
    /// The caller must add any remaining quantity if desired.
    pub fn match_order(
        &mut self,
        book: &mut OrderBook,
        order: &mut Order,
        timestamp: Timestamp,
        tick: Tick,
    ) -> MatchResult {
        self.match_order_with_reference(book, order, timestamp, tick, book.mid_price())
    }

    /// Process an incoming order with an explicit reference price for slippage calculation.
    ///
    /// The reference price is used to measure execution quality. Typically this is
    /// the mid price at the time the order was submitted.
    pub fn match_order_with_reference(
        &mut self,
        book: &mut OrderBook,
        order: &mut Order,
        timestamp: Timestamp,
        tick: Tick,
        reference_price: Option<Price>,
    ) -> MatchResult {
        let mut result = MatchResult {
            trades: Vec::new(),
            fills: Vec::new(),
            slippage_metrics: SlippageMetrics::new(reference_price),
            status: OrderStatus::Pending,
            remaining_quantity: order.remaining_quantity,
        };

        let symbol = book.symbol().to_string();

        let ctx = MatchContext {
            symbol: &symbol,
            timestamp,
            tick,
            reference_price,
        };

        // Determine which side of the book to match against
        match order.side {
            OrderSide::Buy => {
                self.match_buy_order(book, order, &mut result, &ctx);
            }
            OrderSide::Sell => {
                self.match_sell_order(book, order, &mut result, &ctx);
            }
        }

        // Update order status based on remaining quantity
        if result.remaining_quantity.is_zero() {
            result.status = OrderStatus::Filled;
            order.status = OrderStatus::Filled;
        } else if result.remaining_quantity < order.quantity {
            result.status = OrderStatus::PartialFill {
                filled: order.quantity - result.remaining_quantity,
            };
            order.status = result.status;
        }

        order.remaining_quantity = result.remaining_quantity;
        result
    }

    /// Match a buy order against asks (sell orders).
    #[allow(clippy::too_many_arguments)]
    fn match_buy_order(
        &mut self,
        book: &mut OrderBook,
        order: &mut Order,
        result: &mut MatchResult,
        ctx: &MatchContext<'_>,
    ) {
        let limit_price = match order.order_type {
            OrderType::Limit { price } => Some(price),
            OrderType::Market => None, // Market orders have no price limit
        };

        // Keep matching while there's quantity remaining
        while !result.remaining_quantity.is_zero() {
            // Get best ask price first (separate borrow)
            let Some(ask_price) = book.best_ask_price() else {
                break; // No more asks available
            };

            // Check if price is acceptable
            if let Some(limit) = limit_price
                && ask_price > limit
            {
                break; // Ask price too high for our limit
            }

            // Peek at the best ask order info
            let Some((resting_agent_id, resting_order_id, resting_qty)) =
                book.peek_best_ask_order()
            else {
                break;
            };

            let trade_quantity = result.remaining_quantity.min(resting_qty);

            // Create the trade (aggregated view)
            let trade = Trade {
                id: self.next_trade_id(),
                symbol: ctx.symbol.to_string(),
                buyer_id: order.agent_id,
                seller_id: resting_agent_id,
                buyer_order_id: order.id,
                seller_order_id: resting_order_id,
                price: ask_price, // Trade at resting order's price
                quantity: trade_quantity,
                timestamp: ctx.timestamp,
                tick: ctx.tick,
            };

            // Create the fill (V2.2 - per-level execution detail)
            let fill = Fill {
                id: self.next_fill_id(),
                symbol: ctx.symbol.to_string(),
                order_id: order.id,
                aggressor_id: order.agent_id,
                resting_id: resting_agent_id,
                resting_order_id,
                aggressor_side: OrderSide::Buy,
                price: ask_price,
                quantity: trade_quantity,
                reference_price: ctx.reference_price,
                timestamp: ctx.timestamp,
                tick: ctx.tick,
            };

            // Track slippage metrics
            result
                .slippage_metrics
                .record_fill(ask_price, trade_quantity);

            result.trades.push(trade);
            result.fills.push(fill);
            result.remaining_quantity -= trade_quantity;

            // Update the book (fills and potentially removes the order)
            book.fill_best_ask(trade_quantity);
            book.set_last_price(ask_price);
        }
    }

    /// Match a sell order against bids (buy orders).
    #[allow(clippy::too_many_arguments)]
    fn match_sell_order(
        &mut self,
        book: &mut OrderBook,
        order: &mut Order,
        result: &mut MatchResult,
        ctx: &MatchContext<'_>,
    ) {
        let limit_price = match order.order_type {
            OrderType::Limit { price } => Some(price),
            OrderType::Market => None,
        };

        while !result.remaining_quantity.is_zero() {
            // Get best bid price first (separate borrow)
            let Some(bid_price) = book.best_bid_price() else {
                break;
            };

            // Check if price is acceptable
            if let Some(limit) = limit_price
                && bid_price < limit
            {
                break; // Bid price too low for our limit
            }

            // Peek at the best bid order info
            let Some((resting_agent_id, resting_order_id, resting_qty)) =
                book.peek_best_bid_order()
            else {
                break;
            };

            let trade_quantity = result.remaining_quantity.min(resting_qty);

            // Create the trade (aggregated view)
            let trade = Trade {
                id: self.next_trade_id(),
                symbol: ctx.symbol.to_string(),
                buyer_id: resting_agent_id,
                seller_id: order.agent_id,
                buyer_order_id: resting_order_id,
                seller_order_id: order.id,
                price: bid_price,
                quantity: trade_quantity,
                timestamp: ctx.timestamp,
                tick: ctx.tick,
            };

            // Create the fill (V2.2 - per-level execution detail)
            let fill = Fill {
                id: self.next_fill_id(),
                symbol: ctx.symbol.to_string(),
                order_id: order.id,
                aggressor_id: order.agent_id,
                resting_id: resting_agent_id,
                resting_order_id,
                aggressor_side: OrderSide::Sell,
                price: bid_price,
                quantity: trade_quantity,
                reference_price: ctx.reference_price,
                timestamp: ctx.timestamp,
                tick: ctx.tick,
            };

            // Track slippage metrics
            result
                .slippage_metrics
                .record_fill(bid_price, trade_quantity);

            result.trades.push(trade);
            result.fills.push(fill);
            result.remaining_quantity -= trade_quantity;

            // Update the book
            book.fill_best_bid(trade_quantity);
            book.set_last_price(bid_price);
        }
    }

    /// Check if an incoming order would match (without executing).
    pub fn would_match(&self, book: &OrderBook, order: &Order) -> bool {
        match order.side {
            OrderSide::Buy => {
                if let Some(ask_price) = book.best_ask_price() {
                    match order.order_type {
                        OrderType::Market => true,
                        OrderType::Limit { price } => price >= ask_price,
                    }
                } else {
                    false
                }
            }
            OrderSide::Sell => {
                if let Some(bid_price) = book.best_bid_price() {
                    match order.order_type {
                        OrderType::Market => true,
                        OrderType::Limit { price } => price <= bid_price,
                    }
                } else {
                    false
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use types::{AgentId, OrderId, Price};

    fn make_limit_order(
        id: u64,
        agent_id: u64,
        side: OrderSide,
        price: f64,
        quantity: u64,
    ) -> Order {
        let mut order = Order::limit(
            AgentId(agent_id),
            "TEST",
            side,
            Price::from_float(price),
            Quantity(quantity),
        );
        order.id = OrderId(id);
        order
    }

    fn make_market_order(id: u64, agent_id: u64, side: OrderSide, quantity: u64) -> Order {
        let mut order = Order::market(AgentId(agent_id), "TEST", side, Quantity(quantity));
        order.id = OrderId(id);
        order
    }

    #[test]
    fn test_no_match_empty_book() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();
        let mut order = make_limit_order(1, 1, OrderSide::Buy, 100.0, 50);

        let result = engine.match_order(&mut book, &mut order, 0, 0);

        assert!(!result.has_trades());
        assert_eq!(result.remaining_quantity, 50);
        assert_eq!(result.status, OrderStatus::Pending);
    }

    #[test]
    fn test_exact_match() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add a sell order to the book
        let sell_order = make_limit_order(1, 1, OrderSide::Sell, 100.0, 50);
        book.add_order(sell_order).unwrap();

        // Submit a buy order that exactly matches
        let mut buy_order = make_limit_order(2, 2, OrderSide::Buy, 100.0, 50);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.remaining_quantity, 0);
        assert_eq!(result.status, OrderStatus::Filled);

        let trade = &result.trades[0];
        assert_eq!(trade.quantity, 50);
        assert_eq!(trade.price, Price::from_float(100.0));
        assert_eq!(trade.buyer_id, AgentId(2));
        assert_eq!(trade.seller_id, AgentId(1));

        // Book should be empty now
        assert!(book.is_empty());
    }

    #[test]
    fn test_partial_match() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add a sell order for 30 shares
        let sell_order = make_limit_order(1, 1, OrderSide::Sell, 100.0, 30);
        book.add_order(sell_order).unwrap();

        // Submit a buy order for 50 shares
        let mut buy_order = make_limit_order(2, 2, OrderSide::Buy, 100.0, 50);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.remaining_quantity, 20);
        assert!(matches!(result.status, OrderStatus::PartialFill { filled } if filled == 30));

        // Book should be empty (sell order fully consumed)
        assert!(book.is_empty());
    }

    #[test]
    fn test_match_multiple_levels() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add sell orders at different price levels
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 101.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 100.0, 20))
            .unwrap();
        book.add_order(make_limit_order(3, 3, OrderSide::Sell, 102.0, 50))
            .unwrap();

        // Buy order should match lowest prices first
        let mut buy_order = make_limit_order(4, 4, OrderSide::Buy, 102.0, 60);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades.len(), 3);
        assert_eq!(result.remaining_quantity, 0);

        // First trade at $100 (best ask)
        assert_eq!(result.trades[0].price, Price::from_float(100.0));
        assert_eq!(result.trades[0].quantity, 20);

        // Second trade at $101
        assert_eq!(result.trades[1].price, Price::from_float(101.0));
        assert_eq!(result.trades[1].quantity, 30);

        // Third trade at $102
        assert_eq!(result.trades[2].price, Price::from_float(102.0));
        assert_eq!(result.trades[2].quantity, 10);

        // Some of the $102 order should remain
        assert_eq!(book.ask_depth(10), 40); // 50 - 10 = 40
    }

    #[test]
    fn test_time_priority() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add two sell orders at the same price
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 100.0, 30))
            .unwrap();

        // Buy order should match first order completely before second
        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 100.0, 40);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades.len(), 2);

        // First trade should be with agent 1 (time priority)
        assert_eq!(result.trades[0].seller_id, AgentId(1));
        assert_eq!(result.trades[0].quantity, 30);

        // Second trade with agent 2
        assert_eq!(result.trades[1].seller_id, AgentId(2));
        assert_eq!(result.trades[1].quantity, 10);
    }

    #[test]
    fn test_limit_price_respected() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add sell orders
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 105.0, 50))
            .unwrap();

        // Buy limit at $102 should only match the $100 order
        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 102.0, 100);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].price, Price::from_float(100.0));
        assert_eq!(result.remaining_quantity, 50);
    }

    #[test]
    fn test_market_order_buy() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add sell orders at different prices
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 110.0, 30))
            .unwrap();

        // Market order should sweep through all prices
        let mut market_order = make_market_order(3, 3, OrderSide::Buy, 50);
        let result = engine.match_order(&mut book, &mut market_order, 1000, 1);

        assert_eq!(result.trades.len(), 2);
        assert_eq!(result.remaining_quantity, 0);
        assert_eq!(result.filled_quantity(), 50);
    }

    #[test]
    fn test_market_order_sell() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add buy orders at different prices
        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Buy, 95.0, 30))
            .unwrap();

        // Market sell should hit highest bids first
        let mut market_order = make_market_order(3, 3, OrderSide::Sell, 40);
        let result = engine.match_order(&mut book, &mut market_order, 1000, 1);

        assert_eq!(result.trades.len(), 2);
        assert_eq!(result.trades[0].price, Price::from_float(100.0)); // Best bid first
        assert_eq!(result.trades[0].quantity, 30);
        assert_eq!(result.trades[1].price, Price::from_float(95.0));
        assert_eq!(result.trades[1].quantity, 10);
    }

    #[test]
    fn test_would_match() {
        let mut book = OrderBook::new("TEST");
        let engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Buy, 98.0, 50))
            .unwrap();

        // Buy at 100 would match
        let buy_100 = make_limit_order(3, 3, OrderSide::Buy, 100.0, 10);
        assert!(engine.would_match(&book, &buy_100));

        // Buy at 99 would not match (below best ask)
        let buy_99 = make_limit_order(4, 4, OrderSide::Buy, 99.0, 10);
        assert!(!engine.would_match(&book, &buy_99));

        // Market buy would match
        let market_buy = make_market_order(5, 5, OrderSide::Buy, 10);
        assert!(engine.would_match(&book, &market_buy));

        // Sell at 98 would match
        let sell_98 = make_limit_order(6, 6, OrderSide::Sell, 98.0, 10);
        assert!(engine.would_match(&book, &sell_98));

        // Sell at 99 would not match (above best bid)
        let sell_99 = make_limit_order(7, 7, OrderSide::Sell, 99.0, 10);
        assert!(!engine.would_match(&book, &sell_99));
    }

    #[test]
    fn test_self_trade_prevention_not_implemented() {
        // Note: This test documents that self-trading is currently allowed.
        // Self-trade prevention could be added as a feature later.
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Same agent places both orders
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 50))
            .unwrap();

        let mut buy_order = make_limit_order(2, 1, OrderSide::Buy, 100.0, 50);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        // Currently, this will match (self-trade)
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].buyer_id, result.trades[0].seller_id);
    }

    #[test]
    fn test_last_price_updated() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        assert_eq!(book.last_price(), None);

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 50))
            .unwrap();

        let mut buy_order = make_limit_order(2, 2, OrderSide::Buy, 100.0, 25);
        engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(book.last_price(), Some(Price::from_float(100.0)));
    }

    #[test]
    fn test_trade_ids_increment() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 100))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 101.0, 100))
            .unwrap();

        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 101.0, 150);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.trades[0].id, TradeId(1));
        assert_eq!(result.trades[1].id, TradeId(2));
    }

    // V2.2 Fill Tests
    #[test]
    fn test_fills_created_with_trades() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Add sell orders at different price levels
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 101.0, 30))
            .unwrap();

        // Buy order that crosses two levels
        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 101.0, 50);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        // Should have 2 trades and 2 fills
        assert_eq!(result.trades.len(), 2);
        assert_eq!(result.fills.len(), 2);
        assert!(result.has_fills());

        // Verify first fill
        assert_eq!(result.fills[0].price, Price::from_float(100.0));
        assert_eq!(result.fills[0].quantity, Quantity(30));
        assert_eq!(result.fills[0].aggressor_side, OrderSide::Buy);
        assert_eq!(result.fills[0].aggressor_id, AgentId(3));
        assert_eq!(result.fills[0].resting_id, AgentId(1));

        // Verify second fill
        assert_eq!(result.fills[1].price, Price::from_float(101.0));
        assert_eq!(result.fills[1].quantity, Quantity(20));
    }

    #[test]
    fn test_fill_ids_increment() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 101.0, 50))
            .unwrap();

        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 101.0, 80);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.fills[0].id, FillId(1));
        assert_eq!(result.fills[1].id, FillId(2));
    }

    #[test]
    fn test_slippage_metrics_in_result() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        // Set up book with mid price = $100
        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 101.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 102.0, 50))
            .unwrap();
        book.add_order(make_limit_order(3, 3, OrderSide::Buy, 99.0, 50))
            .unwrap();

        // Buy 70 shares - should fill 50 at $101 and 20 at $102
        let mut buy_order = make_limit_order(4, 4, OrderSide::Buy, 103.0, 70);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        // Check slippage metrics
        let vwap = result.slippage_metrics.vwap().unwrap();
        // VWAP = (50*101 + 20*102) / 70 = (5050 + 2040) / 70 = 101.2857...
        assert!((vwap.to_float() - 101.2857).abs() < 0.01);

        assert_eq!(result.slippage_metrics.levels_crossed, 2);
        assert_eq!(
            result.slippage_metrics.best_fill_price,
            Some(Price::from_float(101.0))
        );
        assert_eq!(
            result.slippage_metrics.worst_fill_price,
            Some(Price::from_float(102.0))
        );
    }

    #[test]
    fn test_vwap_method_on_result() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 100))
            .unwrap();

        let mut buy_order = make_limit_order(2, 2, OrderSide::Buy, 100.0, 50);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        // Single fill, VWAP should equal fill price
        let vwap = result.vwap().unwrap();
        assert_eq!(vwap, Price::from_float(100.0));
    }

    #[test]
    fn test_levels_crossed_method() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Sell, 101.0, 30))
            .unwrap();
        book.add_order(make_limit_order(3, 3, OrderSide::Sell, 102.0, 30))
            .unwrap();

        let mut buy_order = make_limit_order(4, 4, OrderSide::Buy, 102.0, 80);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        assert_eq!(result.levels_crossed(), 3);
    }

    #[test]
    fn test_fill_reference_price() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Sell, 101.0, 50))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Buy, 99.0, 50))
            .unwrap();

        // Mid price is $100
        let mid = book.mid_price();
        assert_eq!(mid, Some(Price::from_float(100.0)));

        let mut buy_order = make_limit_order(3, 3, OrderSide::Buy, 101.0, 30);
        let result = engine.match_order(&mut book, &mut buy_order, 1000, 1);

        // Fill should have mid price as reference
        assert_eq!(
            result.fills[0].reference_price,
            Some(Price::from_float(100.0))
        );
    }

    #[test]
    fn test_sell_order_fills() {
        let mut book = OrderBook::new("TEST");
        let mut engine = MatchingEngine::new();

        book.add_order(make_limit_order(1, 1, OrderSide::Buy, 100.0, 30))
            .unwrap();
        book.add_order(make_limit_order(2, 2, OrderSide::Buy, 99.0, 30))
            .unwrap();

        // Market sell order
        let mut sell_order = make_market_order(3, 3, OrderSide::Sell, 50);
        let result = engine.match_order(&mut book, &mut sell_order, 1000, 1);

        assert_eq!(result.fills.len(), 2);
        assert_eq!(result.fills[0].aggressor_side, OrderSide::Sell);
        assert_eq!(result.fills[0].price, Price::from_float(100.0)); // Best bid first
        assert_eq!(result.fills[1].price, Price::from_float(99.0));
    }
}
