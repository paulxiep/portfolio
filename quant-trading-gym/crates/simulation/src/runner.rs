//! Simulation runner implementing the tick-based event loop.
//!
//! The simulation holds the order book, agents, and coordinates the tick loop.
//!
//! # Position Limits (V2.1)
//!
//! When `enforce_position_limits` is enabled in config, orders are validated
//! against:
//! - Cash sufficiency for buys
//! - Shares outstanding limits for long positions
//! - Short-selling constraints and borrow availability
//!
//! Rejected orders are logged but do not cause errors.

use std::collections::HashMap;

use agents::{Agent, AgentAction, BorrowLedger, PositionValidator, StrategyContext};
use quant::{
    AgentRiskSnapshot, AgentRiskTracker, IndicatorCache, IndicatorEngine, IndicatorSnapshot,
};
use rand::seq::SliceRandom;
use sim_core::{MatchingEngine, OrderBook, SingleSymbolMarket};
use types::{
    AgentId, Candle, Order, OrderId, Price, Quantity, RiskViolation, Symbol, Tick, Timestamp, Trade,
};

use crate::config::SimulationConfig;

/// Statistics about the simulation state.
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Current tick number.
    pub tick: Tick,

    /// Total trades executed.
    pub total_trades: u64,

    /// Total orders submitted.
    pub total_orders: u64,

    /// Total orders that resulted in fills.
    pub filled_orders: u64,

    /// Total orders that were added to book (resting).
    pub resting_orders: u64,

    /// Total orders rejected due to position limit violations (V2.1).
    pub rejected_orders: u64,
}

/// The main simulation runner.
///
/// Coordinates the tick-based event loop:
/// 1. Build market data snapshot
/// 2. Call each agent's `on_tick` to get their orders
/// 3. Validate orders against position limits (V2.1)
/// 4. Process valid orders through the matching engine
/// 5. Update borrow ledger on short trades (V2.1)
/// 6. Notify agents of fills
/// 7. Advance tick counter
pub struct Simulation {
    /// Configuration for this simulation.
    config: SimulationConfig,

    /// The order book.
    book: OrderBook,

    /// The matching engine.
    engine: MatchingEngine,

    /// Trading agents.
    agents: Vec<Box<dyn Agent>>,

    /// Current tick.
    tick: Tick,

    /// Current timestamp.
    timestamp: Timestamp,

    /// Recent trades for market data.
    recent_trades: Vec<Trade>,

    /// Counter for generating unique order IDs.
    next_order_id: u64,

    /// Simulation statistics.
    stats: SimulationStats,

    /// Historical candles for indicator calculations.
    candles: Vec<Candle>,

    /// Current candle being built (OHLCV within candle interval).
    current_candle: Option<CandleBuilder>,

    /// Indicator engine for computing technical indicators.
    indicator_engine: IndicatorEngine,

    /// Indicator cache for current tick (reserved for future per-tick caching).
    #[allow(dead_code)]
    indicator_cache: IndicatorCache,

    /// Per-agent risk tracking.
    risk_tracker: AgentRiskTracker,

    /// Position validator for order validation (V2.1).
    position_validator: PositionValidator,

    /// Borrow ledger for short-selling tracking (V2.1).
    borrow_ledger: BorrowLedger,

    /// Cache of total shares held (sum of all long positions) for validation.
    /// Updated after each trade.
    total_shares_held: Quantity,
}

/// Helper for building candles incrementally.
#[derive(Debug, Clone)]
struct CandleBuilder {
    symbol: String,
    open: Price,
    high: Price,
    low: Price,
    close: Price,
    volume: Quantity,
    #[allow(dead_code)] // Used for debugging/future candle timing features
    start_tick: Tick,
    trade_count: usize,
}

impl CandleBuilder {
    fn new(symbol: &str, price: Price, tick: Tick) -> Self {
        Self {
            symbol: symbol.to_string(),
            open: price,
            high: price,
            low: price,
            close: price,
            volume: Quantity::ZERO,
            start_tick: tick,
            trade_count: 0,
        }
    }

    fn update(&mut self, trade: &Trade) {
        self.high = self.high.max(trade.price);
        self.low = self.low.min(trade.price);
        self.close = trade.price;
        self.volume += trade.quantity;
        self.trade_count += 1;
    }

    fn finalize(self, end_tick: Tick, timestamp: Timestamp) -> Candle {
        Candle {
            symbol: self.symbol,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            timestamp,
            tick: end_tick,
        }
    }
}

impl Simulation {
    /// Create a new simulation with the given configuration.
    pub fn new(config: SimulationConfig) -> Self {
        let book = OrderBook::new(config.symbol());

        // Initialize position validator from config
        let position_validator =
            PositionValidator::new(config.symbol_config.clone(), config.short_selling.clone());

        // Initialize borrow ledger with symbol's borrow pool
        let mut borrow_ledger = BorrowLedger::new();
        borrow_ledger.init_symbol(config.symbol(), config.symbol_config.borrow_pool_size());

        Self {
            book,
            engine: MatchingEngine::new(),
            agents: Vec::new(),
            tick: 0,
            timestamp: 0,
            recent_trades: Vec::new(),
            next_order_id: 1,
            stats: SimulationStats::default(),
            candles: Vec::new(),
            current_candle: None,
            indicator_engine: IndicatorEngine::with_common_indicators(),
            indicator_cache: IndicatorCache::new(),
            risk_tracker: AgentRiskTracker::with_defaults(),
            position_validator,
            borrow_ledger,
            total_shares_held: Quantity::ZERO,
            config,
        }
    }

    /// Create a simulation with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SimulationConfig::default())
    }

    /// Add an agent to the simulation.
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        self.agents.push(agent);
    }

    /// Get the current tick.
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// Get the current timestamp.
    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    /// Get simulation statistics.
    pub fn stats(&self) -> &SimulationStats {
        &self.stats
    }

    /// Get a reference to the order book.
    pub fn book(&self) -> &OrderBook {
        &self.book
    }

    /// Get the number of agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get a reference to the borrow ledger (V2.1).
    pub fn borrow_ledger(&self) -> &BorrowLedger {
        &self.borrow_ledger
    }

    /// Get a reference to the position validator (V2.1).
    pub fn position_validator(&self) -> &PositionValidator {
        &self.position_validator
    }

    /// Get the total shares currently held (long positions) across all agents.
    pub fn total_shares_held(&self) -> Quantity {
        self.total_shares_held
    }

    /// Get agent summaries (name, position, cash) for display.
    /// Returns a vec of (name, position, cash) tuples.
    pub fn agent_summaries(&self) -> Vec<(&str, i64, types::Cash)> {
        self.agents
            .iter()
            .map(|a| (a.name(), a.position(), a.cash()))
            .collect()
    }

    /// Get risk metrics for all agents.
    pub fn agent_risk_metrics(&self) -> HashMap<AgentId, AgentRiskSnapshot> {
        self.risk_tracker.compute_all_metrics()
    }

    /// Get risk metrics for a specific agent.
    pub fn agent_risk(&self, agent_id: AgentId) -> AgentRiskSnapshot {
        self.risk_tracker.compute_metrics(agent_id)
    }

    /// Get recent trades.
    pub fn recent_trades(&self) -> &[Trade] {
        &self.recent_trades
    }

    /// Get historical candles.
    pub fn candles(&self) -> &[Candle] {
        &self.candles
    }

    /// Get a reference to the indicator engine.
    pub fn indicator_engine(&self) -> &IndicatorEngine {
        &self.indicator_engine
    }

    /// Get a mutable reference to the indicator engine for registration.
    pub fn indicator_engine_mut(&mut self) -> &mut IndicatorEngine {
        &mut self.indicator_engine
    }

    /// Generate the next order ID.
    fn next_order_id(&mut self) -> OrderId {
        let id = OrderId(self.next_order_id);
        self.next_order_id += 1;
        id
    }

    /// Build indicator snapshot for current tick.
    fn build_indicator_snapshot(&mut self) -> Option<IndicatorSnapshot> {
        if self.candles.is_empty() {
            return None;
        }

        let mut snapshot = IndicatorSnapshot::new(self.tick);
        let symbol = self.config.symbol().to_string();

        // Compute all registered indicators
        let values = self.indicator_engine.compute_all(&self.candles);
        if !values.is_empty() {
            snapshot.insert(symbol, values);
        }

        Some(snapshot)
    }

    /// Build market data snapshot for agents (deprecated - use StrategyContext).
    #[deprecated(
        since = "2.3.0",
        note = "Use build_candles_map and StrategyContext instead"
    )]
    #[allow(dead_code)]
    fn build_market_data(&mut self) -> agents::MarketData {
        // Build indicator snapshot
        let indicators = self.build_indicator_snapshot();

        agents::MarketData {
            tick: self.tick,
            timestamp: self.timestamp,
            book_snapshot: self.book.snapshot(
                self.timestamp,
                self.tick,
                self.config.snapshot_depth,
            ),
            recent_trades: self.recent_trades.clone(),
            last_price: self.book.last_price(),
            candles: self.candles.clone(),
            indicators,
        }
    }

    /// Build candles map for multi-symbol support.
    fn build_candles_map(&self) -> HashMap<Symbol, Vec<Candle>> {
        let mut map = HashMap::new();
        map.insert(self.config.symbol().into(), self.candles.clone());
        map
    }

    /// Build trades map for multi-symbol support.
    fn build_trades_map(&self) -> HashMap<Symbol, Vec<Trade>> {
        let mut map = HashMap::new();
        map.insert(self.config.symbol().into(), self.recent_trades.clone());
        map
    }

    /// Update candle with trade data.
    fn update_candles(&mut self, trades: &[Trade]) {
        for trade in trades {
            // Initialize candle builder if needed
            if self.current_candle.is_none() {
                self.current_candle = Some(CandleBuilder::new(
                    self.config.symbol(),
                    trade.price,
                    self.tick,
                ));
            }

            // Update current candle
            if let Some(ref mut builder) = self.current_candle {
                builder.update(trade);
            }
        }

        // Check if we should finalize the candle (every candle_interval ticks)
        if self.tick > 0
            && self.tick.is_multiple_of(self.config.candle_interval)
            && let Some(builder) = self.current_candle.take()
        {
            let candle = builder.finalize(self.tick, self.timestamp);
            self.candles.push(candle);

            // Limit candle history
            if self.candles.len() > self.config.max_candles {
                self.candles.remove(0);
            }
        }
    }

    /// Validate an order against position limits (V2.1).
    ///
    /// Returns `Ok(())` if the order passes validation, or `Err(RiskViolation)` if rejected.
    fn validate_order(
        &self,
        order: &Order,
        agent_position: i64,
        agent_cash: types::Cash,
        is_market_maker: bool,
    ) -> Result<(), RiskViolation> {
        if !self.config.enforce_position_limits {
            return Ok(());
        }

        self.position_validator.validate_order(
            order,
            agent_position,
            agent_cash,
            &self.borrow_ledger,
            self.total_shares_held,
            is_market_maker, // MMs exempt from short limit
        )
    }

    /// Update borrow ledger after a trade (V2.1).
    ///
    /// When a short sale occurs, the seller borrows shares.
    /// When covering a short, the buyer returns borrowed shares.
    fn update_borrow_ledger(
        &mut self,
        trade: &Trade,
        seller_position_before: i64,
        buyer_position_before: i64,
    ) {
        // Check if seller is going short (position becoming more negative)
        let seller_position_after = seller_position_before - trade.quantity.raw() as i64;
        if seller_position_after < 0 && seller_position_before >= seller_position_after {
            // Calculate additional borrow needed
            let was_short = seller_position_before.min(0).unsigned_abs();
            let now_short = seller_position_after.unsigned_abs();
            let additional_borrow = now_short.saturating_sub(was_short);

            if additional_borrow > 0 {
                // Attempt to borrow (should succeed as we validated earlier)
                let _ = self.borrow_ledger.borrow(
                    trade.seller_id,
                    &trade.symbol,
                    Quantity(additional_borrow),
                    self.tick,
                );
            }
        }

        // Check if buyer is covering a short (position becoming less negative)
        let buyer_position_after = buyer_position_before + trade.quantity.raw() as i64;
        if buyer_position_before < 0 && buyer_position_after > buyer_position_before {
            // Calculate shares to return
            let was_short = buyer_position_before.unsigned_abs();
            let now_short = buyer_position_after.min(0).unsigned_abs();
            let to_return = was_short.saturating_sub(now_short);

            if to_return > 0 {
                self.borrow_ledger.return_shares(
                    trade.buyer_id,
                    &trade.symbol,
                    Quantity(to_return),
                );
            }
        }
    }

    /// Update total shares held after a trade.
    fn update_total_shares_held(
        &mut self,
        trade: &Trade,
        seller_position_before: i64,
        buyer_position_before: i64,
    ) {
        // Calculate change in aggregate long positions
        let seller_position_after = seller_position_before - trade.quantity.raw() as i64;
        let buyer_position_after = buyer_position_before + trade.quantity.raw() as i64;

        // Seller's long position change (only count positive positions)
        let seller_long_before = seller_position_before.max(0) as u64;
        let seller_long_after = seller_position_after.max(0) as u64;

        // Buyer's long position change
        let buyer_long_before = buyer_position_before.max(0) as u64;
        let buyer_long_after = buyer_position_after.max(0) as u64;

        // Net change
        let delta = (buyer_long_after + seller_long_after) as i64
            - (buyer_long_before + seller_long_before) as i64;

        if delta > 0 {
            self.total_shares_held += Quantity(delta as u64);
        } else {
            self.total_shares_held = self
                .total_shares_held
                .saturating_sub(Quantity((-delta) as u64));
        }
    }

    /// Process a single order through the matching engine.
    ///
    /// Returns the trades that occurred and whether the order is now resting on the book.
    fn process_order(&mut self, mut order: Order) -> (Vec<Trade>, bool) {
        // Assign order ID and timestamp
        order.id = self.next_order_id();
        order.timestamp = self.timestamp;

        self.stats.total_orders += 1;

        // Try to match the order
        let result = self
            .engine
            .match_order(&mut self.book, &mut order, self.timestamp, self.tick);

        let mut is_resting = false;

        // If there's remaining quantity for a limit order, add to book
        if !result.remaining_quantity.is_zero() && order.limit_price().is_some() {
            // Update order's remaining quantity before adding to book
            order.remaining_quantity = result.remaining_quantity;
            if self.book.add_order(order).is_ok() {
                is_resting = true;
                self.stats.resting_orders += 1;
            }
            // Market orders with remaining quantity are dropped (no liquidity)
        }

        if result.has_trades() {
            self.stats.filled_orders += 1;
        }

        (result.trades, is_resting)
    }

    /// Advance the simulation by one tick.
    ///
    /// Returns trades that occurred during this tick.
    pub fn step(&mut self) -> Vec<Trade> {
        let mut tick_trades = Vec::new();

        // Phase 1: Build all owned data for StrategyContext.
        // These must live for the duration of agent ticks.
        let candles_map = self.build_candles_map();
        let trades_map = self.build_trades_map();
        let indicators = self.build_indicator_snapshot().unwrap_or_default();
        let market = SingleSymbolMarket::new(&self.book);

        // Build StrategyContext with references to owned data
        let ctx = StrategyContext::new(
            self.tick,
            self.timestamp,
            &market,
            &candles_map,
            &indicators,
            &trades_map,
        );

        // Shuffle agent processing order to eliminate ID-based priority bias.
        // Without this, lower-indexed agents would always get their orders
        // processed first within each tick, creating unfair advantages.
        let mut indices: Vec<usize> = (0..self.agents.len()).collect();
        indices.shuffle(&mut rand::rng());

        // Phase 2: Collect all agent actions with their current state for validation.
        // We need to snapshot position/cash before processing to validate orders.
        let actions_with_state: Vec<(AgentId, AgentAction, i64, types::Cash, bool)> = indices
            .iter()
            .map(|&i| {
                let agent = &mut self.agents[i];
                let action = agent.on_tick(&ctx);
                let position = agent.position();
                let cash = agent.cash();
                let is_mm = agent.is_market_maker();
                (agent.id(), action, position, cash, is_mm)
            })
            .collect();

        // Phase 3: ctx is now dropped (goes out of scope) - releases borrow on book
        // Process all orders with validation
        let mut fill_notifications: Vec<(AgentId, Trade, i64)> = Vec::new(); // Include position before trade

        for (agent_id, action, agent_position, agent_cash, is_market_maker) in actions_with_state {
            let mut current_position = agent_position;

            for order in action.orders {
                // Validate order against position limits (V2.1)
                if let Err(violation) =
                    self.validate_order(&order, current_position, agent_cash, is_market_maker)
                {
                    self.stats.rejected_orders += 1;
                    if self.config.verbose {
                        eprintln!(
                            "Order rejected for {}: {} (order: {} {} @ {:?})",
                            agent_id,
                            violation,
                            order.side,
                            order.quantity,
                            order.limit_price()
                        );
                    }
                    continue; // Skip this order
                }

                let (trades, _is_resting) = self.process_order(order);

                // Record trades and prepare fill notifications
                for trade in trades {
                    self.stats.total_trades += 1;

                    // Track position before trade for borrow ledger updates
                    let buyer_pos_before = if trade.buyer_id == agent_id {
                        current_position
                    } else {
                        // Need to look up other agent's position
                        self.agents
                            .iter()
                            .find(|a| a.id() == trade.buyer_id)
                            .map(|a| a.position())
                            .unwrap_or(0)
                    };

                    let seller_pos_before = if trade.seller_id == agent_id {
                        current_position
                    } else {
                        self.agents
                            .iter()
                            .find(|a| a.id() == trade.seller_id)
                            .map(|a| a.position())
                            .unwrap_or(0)
                    };

                    // Update borrow ledger (V2.1)
                    self.update_borrow_ledger(&trade, seller_pos_before, buyer_pos_before);

                    // Update total shares held
                    self.update_total_shares_held(&trade, seller_pos_before, buyer_pos_before);

                    // Notify both parties with position before trade
                    fill_notifications.push((trade.buyer_id, trade.clone(), buyer_pos_before));
                    fill_notifications.push((trade.seller_id, trade.clone(), seller_pos_before));

                    // Update tracking of current position for this agent's remaining orders
                    if trade.buyer_id == agent_id {
                        current_position += trade.quantity.raw() as i64;
                    }
                    if trade.seller_id == agent_id {
                        current_position -= trade.quantity.raw() as i64;
                    }

                    tick_trades.push(trade);
                }
            }
        }

        // Add trades to recent trades (newest first)
        for trade in tick_trades.iter().rev() {
            self.recent_trades.insert(0, trade.clone());
        }

        // Trim recent trades to max
        if self.recent_trades.len() > self.config.max_recent_trades {
            self.recent_trades.truncate(self.config.max_recent_trades);
        }

        // Update candles with trade data
        self.update_candles(&tick_trades);

        // Notify agents of fills
        for (agent_id, trade, _pos_before) in fill_notifications {
            if let Some(agent) = self.agents.iter_mut().find(|a| a.id() == agent_id) {
                agent.on_fill(&trade);
            }
        }

        // Update risk tracking with current equity values
        // Use last price or initial price for mark-to-market
        let mark_price = self
            .book
            .last_price()
            .unwrap_or(self.config.initial_price());
        for agent in &self.agents {
            let equity = agent.equity(mark_price).to_float();
            self.risk_tracker.record_equity(agent.id(), equity);
        }

        // Advance time
        self.tick += 1;
        self.timestamp += 1; // Simple 1:1 mapping for now

        // Update stats tick
        self.stats.tick = self.tick;

        tick_trades
    }

    /// Run the simulation for a given number of ticks.
    ///
    /// Returns total trades across all ticks.
    pub fn run(&mut self, ticks: u64) -> Vec<Trade> {
        let mut all_trades = Vec::new();

        for _ in 0..ticks {
            let trades = self.step();
            all_trades.extend(trades);
        }

        all_trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agents::StrategyContext;
    use types::{Order, OrderSide, Price, Quantity};

    /// A simple test agent that does nothing.
    struct PassiveAgent {
        id: AgentId,
    }

    impl Agent for PassiveAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &StrategyContext<'_>) -> AgentAction {
            AgentAction::none()
        }

        fn name(&self) -> &str {
            "PassiveAgent"
        }
    }

    /// An agent that places a single order on construction.
    struct OneShotAgent {
        id: AgentId,
        order: Option<Order>,
    }

    impl OneShotAgent {
        fn new(id: AgentId, order: Order) -> Self {
            Self {
                id,
                order: Some(order),
            }
        }
    }

    impl Agent for OneShotAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _ctx: &StrategyContext<'_>) -> AgentAction {
            if let Some(order) = self.order.take() {
                AgentAction::single(order)
            } else {
                AgentAction::none()
            }
        }

        fn name(&self) -> &str {
            "OneShotAgent"
        }
    }

    #[test]
    fn test_empty_simulation_runs() {
        let mut sim = Simulation::with_defaults();

        // Run 1000 ticks with no agents
        let trades = sim.run(1000);

        assert!(trades.is_empty());
        assert_eq!(sim.tick(), 1000);
        assert_eq!(sim.stats().total_trades, 0);
    }

    #[test]
    fn test_passive_agents_no_trades() {
        let mut sim = Simulation::with_defaults();

        // Add some passive agents
        for i in 1..=10 {
            sim.add_agent(Box::new(PassiveAgent { id: AgentId(i) }));
        }

        let trades = sim.run(100);

        assert!(trades.is_empty());
        assert_eq!(sim.tick(), 100);
        assert_eq!(sim.agent_count(), 10);
    }

    #[test]
    fn test_orders_match() {
        let config = SimulationConfig::new("TEST").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Agent 1 places a sell order
        let sell_order = Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Sell,
            Price::from_float(100.0),
            Quantity(50),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), sell_order)));

        // Agent 2 places a buy order
        let buy_order = Order::limit(
            AgentId(2),
            "TEST",
            OrderSide::Buy,
            Price::from_float(100.0),
            Quantity(50),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), buy_order)));

        // First tick: both orders submitted, sell order processed first (rests),
        // then buy order matches against it
        let trades = sim.step();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, Quantity(50));
        assert_eq!(trades[0].price, Price::from_float(100.0));
        assert_eq!(trades[0].buyer_id, AgentId(2));
        assert_eq!(trades[0].seller_id, AgentId(1));

        assert_eq!(sim.stats().total_trades, 1);
    }

    #[test]
    fn test_market_data_updated() {
        let config = SimulationConfig::new("TEST").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Add an agent that places a limit order
        let sell_order = Order::limit(
            AgentId(1),
            "TEST",
            OrderSide::Sell,
            Price::from_float(105.0),
            Quantity(100),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), sell_order)));

        // Tick 0: order placed
        sim.step();

        // Market data should now show the ask
        let market = sim.build_market_data();
        assert_eq!(market.best_ask(), Some(Price::from_float(105.0)));
        assert_eq!(market.best_bid(), None);
    }

    #[test]
    fn test_recent_trades_tracked() {
        let config = SimulationConfig::new("TEST")
            .with_max_recent_trades(5)
            .with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        // Pre-populate book with sell orders
        for i in 1..=10 {
            let sell = Order::limit(
                AgentId(100 + i),
                "TEST",
                OrderSide::Sell,
                Price::from_float(100.0),
                Quantity(10),
            );
            // Process directly to set up book
            sim.process_order(sell);
        }

        // Add buyer agents
        for i in 1..=3 {
            let buy = Order::limit(
                AgentId(i),
                "TEST",
                OrderSide::Buy,
                Price::from_float(100.0),
                Quantity(10),
            );
            sim.add_agent(Box::new(OneShotAgent::new(AgentId(i), buy)));
        }

        // Run ticks to generate trades
        sim.run(3);

        // Should have at most 5 recent trades (configured max)
        assert!(sim.recent_trades().len() <= 5);
    }

    #[test]
    fn test_order_ids_unique() {
        let config = SimulationConfig::new("SIM").with_position_limits(false); // Disable for V0 test
        let mut sim = Simulation::new(config);

        let order1 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));
        let order2 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));

        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), order1)));
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), order2)));

        sim.step();

        assert_eq!(sim.stats().total_orders, 2);
        // Orders should have gotten unique IDs (1 and 2)
    }
}
