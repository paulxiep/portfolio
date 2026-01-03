//! Simulation runner implementing the tick-based event loop.
//!
//! The simulation holds the order book, agents, and coordinates the tick loop.

use std::collections::HashMap;

use agents::{Agent, AgentAction, MarketData};
use quant::{
    AgentRiskSnapshot, AgentRiskTracker, IndicatorCache, IndicatorEngine, IndicatorSnapshot,
};
use sim_core::{MatchingEngine, OrderBook};
use types::{AgentId, Candle, Order, OrderId, Price, Quantity, Tick, Timestamp, Trade};

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
}

/// The main simulation runner.
///
/// Coordinates the tick-based event loop:
/// 1. Build market data snapshot
/// 2. Call each agent's `on_tick` to get their orders
/// 3. Process orders through the matching engine
/// 4. Notify agents of fills
/// 5. Advance tick counter
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
        let book = OrderBook::new(&config.symbol);
        Self {
            config,
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
        let symbol = self.config.symbol.clone();

        // Compute all registered indicators
        let values = self.indicator_engine.compute_all(&self.candles);
        if !values.is_empty() {
            snapshot.insert(symbol, values);
        }

        Some(snapshot)
    }

    /// Build market data snapshot for agents.
    fn build_market_data(&mut self) -> MarketData {
        // Build indicator snapshot
        let indicators = self.build_indicator_snapshot();

        MarketData {
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

    /// Update candle with trade data.
    fn update_candles(&mut self, trades: &[Trade]) {
        for trade in trades {
            // Initialize candle builder if needed
            if self.current_candle.is_none() {
                self.current_candle = Some(CandleBuilder::new(
                    &self.config.symbol,
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

        // Build market data snapshot
        let market_data = self.build_market_data();

        // Collect all agent actions first (to avoid borrow issues)
        let actions: Vec<(AgentId, AgentAction)> = self
            .agents
            .iter_mut()
            .map(|agent| {
                let action = agent.on_tick(&market_data);
                (agent.id(), action)
            })
            .collect();

        // Process all orders
        let mut fill_notifications: Vec<(AgentId, Trade)> = Vec::new();

        for (_agent_id, action) in actions {
            for order in action.orders {
                let (trades, _is_resting) = self.process_order(order);

                // Record trades and prepare fill notifications
                for trade in trades {
                    self.stats.total_trades += 1;

                    // Notify both parties
                    fill_notifications.push((trade.buyer_id, trade.clone()));
                    fill_notifications.push((trade.seller_id, trade.clone()));

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
        for (agent_id, trade) in fill_notifications {
            if let Some(agent) = self.agents.iter_mut().find(|a| a.id() == agent_id) {
                agent.on_fill(&trade);
            }
        }

        // Update risk tracking with current equity values
        // Use last price or initial price for mark-to-market
        let mark_price = self.book.last_price().unwrap_or(self.config.initial_price);
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
    use types::{Order, OrderSide, Price, Quantity};

    /// A simple test agent that does nothing.
    struct PassiveAgent {
        id: AgentId,
    }

    impl Agent for PassiveAgent {
        fn id(&self) -> AgentId {
            self.id
        }

        fn on_tick(&mut self, _market: &MarketData) -> AgentAction {
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

        fn on_tick(&mut self, _market: &MarketData) -> AgentAction {
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
        let config = SimulationConfig::new("TEST");
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
        let config = SimulationConfig::new("TEST");
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
        let config = SimulationConfig::new("TEST").with_max_recent_trades(5);
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
        let mut sim = Simulation::with_defaults();

        let order1 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));
        let order2 = Order::market(AgentId(1), "SIM", OrderSide::Buy, Quantity(10));

        sim.add_agent(Box::new(OneShotAgent::new(AgentId(1), order1)));
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), order2)));

        sim.step();

        assert_eq!(sim.stats().total_orders, 2);
        // Orders should have gotten unique IDs (1 and 2)
    }
}
