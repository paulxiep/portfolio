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
//!
//! # Multi-Symbol Support (V2.3)
//!
//! The simulation supports multiple symbols via `Market` (HashMap<Symbol, OrderBook>).
//! Each symbol has independent candles, trades, and indicators.
//!
//! # Tiered Agent Architecture (V3.2)
//!
//! Agents are split into two tiers:
//! - **Tier 1**: Called every tick via `on_tick()` (market makers, technical traders)
//! - **Tier 2**: Reactive agents woken only when conditions trigger (via WakeConditionIndex)
//!
//! This reduces per-tick overhead from O(n) to O(k) where k << n triggered agents.
//!
//! # Parallel Execution & Batch Auction (V3.5)
//!
//! With the `parallel` feature enabled:
//! - T1 and triggered T2 agents execute `on_tick()` in parallel via rayon
//! - Orders are grouped by symbol and processed via **batch auction**
//! - Each symbol's auction runs independently (fully parallel across symbols)
//!
//! Batch auction semantics:
//! 1. **Collection phase**: All agents run `on_tick()` in parallel, collecting orders
//! 2. **Auction phase**: Per-symbol clearing price computed, all crossing orders matched
//!
//! This differs from continuous matching: all agents see the same market state and
//! compete in a single auction per tick, rather than sequential price-time priority.

use std::collections::HashMap;

use agents::{
    Agent, AgentAction, BACKGROUND_POOL_ID, BackgroundAgentPool, BorrowLedger, PoolContext,
    PositionValidator, StrategyContext, WakeConditionIndex,
};
use parking_lot::Mutex;
use quant::{
    AgentRiskSnapshot, AgentRiskTracker, IndicatorCache, IndicatorEngine, IndicatorSnapshot,
};
use rand::seq::SliceRandom;
use sim_core::{Market, MarketView, MatchingEngine, OrderBook, run_parallel_auctions};
use types::{
    AgentId, Candle, Order, OrderId, Price, Quantity, RiskViolation, Symbol, Tick, Timestamp, Trade,
};

use crate::config::SimulationConfig;
use crate::parallel;

/// Agent action with captured state for order validation.
/// (agent_id, action, per-symbol positions, cash, is_market_maker)
type AgentActionWithState = (
    AgentId,
    AgentAction,
    HashMap<Symbol, i64>,
    types::Cash,
    bool,
);

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

    /// Agents called this tick (V3.2 debug).
    pub agents_called_this_tick: usize,

    /// T2 agents triggered this tick (V3.2 debug).
    pub t2_triggered_this_tick: usize,

    /// T3 background pool orders generated this tick (V3.4).
    pub t3_orders_this_tick: usize,
}

/// The main simulation runner.
///
/// Coordinates the tick-based event loop:
/// 1. Build market data snapshot
/// 2. Call each agent's `on_tick` to get their orders
/// 3. Validate orders against position limits (V2.1)
/// 4. Run batch auction per symbol (V3.5 - parallel across symbols)
/// 5. Update borrow ledger on short trades (V2.1)
/// 6. Notify agents of fills
/// 7. Advance tick counter
///
/// # V3.5 Parallel Execution & Batch Auction
///
/// Agents are wrapped in `Mutex` to enable parallel `on_tick()` execution.
/// Orders are processed via batch auction (single clearing price per symbol),
/// enabling full parallelism across symbols.
pub struct Simulation {
    /// Configuration for this simulation.
    config: SimulationConfig,

    /// Multi-symbol market container (V2.3).
    market: Market,

    /// The matching engine (used for T3 background pool orders).
    engine: MatchingEngine,

    /// Trading agents wrapped in Mutex for parallel access (V3.5).
    /// Each agent is accessed by exactly one thread during parallel collection.
    agents: Vec<Mutex<Box<dyn Agent>>>,

    /// Indices of Tier 1 agents (called every tick).
    t1_indices: Vec<usize>,

    /// Indices of Tier 2 agents (called only when triggered).
    t2_indices: Vec<usize>,

    /// Map from AgentId to index in agents vec (for T2 lookup).
    agent_id_to_index: HashMap<AgentId, usize>,

    /// Current tick.
    tick: Tick,

    /// Current timestamp.
    timestamp: Timestamp,

    /// Recent trades per symbol (V2.3).
    recent_trades: HashMap<Symbol, Vec<Trade>>,

    /// Counter for generating unique order IDs.
    next_order_id: u64,

    /// Simulation statistics.
    stats: SimulationStats,

    /// Historical candles per symbol (V2.3).
    candles: HashMap<Symbol, Vec<Candle>>,

    /// Current candle being built per symbol (V2.3).
    current_candles: HashMap<Symbol, CandleBuilder>,

    /// Indicator engine for computing technical indicators.
    /// Note: In multi-symbol, we compute indicators per-symbol using the same engine.
    indicator_engine: IndicatorEngine,

    /// Indicator cache for current tick (reserved for future per-tick caching).
    #[allow(dead_code)]
    indicator_cache: IndicatorCache,

    /// Per-agent risk tracking.
    risk_tracker: AgentRiskTracker,

    /// Position validator for order validation (V2.1).
    /// Note: Currently uses primary symbol config. Future: per-symbol validators.
    position_validator: PositionValidator,

    /// Borrow ledger for short-selling tracking (V2.1).
    borrow_ledger: BorrowLedger,

    /// Cache of total shares held per symbol (sum of all long positions) for validation.
    /// Updated after each trade.
    total_shares_held: HashMap<Symbol, Quantity>,

    /// News event generator (V2.4).
    news_generator: news::NewsGenerator,

    /// Currently active news events (V2.4).
    active_events: Vec<news::NewsEvent>,

    /// Symbol fundamentals for fair value calculation (V2.4).
    fundamentals: news::SymbolFundamentals,

    /// Wake condition index for Tier 2 reactive agents (V3.2).
    wake_index: WakeConditionIndex,

    /// Optional Tier 3 background pool for statistical order generation (V3.4).
    background_pool: Option<BackgroundAgentPool>,
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
        // Initialize multi-symbol market
        let mut market = Market::new();
        let mut borrow_ledger = BorrowLedger::new();
        let mut total_shares_held = HashMap::new();
        let mut candles = HashMap::new();
        let mut recent_trades = HashMap::new();

        // Initialize fundamentals for each symbol (V2.4)
        let mut fundamentals = news::SymbolFundamentals::new(news::MacroEnvironment::default());
        let mut sector_model = news::SectorModel::new();
        let symbols: Vec<_> = config
            .get_symbol_configs()
            .iter()
            .map(|c| c.symbol.clone())
            .collect();

        // Initialize each symbol
        for symbol_config in config.get_symbol_configs() {
            market.add_symbol(&symbol_config.symbol);
            borrow_ledger.init_symbol(&symbol_config.symbol, symbol_config.borrow_pool_size());
            total_shares_held.insert(symbol_config.symbol.clone(), Quantity::ZERO);
            candles.insert(symbol_config.symbol.clone(), Vec::new());
            recent_trades.insert(symbol_config.symbol.clone(), Vec::new());

            // Initialize default fundamentals based on initial price (V2.4)
            // EPS derived as price / 20 (P/E of 20), 5% growth, 40% payout
            let eps = Price::from_float(symbol_config.initial_price.to_float() / 20.0);
            fundamentals.insert(
                &symbol_config.symbol,
                news::Fundamentals::new(eps, 0.05, 0.40),
            );

            // Add symbol to sector model (V2.4)
            sector_model.add(&symbol_config.symbol, symbol_config.sector);
        }

        // Initialize news generator (V2.4)
        let news_generator =
            news::NewsGenerator::new(config.news.clone(), symbols, sector_model, config.seed);

        // Initialize position validator with primary symbol config
        // TODO: In future, support per-symbol position limits
        let primary_config = config
            .get_symbol_configs()
            .first()
            .cloned()
            .unwrap_or_default();
        let position_validator =
            PositionValidator::new(primary_config, config.short_selling.clone());

        // V3.2: Initialize wake index for T2 agent conditions
        let wake_index = WakeConditionIndex::new();

        Self {
            market,
            engine: MatchingEngine::new(),
            agents: Vec::new(),
            t1_indices: Vec::new(),
            t2_indices: Vec::new(),
            agent_id_to_index: HashMap::new(),
            tick: 0,
            timestamp: 0,
            recent_trades,
            next_order_id: 1,
            stats: SimulationStats::default(),
            candles,
            current_candles: HashMap::new(),
            indicator_engine: IndicatorEngine::with_common_indicators(),
            indicator_cache: IndicatorCache::new(),
            risk_tracker: AgentRiskTracker::with_defaults(),
            position_validator,
            borrow_ledger,
            total_shares_held,
            news_generator,
            active_events: Vec::new(),
            fundamentals,
            wake_index,
            background_pool: None,
            config,
        }
    }

    /// Create a simulation with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SimulationConfig::default())
    }

    /// Set the Tier 3 background pool for statistical order generation (V3.4).
    ///
    /// The pool generates orders each tick based on statistical distributions,
    /// simulating 90k+ background agents without individual instances.
    pub fn set_background_pool(&mut self, pool: BackgroundAgentPool) {
        self.background_pool = Some(pool);
    }

    /// Get a reference to the background pool (if configured).
    pub fn background_pool(&self) -> Option<&BackgroundAgentPool> {
        self.background_pool.as_ref()
    }

    /// Get a mutable reference to the background pool (if configured).
    pub fn background_pool_mut(&mut self) -> Option<&mut BackgroundAgentPool> {
        self.background_pool.as_mut()
    }

    /// Add an agent to the simulation.
    ///
    /// For reactive (T2) agents, registers their initial wake conditions.
    pub fn add_agent(&mut self, agent: Box<dyn Agent>) {
        let agent_id = agent.id();
        let is_reactive = agent.is_reactive();
        let index = self.agents.len();

        // Track index by tier
        if is_reactive {
            self.t2_indices.push(index);
            // Register initial wake conditions for reactive agents
            // (imperative: wake_index.register is side-effectful)
            for condition in agent.initial_wake_conditions(self.tick) {
                self.wake_index.register(agent_id, condition);
            }
        } else {
            self.t1_indices.push(index);
        }

        // Map agent ID to index for triggered T2 lookup
        self.agent_id_to_index.insert(agent_id, index);

        // V3.5: Wrap agent in Mutex for parallel access
        self.agents.push(Mutex::new(agent));
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

    /// Get a reference to the primary (first) symbol's order book.
    /// For multi-symbol access, use `market()` and `get_book()`.
    pub fn book(&self) -> &OrderBook {
        let symbol = self.config.symbol();
        self.market
            .get_book(&symbol.to_string())
            .expect("Primary symbol book should exist")
    }

    /// Get a reference to a specific symbol's order book.
    pub fn get_book(&self, symbol: &Symbol) -> Option<&OrderBook> {
        self.market.get_book(symbol)
    }

    /// Get a reference to the multi-symbol market (V2.3).
    pub fn market(&self) -> &Market {
        &self.market
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

    /// Get a reference to the simulation configuration.
    pub fn config(&self) -> &SimulationConfig {
        &self.config
    }

    /// Get the total shares currently held for the primary symbol.
    pub fn total_shares_held(&self) -> Quantity {
        let symbol = self.config.symbol().to_string();
        self.total_shares_held
            .get(&symbol)
            .copied()
            .unwrap_or(Quantity::ZERO)
    }

    /// Get total shares held for a specific symbol.
    pub fn total_shares_held_for(&self, symbol: &Symbol) -> Quantity {
        self.total_shares_held
            .get(symbol)
            .copied()
            .unwrap_or(Quantity::ZERO)
    }

    /// Get agent summaries for display (V3.1: per-symbol positions).
    /// Returns a vec of (name, positions_map, cash, total_pnl) tuples.
    /// V3.5: Returns owned String for name due to Mutex wrapper.
    pub fn agent_summaries(&self) -> Vec<(String, HashMap<Symbol, i64>, types::Cash, types::Cash)> {
        // Get current prices for all symbols
        let prices: HashMap<Symbol, types::Price> = self
            .market
            .symbols()
            .filter_map(|sym| {
                self.market
                    .get_book(sym)
                    .and_then(|b| b.last_price())
                    .map(|p| (sym.clone(), p))
            })
            .collect();

        self.agents
            .iter()
            .map(|agent_mutex| {
                let a = agent_mutex.lock();
                let positions: HashMap<Symbol, i64> = a
                    .positions()
                    .iter()
                    .map(|(sym, entry)| (sym.clone(), entry.quantity))
                    .collect();
                let total_pnl = a.state().total_pnl(&prices);
                (a.name().to_owned(), positions, a.cash(), total_pnl)
            })
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

    /// Get recent trades for the primary symbol.
    pub fn recent_trades(&self) -> &[Trade] {
        let symbol = self.config.symbol().to_string();
        self.recent_trades
            .get(&symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get recent trades for a specific symbol.
    pub fn recent_trades_for(&self, symbol: &Symbol) -> &[Trade] {
        self.recent_trades
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all recent trades across all symbols.
    pub fn all_recent_trades(&self) -> &HashMap<Symbol, Vec<Trade>> {
        &self.recent_trades
    }

    /// Get historical candles for the primary symbol.
    pub fn candles(&self) -> &[Candle] {
        let symbol = self.config.symbol().to_string();
        self.candles
            .get(&symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get historical candles for a specific symbol.
    pub fn candles_for(&self, symbol: &Symbol) -> &[Candle] {
        self.candles
            .get(symbol)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all candles across all symbols.
    pub fn all_candles(&self) -> &HashMap<Symbol, Vec<Candle>> {
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

    /// Build indicator snapshot for current tick (all symbols).
    fn build_indicator_snapshot(&mut self) -> IndicatorSnapshot {
        let indicators = self
            .candles
            .iter()
            .filter(|(_, symbol_candles)| !symbol_candles.is_empty())
            .filter_map(|(symbol, symbol_candles)| {
                let values = self.indicator_engine.compute_all(symbol_candles);
                (!values.is_empty()).then(|| (symbol.clone(), values))
            })
            .collect();

        IndicatorSnapshot::from_map(self.tick, indicators)
    }

    /// Build candles map for StrategyContext (just return reference).
    fn build_candles_map(&self) -> HashMap<Symbol, Vec<Candle>> {
        self.candles.clone()
    }

    /// Build trades map for StrategyContext (just return reference).
    fn build_trades_map(&self) -> HashMap<Symbol, Vec<Trade>> {
        self.recent_trades.clone()
    }

    /// Update candle with trade data for the trade's symbol.
    fn update_candles(&mut self, trades: &[Trade]) {
        for trade in trades {
            let symbol = &trade.symbol;

            // Initialize candle builder if needed
            if !self.current_candles.contains_key(symbol) {
                self.current_candles.insert(
                    symbol.clone(),
                    CandleBuilder::new(symbol, trade.price, self.tick),
                );
            }

            // Update current candle for this symbol
            if let Some(builder) = self.current_candles.get_mut(symbol) {
                builder.update(trade);
            }
        }

        // Check if we should finalize candles (every candle_interval ticks)
        if self.tick > 0 && self.tick.is_multiple_of(self.config.candle_interval) {
            // Finalize all current candles
            let builders: Vec<(Symbol, CandleBuilder)> = self.current_candles.drain().collect();
            for (symbol, builder) in builders {
                let candle = builder.finalize(self.tick, self.timestamp);
                let symbol_candles = self.candles.entry(symbol).or_default();
                symbol_candles.push(candle);

                // Limit candle history per symbol
                if symbol_candles.len() > self.config.max_candles {
                    symbol_candles.remove(0);
                }
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

        // Use the order's symbol for shares outstanding check (V3.1 fix)
        let total_held = self
            .total_shares_held
            .get(&order.symbol)
            .copied()
            .unwrap_or(Quantity::ZERO);

        self.position_validator.validate_order(
            order,
            agent_position,
            agent_cash,
            &self.borrow_ledger,
            total_held,
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

    /// Update total shares held after a trade (per-symbol).
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

        let symbol_shares = self
            .total_shares_held
            .entry(trade.symbol.clone())
            .or_insert(Quantity::ZERO);
        if delta > 0 {
            *symbol_shares += Quantity(delta as u64);
        } else {
            *symbol_shares = symbol_shares.saturating_sub(Quantity((-delta) as u64));
        }
    }

    /// Process a single order through the matching engine (for T3 background pool).
    ///
    /// Returns the trades that occurred and optionally the order if it's resting on the book
    /// (with its assigned OrderId).
    ///
    /// Note: Main agent orders now go through batch auction. This is retained for
    /// T3 background pool orders which need continuous matching semantics.
    fn process_order(&mut self, mut order: Order) -> (Vec<Trade>, Option<Order>) {
        // Assign order ID and timestamp
        order.id = self.next_order_id();
        order.timestamp = self.timestamp;

        self.stats.total_orders += 1;

        // Get the order book for this symbol
        let Some(book) = self.market.get_book_mut(&order.symbol) else {
            // Unknown symbol - reject order silently
            self.stats.rejected_orders += 1;
            return (Vec::new(), None);
        };

        // Try to match the order
        let result = self
            .engine
            .match_order(book, &mut order, self.timestamp, self.tick);

        let mut resting_order = None;

        // If there's remaining quantity for a limit order, add to book temporarily
        // (will be cleared at end of tick - all orders are IOC within tick)
        if !result.remaining_quantity.is_zero() && order.limit_price().is_some() {
            order.remaining_quantity = result.remaining_quantity;
            let order_clone = order.clone();
            if let Some(book) = self.market.get_book_mut(&order.symbol)
                && book.add_order(order).is_ok()
            {
                resting_order = Some(order_clone);
                self.stats.resting_orders += 1;
            }
        }

        if result.has_trades() {
            self.stats.filled_orders += 1;
        }

        (result.trades, resting_order)
    }

    // =========================================================================
    // V3.5: Parallel Agent Collection (using parallel:: helpers)
    // =========================================================================

    /// Collect agent actions from the given indices.
    ///
    /// Uses `parallel::map_indices` which automatically parallelizes when
    /// the `parallel` feature is enabled.
    fn collect_agent_actions(
        &self,
        indices: &[usize],
        ctx: &StrategyContext<'_>,
    ) -> Vec<AgentActionWithState> {
        parallel::map_indices(indices, |i| {
            let mut agent = self.agents[i].lock();
            let agent_id = agent.id();
            let action = agent.on_tick(ctx);
            let positions: HashMap<Symbol, i64> = agent
                .positions()
                .iter()
                .map(|(sym, entry)| (sym.clone(), entry.quantity))
                .collect();
            let cash = agent.cash();
            let is_mm = agent.is_market_maker();
            (agent_id, action, positions, cash, is_mm)
        })
    }

    /// Build position cache for all agents (for counterparty lookup during trade processing).
    fn build_position_cache(&self) -> HashMap<AgentId, HashMap<Symbol, i64>> {
        parallel::map_mutex_slice_ref(&self.agents, |agent| {
            let positions: HashMap<Symbol, i64> = agent
                .positions()
                .iter()
                .map(|(sym, entry)| (sym.clone(), entry.quantity))
                .collect();
            (agent.id(), positions)
        })
        .into_iter()
        .collect()
    }

    /// Collect current prices for all symbols (for wake condition checking).
    fn collect_current_prices(&self) -> Vec<(Symbol, Price)> {
        self.config
            .get_symbol_configs()
            .iter()
            .map(|sc| {
                let price = self
                    .market
                    .last_price(&sc.symbol)
                    .unwrap_or(sc.initial_price);
                (sc.symbol.clone(), price)
            })
            .collect()
    }

    /// Determine which agent indices should be called this tick.
    ///
    /// T1 agents are always called; T2 agents only when their conditions trigger.
    /// Returns (indices_to_call, triggered_t2_map).
    fn compute_agents_to_call(
        &mut self,
        current_prices: &[(Symbol, Price)],
    ) -> (Vec<usize>, HashMap<AgentId, smallvec::SmallVec<[agents::WakeCondition; 2]>>) {
        // Get news symbols for condition checking
        let news_symbols: Vec<Symbol> = self
            .active_events
            .iter()
            .filter_map(|e| e.symbol().cloned())
            .collect();

        // Collect triggered T2 agents from wake index
        let triggered_t2 =
            self.wake_index
                .collect_triggered(self.tick, current_prices, &news_symbols);

        // Track triggered count for stats
        self.stats.t2_triggered_this_tick = triggered_t2.len();

        // Remove triggered PriceCross conditions immediately to prevent re-firing
        for (agent_id, conditions) in &triggered_t2 {
            for condition in conditions {
                if matches!(condition, agents::WakeCondition::PriceCross { .. }) {
                    self.wake_index.unregister(*agent_id, condition);
                }
            }
        }

        // Build list: T1 always, T2 only if triggered
        let mut indices_to_call: Vec<usize> = self.t1_indices.clone();
        indices_to_call.extend(
            triggered_t2
                .keys()
                .filter_map(|agent_id| self.agent_id_to_index.get(agent_id).copied()),
        );

        // Randomize order to avoid systematic bias
        indices_to_call.shuffle(&mut rand::thread_rng());

        (indices_to_call, triggered_t2)
    }

    /// Validate and group orders by symbol for batch auction.
    ///
    /// Processes cancellations and validates new orders against position limits.
    fn collect_orders_for_auction(
        &mut self,
        actions_with_state: Vec<AgentActionWithState>,
    ) -> HashMap<Symbol, Vec<Order>> {
        let mut orders_by_symbol: HashMap<Symbol, Vec<Order>> = HashMap::new();

        for (agent_id, action, agent_positions, agent_cash, is_market_maker) in actions_with_state {
            // Process cancellations first
            for order_id in action.cancellations {
                for book in self.market.books_mut() {
                    let _ = book.cancel_order(order_id);
                }
            }

            // Validate and collect orders
            for mut order in action.orders {
                let symbol_position = agent_positions.get(&order.symbol).copied().unwrap_or(0);

                if let Err(violation) =
                    self.validate_order(&order, symbol_position, agent_cash, is_market_maker)
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
                    continue;
                }

                order.id = self.next_order_id();
                order.timestamp = self.timestamp;
                self.stats.total_orders += 1;

                orders_by_symbol
                    .entry(order.symbol.clone())
                    .or_default()
                    .push(order);
            }
        }

        orders_by_symbol
    }

    /// Build reference prices for batch auction clearing.
    ///
    /// Priority: fair_value (fundamentals) > last_price (order book).
    fn build_reference_prices<'a>(&self, symbols: impl Iterator<Item = &'a Symbol>) -> HashMap<String, Price> {
        symbols
            .filter_map(|symbol| {
                self.fundamentals
                    .fair_value(symbol)
                    .or_else(|| {
                        self.market
                            .get_book(symbol)
                            .and_then(|book| book.last_price())
                    })
                    .map(|price| (symbol.clone(), price))
            })
            .collect()
    }

    /// Process batch auction results into trades.
    ///
    /// Updates stats, borrow ledger, share tracking, and collects fill notifications.
    fn process_auction_results(
        &mut self,
        auction_results: HashMap<Symbol, sim_core::BatchAuctionResult>,
        position_cache: &HashMap<AgentId, HashMap<Symbol, i64>>,
    ) -> (Vec<Trade>, Vec<(AgentId, Trade, i64)>) {
        let mut tick_trades = Vec::new();
        let mut fill_notifications = Vec::new();

        for (symbol, result) in auction_results {
            self.stats.filled_orders += result.filled_orders.len() as u64;
            self.stats.total_trades += result.trades.len() as u64;

            // Update last price in order book
            if let Some(clearing_price) = result.clearing_price
                && let Some(book) = self.market.get_book_mut(&symbol)
            {
                book.set_last_price(clearing_price);
            }

            for trade in result.trades {
                let buyer_pos_before = position_cache
                    .get(&trade.buyer_id)
                    .and_then(|positions| positions.get(&trade.symbol).copied())
                    .unwrap_or(0);

                let seller_pos_before = position_cache
                    .get(&trade.seller_id)
                    .and_then(|positions| positions.get(&trade.symbol).copied())
                    .unwrap_or(0);

                self.update_borrow_ledger(&trade, seller_pos_before, buyer_pos_before);
                self.update_total_shares_held(&trade, seller_pos_before, buyer_pos_before);

                fill_notifications.push((trade.buyer_id, trade.clone(), buyer_pos_before));
                fill_notifications.push((trade.seller_id, trade.clone(), seller_pos_before));

                tick_trades.push(trade);
            }
        }

        (tick_trades, fill_notifications)
    }

    /// Process Tier 3 background pool orders.
    ///
    /// Generates orders from the statistical pool and matches them via continuous matching.
    fn process_background_pool(&mut self, tick_trades: &mut Vec<Trade>) {
        let Some(_) = self.background_pool.as_ref() else {
            return;
        };

        // Build mid prices map
        let mid_prices: HashMap<Symbol, Price> = self
            .config
            .get_symbol_configs()
            .iter()
            .filter_map(|sc| {
                self.market
                    .mid_price(&sc.symbol)
                    .map(|p| (sc.symbol.clone(), p))
            })
            .collect();

        // Build symbol->sector mapping
        let symbol_sectors: HashMap<Symbol, types::Sector> = self
            .config
            .get_symbol_configs()
            .iter()
            .map(|sc| (sc.symbol.clone(), sc.sector))
            .collect();

        let pool_ctx = PoolContext {
            tick: self.tick,
            mid_prices: &mid_prices,
            active_events: &self.active_events,
            symbol_sectors: &symbol_sectors,
        };

        let pool = self.background_pool.as_mut().unwrap();
        let t3_orders = pool.generate(&pool_ctx);
        self.stats.t3_orders_this_tick = t3_orders.len();

        // Process each T3 order via continuous matching
        let t3_trades: Vec<_> = t3_orders
            .into_iter()
            .flat_map(|order| {
                let (trades, _resting) = self.process_order(order);
                trades
            })
            .collect();

        // Update pool accounting and notify counterparty agents
        for trade in t3_trades {
            let pool = self.background_pool.as_mut().unwrap();
            if trade.buyer_id == BACKGROUND_POOL_ID {
                pool.accounting_mut().record_trade_as_buyer(
                    &trade.symbol,
                    trade.price,
                    trade.quantity,
                );
            } else if trade.seller_id == BACKGROUND_POOL_ID {
                pool.accounting_mut().record_trade_as_seller(
                    &trade.symbol,
                    trade.price,
                    trade.quantity,
                );
            }

            // Notify the counterparty agent (not the pool)
            let other_agent_id = if trade.buyer_id == BACKGROUND_POOL_ID {
                trade.seller_id
            } else {
                trade.buyer_id
            };

            if let Some(&idx) = self.agent_id_to_index.get(&other_agent_id) {
                self.agents[idx].lock().on_fill(&trade);
            }

            tick_trades.push(trade);
        }
    }

    /// Update recent trades storage with new trades.
    fn update_recent_trades(&mut self, tick_trades: &[Trade]) {
        // Add trades (newest first)
        tick_trades.iter().rev().for_each(|trade| {
            let symbol_trades = self.recent_trades.entry(trade.symbol.clone()).or_default();
            symbol_trades.insert(0, trade.clone());
        });

        // Trim to max
        self.recent_trades.values_mut().for_each(|symbol_trades| {
            if symbol_trades.len() > self.config.max_recent_trades {
                symbol_trades.truncate(self.config.max_recent_trades);
            }
        });
    }

    /// Notify agents of fills and update wake conditions.
    fn process_fill_notifications(&mut self, fill_notifications: Vec<(AgentId, Trade, i64)>) {
        let condition_updates =
            parallel::filter_map_slice(&fill_notifications, |(agent_id, trade, pos_before)| {
                self.agent_id_to_index.get(agent_id).and_then(|&idx| {
                    let mut agent = self.agents[idx].lock();
                    agent.on_fill(trade);
                    agent
                        .is_reactive()
                        .then(|| agent.post_fill_condition_update(*pos_before))
                        .flatten()
                })
            });

        self.wake_index.apply_updates(condition_updates);
    }

    /// Restore wake conditions for triggered T2 agents.
    ///
    /// After triggering, agents may need new conditions (e.g., exit conditions after entry).
    fn restore_t2_wake_conditions(
        &mut self,
        triggered_t2: &HashMap<AgentId, smallvec::SmallVec<[agents::WakeCondition; 2]>>,
    ) {
        let triggered_keys: Vec<_> = triggered_t2.keys().copied().collect();
        let t2_conditions = parallel::filter_map_slice(&triggered_keys, |agent_id| {
            self.agent_id_to_index.get(agent_id).and_then(|&idx| {
                let agent = self.agents[idx].lock();
                agent
                    .is_reactive()
                    .then(|| (*agent_id, agent.current_wake_conditions().to_vec()))
            })
        });

        for (agent_id, conditions) in t2_conditions {
            for condition in conditions {
                self.wake_index.register(agent_id, condition);
            }
        }
    }

    /// Update risk tracking with current equity values.
    fn update_risk_tracking(&mut self) {
        let prices: HashMap<Symbol, Price> = self
            .config
            .get_symbol_configs()
            .iter()
            .map(|sc| {
                let price = self
                    .market
                    .last_price(&sc.symbol)
                    .unwrap_or(sc.initial_price);
                (sc.symbol.clone(), price)
            })
            .collect();

        let equities = parallel::map_mutex_slice_ref(&self.agents, |agent| {
            (agent.id(), agent.equity(&prices).to_float())
        });

        for (agent_id, equity) in equities {
            self.risk_tracker.record_equity(agent_id, equity);
        }
    }

    /// Clear order books and advance tick counter.
    fn finalize_tick(&mut self) {
        // Clear all order books (orders expire after tick)
        for book in self.market.books_mut() {
            book.clear();
        }

        // Advance time
        self.tick += 1;
        self.timestamp += 1;
        self.stats.tick = self.tick;
    }

    /// Advance the simulation by one tick.
    ///
    /// Returns trades that occurred during this tick.
    ///
    /// # Phases
    ///
    /// 0. Process news events (updates fundamentals and active events)
    /// 1. Determine which agents to call (T1 always, T2 when triggered)
    /// 2. Build strategy context for agents
    /// 3. Collect agent actions (orders and cancellations)
    /// 4. Run batch auction for agent orders
    /// 5. Process Tier 3 background pool orders
    /// 6. Update market data (recent trades, candles)
    /// 7. Notify agents of fills and update wake conditions
    /// 8. Update risk tracking
    /// 9. Finalize tick (clear books, advance time)
    pub fn step(&mut self) -> Vec<Trade> {
        // Phase 0: Process news events
        self.process_news_events();

        // Phase 1: Determine which agents to call (before building ctx to avoid borrow conflict)
        // This mutates wake_index, so must happen before taking immutable refs to self
        let current_prices = self.collect_current_prices();
        let (indices_to_call, triggered_t2) = self.compute_agents_to_call(&current_prices);

        // Phase 2: Build strategy context for agents
        let candles_map = self.build_candles_map();
        let trades_map = self.build_trades_map();
        let indicators = self.build_indicator_snapshot();
        let ctx = StrategyContext::new(
            self.tick,
            self.timestamp,
            &self.market,
            &candles_map,
            &indicators,
            &trades_map,
            &self.active_events,
            &self.fundamentals,
        );

        // Phase 3: Collect agent actions
        let actions_with_state = self.collect_agent_actions(&indices_to_call, &ctx);
        self.stats.agents_called_this_tick = actions_with_state.len();

        // Phase 4: Run batch auction for agent orders
        let position_cache = self.build_position_cache();
        let orders_by_symbol = self.collect_orders_for_auction(actions_with_state);
        let reference_prices = self.build_reference_prices(orders_by_symbol.keys());
        let auction_results = run_parallel_auctions(
            orders_by_symbol,
            &reference_prices,
            self.timestamp,
            self.tick,
            self.next_order_id,
        );
        let (mut tick_trades, fill_notifications) =
            self.process_auction_results(auction_results, &position_cache);

        // Phase 5: Process Tier 3 background pool orders
        self.process_background_pool(&mut tick_trades);

        // Phase 6: Update market data
        self.update_recent_trades(&tick_trades);
        self.update_candles(&tick_trades);

        // Phase 7: Notify agents and update wake conditions
        self.process_fill_notifications(fill_notifications);
        self.restore_t2_wake_conditions(&triggered_t2);

        // Phase 8: Update risk tracking
        self.update_risk_tracking();

        // Phase 9: Finalize tick
        self.finalize_tick();

        tick_trades
    }

    /// Run the simulation for a given number of ticks.
    ///
    /// Returns total trades across all ticks.
    pub fn run(&mut self, ticks: u64) -> Vec<Trade> {
        (0..ticks).fold(Vec::new(), |mut all_trades, _| {
            all_trades.extend(self.step());
            all_trades
        })
    }

    // =========================================================================
    // News & Fundamentals (V2.4)
    // =========================================================================

    /// Process news events for the current tick.
    ///
    /// This method:
    /// 1. Generates new events from the news generator
    /// 2. Applies permanent fundamental changes (earnings, guidance, rate decisions)
    /// 3. Prunes expired events from the active list
    fn process_news_events(&mut self) {
        // Generate new events
        let new_events = self.news_generator.tick(self.tick);

        // Apply permanent fundamental changes before adding to active list
        for event in &new_events {
            if event.is_permanent() {
                self.apply_fundamental_event(event);
            }
        }

        // Add new events to active list
        self.active_events.extend(new_events);

        // Prune expired events
        self.active_events.retain(|e| e.is_active(self.tick));
    }

    /// Apply a permanent fundamental event.
    fn apply_fundamental_event(&mut self, event: &news::NewsEvent) {
        match &event.event {
            news::FundamentalEvent::EarningsSurprise {
                symbol,
                surprise_pct,
            } => {
                if let Some(fundamentals) = self.fundamentals.get_mut(symbol) {
                    fundamentals.apply_earnings_surprise(*surprise_pct);
                    if self.config.verbose {
                        eprintln!(
                            "[Tick {}] Earnings surprise for {}: {:.1}% → EPS now ${:.2}",
                            self.tick,
                            symbol,
                            surprise_pct * 100.0,
                            fundamentals.eps.to_float()
                        );
                    }
                }
            }
            news::FundamentalEvent::GuidanceChange { symbol, new_growth } => {
                if let Some(fundamentals) = self.fundamentals.get_mut(symbol) {
                    fundamentals.apply_guidance_change(*new_growth);
                    if self.config.verbose {
                        eprintln!(
                            "[Tick {}] Guidance change for {}: growth now {:.1}%",
                            self.tick,
                            symbol,
                            new_growth * 100.0
                        );
                    }
                }
            }
            news::FundamentalEvent::RateDecision { new_rate } => {
                self.fundamentals.macro_env.apply_rate_decision(*new_rate);
                if self.config.verbose {
                    eprintln!(
                        "[Tick {}] Rate decision: risk-free rate now {:.2}%",
                        self.tick,
                        new_rate * 100.0
                    );
                }
            }
            news::FundamentalEvent::SectorNews { .. } => {
                // Sector news is temporary sentiment, not a permanent fundamental change
            }
        }
    }

    /// Get the currently active news events.
    pub fn active_events(&self) -> &[news::NewsEvent] {
        &self.active_events
    }

    /// Get the symbol fundamentals.
    pub fn fundamentals(&self) -> &news::SymbolFundamentals {
        &self.fundamentals
    }

    /// Get fair value for a symbol.
    pub fn fair_value(&self, symbol: &types::Symbol) -> Option<Price> {
        self.fundamentals.fair_value(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agents::{AgentState, StrategyContext};
    use types::{Cash, Order, OrderSide, Price, Quantity};

    /// A simple test agent that does nothing.
    struct PassiveAgent {
        id: AgentId,
        state: AgentState,
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

        fn state(&self) -> &AgentState {
            &self.state
        }
    }

    /// An agent that places a single order on construction.
    struct OneShotAgent {
        id: AgentId,
        state: AgentState,
        order: Option<Order>,
    }

    impl OneShotAgent {
        fn new(id: AgentId, order: Order) -> Self {
            let symbol = order.symbol.clone();
            Self {
                id,
                state: AgentState::new(Cash::from_float(100_000.0), &[&symbol]),
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

        fn state(&self) -> &AgentState {
            &self.state
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
            sim.add_agent(Box::new(PassiveAgent {
                id: AgentId(i),
                state: AgentState::default(),
            }));
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

        // Agent 2 places a buy order at same price
        let buy_order = Order::limit(
            AgentId(2),
            "TEST",
            OrderSide::Buy,
            Price::from_float(100.0),
            Quantity(50),
        );
        sim.add_agent(Box::new(OneShotAgent::new(AgentId(2), buy_order)));

        // First tick: both orders submitted, one rests briefly, other matches against it
        let trades = sim.step();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, Quantity(50));
        assert_eq!(trades[0].price, Price::from_float(100.0));

        assert_eq!(sim.stats().total_trades, 1);

        // After tick, book should be cleared
        assert!(sim.book().is_empty());
    }

    #[test]
    fn test_book_state_after_order() {
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

        // Tick 0: order placed, then book cleared at end of tick
        sim.step();

        // Book should be empty after tick (orders expire)
        let book = sim.book();
        assert_eq!(book.best_ask_price(), None);
        assert_eq!(book.best_bid_price(), None);
        assert!(book.is_empty());
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
