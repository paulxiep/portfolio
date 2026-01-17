//! Recording hook for ML training data capture.
//!
//! V5.4: Implements SimulationHook to capture features, actions, and outcomes
//! at each tick, writing to separate Parquet files for market and agent data.
//!
//! ## Output Files
//!
//! - `{name}_market.parquet`: Market features (1 row per tick per symbol)
//! - `{name}_agents.parquet`: Agent features + actions + rewards (1 row per agent per tick)

use std::collections::HashMap;

use parking_lot::Mutex;
use simulation::{HookContext, SimulationHook, SimulationStats};
use types::{AgentId, Cash, Order, OrderSide, OrderType, Symbol, Tick, Trade};

use crate::comprehensive_features::{AgentFeatureContext, AgentFeatures, MarketFeatures};
use crate::parquet_writer::{AgentRecord, DualParquetWriter, MarketRecord, ParquetWriterError};
use crate::price_history::PriceHistory;

/// Configuration for the recording hook.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Output Parquet file path (base name, creates `{name}_market.parquet` and `{name}_agents.parquet`).
    pub output_path: String,
    /// Skip first N ticks before recording (warmup period).
    pub warmup: u64,
    /// Record every N ticks (1 = every tick).
    pub interval: u64,
    /// Initial cash per agent for normalization.
    pub initial_cash: f64,
    /// Position limit per agent for normalization.
    pub position_limit: i64,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            output_path: "data/training.parquet".to_string(),
            warmup: 100,
            interval: 1,
            initial_cash: 100_000.0,
            position_limit: 1000,
        }
    }
}

impl RecordingConfig {
    /// Create a new recording config with the given output path.
    pub fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
            ..Default::default()
        }
    }

    /// Set the warmup period.
    pub fn with_warmup(mut self, warmup: u64) -> Self {
        self.warmup = warmup;
        self
    }

    /// Set the recording interval.
    pub fn with_interval(mut self, interval: u64) -> Self {
        self.interval = interval;
        self
    }

    /// Set the initial cash for normalization.
    pub fn with_initial_cash(mut self, cash: f64) -> Self {
        self.initial_cash = cash;
        self
    }

    /// Set the position limit for normalization.
    pub fn with_position_limit(mut self, limit: i64) -> Self {
        self.position_limit = limit;
        self
    }
}

/// Action captured from an order.
#[derive(Debug, Clone, Default)]
struct CapturedAction {
    symbol: Symbol,
    action: i8, // -1 = sell, 0 = hold, 1 = buy
    action_quantity: f64,
    action_price: f64,
}

/// Fill accumulated for an agent.
#[derive(Debug, Clone, Default)]
struct AccumulatedFill {
    quantity: f64,
    value: f64, // quantity * price, for computing average
}

/// Internal state protected by Mutex.
struct RecordingState {
    /// Rolling price history.
    price_history: PriceHistory,
    /// Parquet writer.
    writer: Option<DualParquetWriter>,
    /// Actions captured this tick (from on_orders_collected).
    captured_actions: HashMap<AgentId, CapturedAction>,
    /// Fills accumulated this tick (from on_trades).
    accumulated_fills: HashMap<AgentId, AccumulatedFill>,
    /// Previous tick PnL for reward calculation.
    previous_pnl: HashMap<AgentId, Cash>,
    /// Configuration.
    config: RecordingConfig,
    /// Error encountered (if any).
    #[allow(dead_code)]
    error: Option<String>,
}

/// Recording hook for ML training data.
///
/// Captures features, actions, and outcomes at each tick and writes
/// to Parquet files for Python ML training.
///
/// # Lifecycle
///
/// 1. `on_tick_start()`: Record mid prices to price history, clear state
/// 2. `on_orders_collected()`: Capture actions (orders submitted)
/// 3. `on_trades()`: Accumulate fills per agent
/// 4. `on_tick_end()`: Extract features, write market + agent records
/// 5. `on_simulation_end()`: Final flush and close files
pub struct RecordingHook {
    state: Mutex<RecordingState>,
}

impl RecordingHook {
    /// Create a new recording hook.
    pub fn new(config: RecordingConfig) -> Result<Self, ParquetWriterError> {
        let writer = DualParquetWriter::new(&config.output_path)?;

        Ok(Self {
            state: Mutex::new(RecordingState {
                price_history: PriceHistory::new(),
                writer: Some(writer),
                captured_actions: HashMap::new(),
                accumulated_fills: HashMap::new(),
                previous_pnl: HashMap::new(),
                config,
                error: None,
            }),
        })
    }

    /// Check if a tick should be recorded based on warmup and interval.
    fn should_record(tick: Tick, config: &RecordingConfig) -> bool {
        if tick < config.warmup {
            return false;
        }
        let adjusted_tick = tick - config.warmup;
        adjusted_tick.is_multiple_of(config.interval)
    }
}

impl SimulationHook for RecordingHook {
    fn name(&self) -> &str {
        "RecordingHook"
    }

    fn on_tick_start(&self, ctx: &HookContext) {
        let mut state = self.state.lock();

        // Record mid prices for all symbols
        for (symbol, price) in &ctx.market.mid_prices {
            state.price_history.record(ctx.tick, symbol, *price);
        }

        // Clear per-tick state
        state.captured_actions.clear();
        state.accumulated_fills.clear();
    }

    fn on_orders_collected(&self, orders: Vec<Order>, ctx: &HookContext) {
        let mut state = self.state.lock();

        // Skip if not recording this tick
        if !Self::should_record(ctx.tick, &state.config) {
            return;
        }

        // Capture actions from orders (first order per agent only)
        for order in &orders {
            let agent_id = order.agent_id;

            // Skip if we already have an action for this agent
            if state.captured_actions.contains_key(&agent_id) {
                continue;
            }

            // Determine action from order
            let limit_price = match &order.order_type {
                OrderType::Limit { price } => Some(price.0 as f64 / 100.0),
                OrderType::Market => None,
            };
            let (action, action_quantity, action_price) = match order.side {
                OrderSide::Buy => (1i8, order.quantity.0 as f64, limit_price.unwrap_or(0.0)),
                OrderSide::Sell => (-1i8, order.quantity.0 as f64, limit_price.unwrap_or(0.0)),
            };

            state.captured_actions.insert(
                agent_id,
                CapturedAction {
                    symbol: order.symbol.clone(),
                    action,
                    action_quantity,
                    action_price,
                },
            );
        }
    }

    fn on_trades(&self, trades: Vec<Trade>, ctx: &HookContext) {
        let mut state = self.state.lock();

        // Skip if not recording this tick
        if !Self::should_record(ctx.tick, &state.config) {
            return;
        }

        // Accumulate fills per agent
        for trade in trades {
            let price = trade.price.0 as f64 / 100.0;
            let qty = trade.quantity.0 as f64;

            // Update buyer
            let buyer_fill = state.accumulated_fills.entry(trade.buyer_id).or_default();
            buyer_fill.quantity += qty;
            buyer_fill.value += qty * price;

            // Update seller
            let seller_fill = state.accumulated_fills.entry(trade.seller_id).or_default();
            seller_fill.quantity += qty;
            seller_fill.value += qty * price;
        }
    }

    fn on_tick_end(&self, _stats: &SimulationStats, ctx: &HookContext) {
        let mut state = self.state.lock();

        // Skip if not recording this tick
        if !Self::should_record(ctx.tick, &state.config) {
            // Still update previous PnL for next recording tick
            if let Some(enriched) = ctx.enriched.as_ref() {
                for agent in &enriched.agent_summaries {
                    state.previous_pnl.insert(agent.id, agent.total_pnl);
                }
            }
            return;
        }

        // Get enriched data (required for feature extraction)
        let enriched = match ctx.enriched.as_ref() {
            Some(e) => e,
            None => return, // Can't extract features without enriched data
        };

        // Extract config values (Copy types)
        let initial_cash = state.config.initial_cash;
        let position_limit = state.config.position_limit;
        let tick = ctx.tick;

        // Extract read-only references for parallel closures
        let price_history = &state.price_history;
        let market = &ctx.market;

        // Collect symbols for parallel iteration
        let symbols: Vec<_> = ctx.market.mid_prices.keys().collect();

        // Extract market features in PARALLEL (42 features per symbol is CPU-bound)
        let market_records: Vec<MarketRecord> = parallel::map_slice(
            &symbols,
            |symbol| {
                let market_features =
                    MarketFeatures::extract(symbol, tick, market, Some(enriched), price_history);
                MarketRecord {
                    tick,
                    symbol: symbol.to_string(),
                    features: market_features.features,
                }
            },
            false, // Use parallel when available
        );

        // Write market records SEQUENTIALLY (Parquet writer not thread-safe)
        if let Some(ref mut writer) = state.writer {
            for market_record in market_records {
                if let Err(e) = writer.write_market(market_record) {
                    state.error = Some(e.to_string());
                    return;
                }
            }
        }

        // Extract read-only references for agent parallel closure
        let captured_actions = &state.captured_actions;
        let accumulated_fills = &state.accumulated_fills;
        let previous_pnl_map = &state.previous_pnl;
        let risk_metrics = &enriched.risk_metrics;
        let agent_summaries = &enriched.agent_summaries;

        // Build agent records in PARALLEL (feature extraction is CPU-bound)
        let agent_records: Vec<AgentRecord> = parallel::filter_map_slice(
            agent_summaries,
            |agent_summary| {
                let agent_id = agent_summary.id;

                // Skip MarketMaker agents (liquidity providers, not learning)
                // Skip PairsTrading/SectorRotator (multi-symbol, can't capture with single action)
                let name = &agent_summary.name;
                if name.contains("MarketMaker")
                    || name.contains("PairsTrading")
                    || name.contains("SectorRotator")
                {
                    return None;
                }

                // Skip multi-symbol agents (catch-all for any new multi-symbol strategies)
                if agent_summary.positions.len() > 1 {
                    return None;
                }

                // Get captured action (if any) or use "hold"
                let captured = captured_actions.get(&agent_id);
                let (agent_symbol, action, action_quantity, action_price) = match captured {
                    Some(cap) => (
                        cap.symbol.clone(),
                        cap.action,
                        cap.action_quantity,
                        cap.action_price,
                    ),
                    None => {
                        // No action - use first position symbol or skip
                        match agent_summary.positions.keys().next() {
                            Some(s) => (s.clone(), 0i8, 0.0, 0.0),
                            None => return None, // No position, skip this agent
                        }
                    }
                };

                // Build agent feature context
                let risk_snapshot = risk_metrics.get(&agent_id);
                let agent_ctx = AgentFeatureContext {
                    agent_summary,
                    risk_snapshot,
                    symbol: &agent_symbol,
                    initial_cash,
                    position_limit,
                };

                // Extract agent features (only 10 features, fast)
                let agent_features = AgentFeatures::extract(&agent_ctx);

                // Get fills
                let fill = accumulated_fills.get(&agent_id);
                let (fill_quantity, fill_price) = match fill {
                    Some(f) if f.quantity > 0.0 => (f.quantity, f.value / f.quantity),
                    _ => (0.0, 0.0),
                };

                // Compute reward (PnL change)
                let current_pnl = agent_summary.total_pnl;
                let prev_pnl = previous_pnl_map.get(&agent_id).copied().unwrap_or(Cash(0));
                let reward = (current_pnl.0 - prev_pnl.0) as f64 / 100.0;
                let reward_normalized = if initial_cash > 0.0 {
                    reward / initial_cash
                } else {
                    0.0
                };

                // Get next-tick values (current state at tick end)
                let next_position = agent_summary
                    .positions
                    .get(&agent_symbol)
                    .copied()
                    .unwrap_or(0) as f64;
                let next_pnl = agent_summary.total_pnl.0 as f64 / 100.0;

                // Build agent record
                Some(AgentRecord {
                    tick,
                    symbol: agent_symbol.to_string(),
                    agent_id: agent_id.0,
                    agent_name: agent_summary.name.clone(),
                    features: agent_features.features,
                    action,
                    action_quantity,
                    action_price,
                    fill_quantity,
                    fill_price,
                    reward,
                    reward_normalized,
                    next_position,
                    next_pnl,
                })
            },
            false, // Use parallel when available
        );

        // Write agent records SEQUENTIALLY (Parquet writer not thread-safe)
        if let Some(ref mut writer) = state.writer {
            for agent_record in agent_records {
                if let Err(e) = writer.write_agent(agent_record) {
                    state.error = Some(e.to_string());
                    break;
                }
            }
        }

        // Update previous PnL for next tick
        for agent in &enriched.agent_summaries {
            state.previous_pnl.insert(agent.id, agent.total_pnl);
        }
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        let mut state = self.state.lock();

        // Finish writing
        if let Some(writer) = state.writer.take() {
            match writer.finish() {
                Ok((market_count, agent_count)) => {
                    eprintln!(
                        "[RecordingHook] Finished: {} market rows, {} agent rows",
                        market_count, agent_count
                    );
                }
                Err(e) => {
                    eprintln!("[RecordingHook] Error closing Parquet file: {}", e);
                    state.error = Some(e.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_record() {
        let config = RecordingConfig {
            warmup: 100,
            interval: 5,
            ..Default::default()
        };

        // Before warmup
        assert!(!RecordingHook::should_record(0, &config));
        assert!(!RecordingHook::should_record(50, &config));
        assert!(!RecordingHook::should_record(99, &config));

        // At and after warmup
        assert!(RecordingHook::should_record(100, &config)); // 100 - 100 = 0, 0 % 5 == 0
        assert!(!RecordingHook::should_record(101, &config)); // 1 % 5 != 0
        assert!(!RecordingHook::should_record(104, &config)); // 4 % 5 != 0
        assert!(RecordingHook::should_record(105, &config)); // 5 % 5 == 0
        assert!(RecordingHook::should_record(110, &config)); // 10 % 5 == 0
    }

    #[test]
    fn test_config_builder() {
        let config = RecordingConfig::new("output.parquet")
            .with_warmup(200)
            .with_interval(10)
            .with_initial_cash(50_000.0)
            .with_position_limit(500);

        assert_eq!(config.output_path, "output.parquet");
        assert_eq!(config.warmup, 200);
        assert_eq!(config.interval, 10);
        assert_eq!(config.initial_cash, 50_000.0);
        assert_eq!(config.position_limit, 500);
    }
}
