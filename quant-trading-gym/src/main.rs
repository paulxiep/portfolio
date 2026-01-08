//! Quant Trading Gym - Main binary
//!
//! Runs a live trading simulation with TUI visualization.
//!
//! # Architecture
//!
//! The simulation and TUI run in separate threads, communicating via channels:
//!
//! ```text
//! ┌────────────────┐     SimUpdate      ┌────────────────┐
//! │   Simulation   │ ────────────────►  │      TUI       │
//! │   (Thread A)   │   (channel)        │   (Thread B)   │
//! │                │ ◄────────────────  │                │
//! └────────────────┘     SimCommand     └────────────────┘
//! ```
//!
//! The TUI starts paused. Press Space to start/stop the simulation.
//!
//! # Headless Mode (V3.7)
//!
//! Run `--headless` to disable TUI and run simulation to completion.
//! Useful for benchmarks, CI, and Docker containers.

mod config;

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use agents::{
    Agent, BackgroundAgentPool, BackgroundPoolConfig, BollingerReversion, BollingerReversionConfig,
    MacdCrossover, MacdCrossoverConfig, MarketMaker, MarketMakerConfig, MomentumConfig,
    MomentumTrader, NoiseTrader, NoiseTraderConfig, PairsTrading, PairsTradingConfig,
    ReactiveAgent, ReactiveStrategyType, SectorRotator, SectorRotatorConfig, TrendFollower,
    TrendFollowerConfig, VwapExecutor, VwapExecutorConfig,
};
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, bounded};
use news::config::{
    EarningsConfig, EventFrequency, GuidanceConfig, NewsGeneratorConfig, RateDecisionConfig,
    SectorNewsConfig,
};
use rand::Rng;
use rand::prelude::SliceRandom;
use simulation::{Simulation, SimulationConfig};
use tui::{AgentInfo, RiskInfo, SimCommand, SimUpdate, TuiApp};
use types::{AgentId, Cash, Price, Quantity, Sector, ShortSellingConfig, Symbol, SymbolConfig};

pub use config::{SimConfig, SymbolSpec, Tier1AgentType};

/// Quant Trading Gym - Market simulation with TUI visualization
#[derive(Parser, Debug)]
#[command(name = "quant-trading-gym")]
#[command(about = "A quantitative trading simulation with TUI visualization")]
#[command(version)]
struct Args {
    /// Run without TUI (headless mode for benchmarks/CI/Docker)
    #[arg(long, env = "SIM_HEADLESS")]
    headless: bool,

    /// Total ticks to run (0 = infinite in TUI mode)
    #[arg(long, env = "SIM_TICKS")]
    ticks: Option<u64>,

    /// Number of Tier 1 agents
    #[arg(long, env = "SIM_TIER1")]
    tier1: Option<usize>,

    /// Number of Tier 2 agents
    #[arg(long, env = "SIM_TIER2")]
    tier2: Option<usize>,

    /// Background pool size (Tier 3)
    #[arg(long, env = "SIM_POOL_SIZE")]
    pool_size: Option<usize>,

    /// Tick delay in milliseconds
    #[arg(long, env = "SIM_TICK_DELAY")]
    tick_delay: Option<u64>,

    /// Disable parallel agent collection (V3.7 profiling)
    #[arg(long, env = "PAR_AGENT_COLLECTION")]
    par_agent_collection: Option<bool>,

    /// Disable parallel indicators (V3.7 profiling)
    #[arg(long, env = "PAR_INDICATORS")]
    par_indicators: Option<bool>,

    /// Disable parallel order validation (V3.7 profiling)
    #[arg(long, env = "PAR_ORDER_VALIDATION")]
    par_order_validation: Option<bool>,

    /// Disable parallel auctions (V3.7 profiling)
    #[arg(long, env = "PAR_AUCTIONS")]
    par_auctions: Option<bool>,

    /// Disable parallel candle updates (V3.7 profiling)
    #[arg(long, env = "PAR_CANDLE_UPDATES")]
    par_candle_updates: Option<bool>,

    /// Disable parallel trade updates (V3.7 profiling)
    #[arg(long, env = "PAR_TRADE_UPDATES")]
    par_trade_updates: Option<bool>,

    /// Disable parallel fill notifications (V3.7 profiling)
    #[arg(long, env = "PAR_FILL_NOTIFICATIONS")]
    par_fill_notifications: Option<bool>,

    /// Disable parallel wake conditions (V3.7 profiling)
    #[arg(long, env = "PAR_WAKE_CONDITIONS")]
    par_wake_conditions: Option<bool>,

    /// Disable parallel risk tracking (V3.7 profiling)
    #[arg(long, env = "PAR_RISK_TRACKING")]
    par_risk_tracking: Option<bool>,
}

/// Calculate the number of digits needed to display a number.
fn digit_width(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as usize + 1
    }
}

/// Build a SimUpdate from current simulation state.
///
/// V2.3: Now builds per-symbol data for multi-symbol support.
/// V3.2: Added tier1_count and tier2_count for agent tier display.
fn build_update(
    sim: &Simulation,
    price_history: &HashMap<Symbol, Vec<f64>>,
    finished: bool,
    tier1_count: usize,
    tier2_count: usize,
) -> SimUpdate {
    let stats = sim.stats();
    let symbols: Vec<Symbol> = sim.config().symbols();

    // Calculate digit width for agent numbering
    let agent_summaries = sim.agent_summaries();
    let num_agents = agent_summaries.len();
    let width = digit_width(num_agents);

    // Build agent info from simulation (V3.1: per-symbol positions)
    let agents: Vec<AgentInfo> = agent_summaries
        .iter()
        .enumerate()
        .map(|(i, (name, positions, cash, total_pnl))| {
            let is_mm = name.contains("Market");
            // Calculate equity from all positions
            let position_value: f64 = positions
                .iter()
                .map(|(sym, qty)| {
                    let price = sim
                        .get_book(sym)
                        .and_then(|b| b.last_price())
                        .map(|p| p.to_float())
                        .unwrap_or(100.0);
                    *qty as f64 * price
                })
                .sum();
            let equity = cash.to_float() + position_value;

            AgentInfo {
                name: format!("{:0width$}-{}", i + 1, name, width = width),
                positions: positions.clone(),
                total_pnl: *total_pnl,
                cash: *cash,
                is_market_maker: is_mm,
                equity,
            }
        })
        .collect();

    // Build risk metrics from simulation
    let risk_metrics_map = sim.agent_risk_metrics();
    let risk_metrics: Vec<RiskInfo> = agent_summaries
        .iter()
        .enumerate()
        .filter_map(|(i, (name, _, _, _))| {
            let agent_id = AgentId((i + 1) as u64);
            let is_mm = name.contains("Market");
            risk_metrics_map.get(&agent_id).map(|metrics| RiskInfo {
                name: format!("{:0width$}-{}", i + 1, name, width = width),
                sharpe: metrics.sharpe,
                max_drawdown: metrics.max_drawdown,
                total_return: metrics.total_return,
                var_95: metrics.var_95,
                equity: metrics.equity,
                is_market_maker: is_mm,
            })
        })
        .collect();

    // Build per-symbol book data (V2.3)
    let (bids_map, asks_map): (HashMap<_, _>, HashMap<_, _>) = symbols
        .iter()
        .filter_map(|symbol| sim.get_book(symbol).map(|book| (symbol, book)))
        .map(|(symbol, book)| {
            let snapshot = book.snapshot(sim.timestamp(), sim.tick(), 10);
            (
                (symbol.clone(), snapshot.bids),
                (symbol.clone(), snapshot.asks),
            )
        })
        .unzip();

    let last_price_map: HashMap<_, _> = symbols
        .iter()
        .filter_map(|symbol| {
            sim.get_book(symbol)
                .and_then(|book| book.last_price().map(|price| (symbol.clone(), price)))
        })
        .collect();

    // Aggregate trades across all symbols
    let trades: Vec<_> = symbols
        .iter()
        .flat_map(|s| sim.recent_trades_for(s).iter().cloned())
        .collect();

    SimUpdate {
        tick: sim.tick(),
        symbols,
        selected_symbol: 0,
        price_history: price_history.clone(),
        bids: bids_map,
        asks: asks_map,
        last_price: last_price_map,
        trades,
        agents,
        tier1_count,
        tier2_count,
        tier3_count: sim
            .background_pool()
            .map(|p| p.config().pool_size)
            .unwrap_or(0),
        total_trades: stats.total_trades,
        total_orders: stats.total_orders,
        agents_called: stats.agents_called_this_tick,
        t2_triggered: stats.t2_triggered_this_tick,
        t3_orders: stats.t3_orders_this_tick,
        background_pnl: sim
            .background_pool()
            .map(|p| p.accounting().realized_pnl().to_float())
            .unwrap_or(0.0),
        finished,
        risk_metrics,
    }
}

/// Spawn a single agent of the given type with the specified ID.
///
/// For multi-symbol support: symbol and initial_price are passed explicitly
/// so agents can be spawned per-symbol (market makers, noise traders) or
/// assigned to the primary symbol (quant strategies).
fn create_agent(
    agent_type: Tier1AgentType,
    id: u64,
    config: &SimConfig,
    symbol: &str,
    initial_price: Price,
) -> Box<dyn Agent> {
    let id = AgentId(id);

    match agent_type {
        Tier1AgentType::NoiseTrader => {
            // Adjust initial cash to give equal net worth regardless of symbol price.
            // Target: same equity as quant strategies (100k) at any price.
            // Formula: cash = target_equity - (initial_position * price)
            // With initial_position = 50 and target equity = 100k:
            //   At $100: cash = 100k - 50*100 = 95k (matches nt_initial_cash default)
            //   At $50:  cash = 100k - 50*50 = 97.5k
            //   At $200: cash = 100k - 50*200 = 90k
            let target_equity = config.quant_initial_cash.to_float();
            let position_value = config.nt_initial_position as f64 * initial_price.to_float();
            let adjusted_cash = target_equity - position_value;

            let nt_config = NoiseTraderConfig {
                symbol: symbol.to_string(),
                order_probability: config.nt_order_probability,
                initial_price,
                price_deviation: config.nt_price_deviation,
                min_quantity: config.nt_min_quantity,
                max_quantity: config.nt_max_quantity,
                initial_cash: Cash::from_float(adjusted_cash),
                initial_position: config.nt_initial_position,
            };
            Box::new(NoiseTrader::new(id, nt_config))
        }
        Tier1AgentType::MomentumTrader => {
            let momentum_config = MomentumConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            Box::new(MomentumTrader::new(id, momentum_config))
        }
        Tier1AgentType::TrendFollower => {
            let trend_config = TrendFollowerConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            Box::new(TrendFollower::new(id, trend_config))
        }
        Tier1AgentType::MacdTrader => {
            let macd_config = MacdCrossoverConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            Box::new(MacdCrossover::new(id, macd_config))
        }
        Tier1AgentType::BollingerTrader => {
            let bollinger_config = BollingerReversionConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            Box::new(BollingerReversion::new(id, bollinger_config))
        }
        Tier1AgentType::VwapExecutor => {
            let vwap_config = VwapExecutorConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                ..Default::default()
            };
            Box::new(VwapExecutor::new(id, vwap_config))
        }
        Tier1AgentType::PairsTrading => {
            // PairsTrading requires two symbols - should be handled separately
            // This is a fallback that creates a self-referencing pair (effectively no-op)
            let pairs_config = PairsTradingConfig::new(symbol, symbol)
                .with_initial_cash(config.quant_initial_cash)
                .with_max_position(config.quant_max_position);
            Box::new(PairsTrading::new(id, pairs_config))
        }
    }
}

/// Spawn Tier 2 reactive agents distributed across symbols.
///
/// Each agent gets:
/// - ThresholdBuyer entry (buy at absolute price level)
/// - 1-2 exit strategies (StopLoss, TakeProfit, or ThresholdSeller)
/// - Optionally NewsReactor (20% chance)
///
/// Price assumptions:
/// - Initial price: $100 (1,000,000 in fixed-point)
/// - Can drop to $50 initially due to selling pressure
/// - ThresholdBuyer targets: $50-$95 range to catch dips
/// - ThresholdSeller targets: above entry price for profit taking
fn spawn_tier2_agents(
    sim: &mut Simulation,
    next_id: &mut u64,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    rng: &mut rand::prelude::ThreadRng,
) {
    use types::Price;

    let num_agents = config.num_tier2_agents;
    if num_agents == 0 {
        return;
    }

    let num_symbols = symbols.len();
    let agents_per_symbol = num_agents / num_symbols;
    let remainder = num_agents % num_symbols;

    // Fixed-point scale: 10000 per dollar
    // $50 = 500_000, $100 = 1,000,000
    const PRICE_SCALE: i64 = 10_000;

    // Helper to create strategies for one agent
    let make_strategies = |rng: &mut rand::prelude::ThreadRng| -> Vec<ReactiveStrategyType> {
        let buy_dollars = config.t2_buy_threshold_min
            + rng.r#gen::<f64>() * (config.t2_buy_threshold_max - config.t2_buy_threshold_min);
        let buy_price = Price((buy_dollars * PRICE_SCALE as f64) as i64);
        let entry_size = config.t2_order_size_min
            + rng.r#gen::<f64>() * (config.t2_order_size_max - config.t2_order_size_min);

        let stop_pct = config.t2_stop_loss_min
            + rng.r#gen::<f64>() * (config.t2_stop_loss_max - config.t2_stop_loss_min);

        let mut strategies = vec![
            ReactiveStrategyType::ThresholdBuyer {
                buy_price,
                size_fraction: entry_size,
            },
            ReactiveStrategyType::StopLoss { stop_pct },
        ];

        // TakeProfit or ThresholdSeller based on config probability
        if rng.r#gen::<f64>() < config.t2_take_profit_prob {
            let target_pct = config.t2_take_profit_min
                + rng.r#gen::<f64>() * (config.t2_take_profit_max - config.t2_take_profit_min);
            strategies.push(ReactiveStrategyType::TakeProfit { target_pct });
        } else {
            let sell_dollars = config.t2_sell_threshold_min
                + rng.r#gen::<f64>()
                    * (config.t2_sell_threshold_max - config.t2_sell_threshold_min);
            strategies.push(ReactiveStrategyType::ThresholdSeller {
                sell_price: Price((sell_dollars * PRICE_SCALE as f64) as i64),
                size_fraction: 1.0,
            });
        }

        // NewsReactor based on config probability
        if rng.r#gen::<f64>() < config.t2_news_reactor_prob {
            strategies.push(ReactiveStrategyType::NewsReactor {
                min_magnitude: 0.3 + rng.r#gen::<f64>() * 0.4,
                sentiment_multiplier: 1.0 + rng.r#gen::<f64>() * 2.0,
            });
        }

        strategies
    };

    // Build (spec, count) pairs then flatten to agent assignments
    let agent_specs: Vec<_> = symbols
        .iter()
        .enumerate()
        .flat_map(|(sym_idx, spec)| {
            let count = agents_per_symbol + if sym_idx < remainder { 1 } else { 0 };
            std::iter::repeat_n(spec, count)
        })
        .collect();

    let start_id = *next_id;
    let agents: Vec<_> = agent_specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            Box::new(ReactiveAgent::new(
                AgentId(start_id + i as u64),
                spec.symbol.clone().into(),
                make_strategies(rng),
                Quantity(config.t2_max_position),
                config.t2_initial_cash,
            )) as Box<dyn Agent>
        })
        .collect();

    *next_id += agents.len() as u64;
    for agent in agents {
        sim.add_agent(agent);
    }
}

/// Spawn Sector Rotator agents (V3.3 - special Tier 2 category).
///
/// SectorRotator agents:
/// - Watch all symbols across sectors
/// - React to news events affecting their watched sectors
/// - Rotate allocation to highest-sentiment sectors
///
/// These are displayed as "SectorRotator" in TUI (not "ReactiveAgent").
fn spawn_sector_rotators(
    sim: &mut Simulation,
    next_id: &mut u64,
    config: &SimConfig,
    symbols: &[SymbolSpec],
) {
    let num_agents = config.num_sector_rotators;
    if num_agents == 0 || symbols.is_empty() {
        return;
    }

    // Group symbols by sector
    use std::collections::HashMap;
    let mut sector_symbols: HashMap<Sector, Vec<String>> = HashMap::new();
    for spec in symbols {
        sector_symbols
            .entry(spec.sector)
            .or_default()
            .push(spec.symbol.clone());
    }

    // Create sector rotators - each watches all sectors
    let start_id = *next_id;
    let agents: Vec<_> = (0..num_agents)
        .map(|i| {
            let mut config = SectorRotatorConfig::new()
                .with_initial_cash(config.quant_initial_cash)
                .with_sentiment_scale(0.3) // ±30% allocation shift based on sentiment
                .with_rebalance_threshold(0.05); // Rebalance on 5% drift

            // Add all sector -> symbols mappings
            for (sector, syms) in &sector_symbols {
                config = config.with_sector(*sector, syms.clone());
            }

            Box::new(SectorRotator::new(AgentId(start_id + i as u64), config)) as Box<dyn Agent>
        })
        .collect();

    *next_id += agents.len() as u64;
    for agent in agents {
        sim.add_agent(agent);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation Setup Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build simulation configuration from SimConfig.
fn build_simulation_config(config: &SimConfig, args: &Args) -> SimulationConfig {
    let symbol_configs: Vec<SymbolConfig> = config
        .get_symbols()
        .iter()
        .map(|spec| {
            SymbolConfig::with_sector(
                &spec.symbol,
                Quantity(10_000_000), // 10M shares outstanding
                spec.initial_price,
                spec.sector,
            )
            .with_borrow_pool_bps(2000) // 20% available to borrow
        })
        .collect();

    // Enable short selling with tight limits matching agent configs
    let short_config = ShortSellingConfig::enabled_default().with_max_short(Quantity(500));

    // Build news/event config from SimConfig settings
    let news_config = if config.events_enabled {
        NewsGeneratorConfig {
            earnings: EarningsConfig {
                frequency: EventFrequency::new(
                    config.event_earnings_prob,
                    config.event_earnings_interval,
                ),
                ..Default::default()
            },
            guidance: GuidanceConfig {
                frequency: EventFrequency::new(
                    config.event_guidance_prob,
                    config.event_guidance_interval,
                ),
                ..Default::default()
            },
            rate_decision: RateDecisionConfig {
                frequency: EventFrequency::new(
                    config.event_rate_decision_prob,
                    config.event_rate_decision_interval,
                ),
                ..Default::default()
            },
            sector_news: SectorNewsConfig {
                frequency: EventFrequency::new(
                    config.event_sector_news_prob,
                    config.event_sector_news_interval,
                ),
                ..Default::default()
            },
        }
    } else {
        NewsGeneratorConfig::disabled()
    };

    // Build parallelization config from args (V3.7)
    let mut par_config = simulation::ParallelizationConfig::default();
    if let Some(val) = args.par_agent_collection {
        par_config.parallel_agent_collection = val;
    }
    if let Some(val) = args.par_indicators {
        par_config.parallel_indicators = val;
    }
    if let Some(val) = args.par_order_validation {
        par_config.parallel_order_validation = val;
    }
    if let Some(val) = args.par_auctions {
        par_config.parallel_auctions = val;
    }
    if let Some(val) = args.par_candle_updates {
        par_config.parallel_candle_updates = val;
    }
    if let Some(val) = args.par_trade_updates {
        par_config.parallel_trade_updates = val;
    }
    if let Some(val) = args.par_fill_notifications {
        par_config.parallel_fill_notifications = val;
    }
    if let Some(val) = args.par_wake_conditions {
        par_config.parallel_wake_conditions = val;
    }
    if let Some(val) = args.par_risk_tracking {
        par_config.parallel_risk_tracking = val;
    }

    SimulationConfig::with_symbols(symbol_configs)
        .with_short_selling(short_config)
        .with_verbose(config.verbose)
        .with_news_config(news_config)
        .with_parallelization(par_config)
}

/// Create a MarketMakerConfig for a given symbol spec.
fn make_mm_config(spec: &SymbolSpec, config: &SimConfig) -> MarketMakerConfig {
    MarketMakerConfig {
        symbol: spec.symbol.clone(),
        initial_price: spec.initial_price,
        half_spread: config.mm_half_spread,
        quote_size: config.mm_quote_size,
        refresh_interval: config.mm_refresh_interval,
        max_inventory: config.mm_max_inventory,
        inventory_skew: config.mm_inventory_skew,
        initial_cash: config.mm_initial_cash,
        initial_position: 500,
        fair_value_weight: 0.3,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Spawning Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Spawn market makers distributed across symbols.
/// Returns (agents, next_id).
fn spawn_market_makers(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let num_symbols = symbols.len();
    let per_symbol = config.num_market_makers / num_symbols;
    let remainder = config.num_market_makers % num_symbols;

    let mut next_id = start_id;

    // Distributed evenly across symbols
    let distributed: Vec<_> = symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, per_symbol))
        .zip(next_id..)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec, config))) as Box<dyn Agent>
        })
        .collect();
    next_id += distributed.len() as u64;

    // Remainder randomly assigned
    let remainder_agents: Vec<_> = (0..remainder)
        .map(|_| symbols.choose(rng).unwrap())
        .zip(next_id..)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec, config))) as Box<dyn Agent>
        })
        .collect();
    next_id += remainder_agents.len() as u64;

    let agents = distributed.into_iter().chain(remainder_agents).collect();
    (agents, next_id)
}

/// Spawn noise traders distributed across symbols.
/// Returns (agents, next_id).
fn spawn_noise_traders(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let num_symbols = symbols.len();
    let per_symbol = config.num_noise_traders / num_symbols;
    let remainder = config.num_noise_traders % num_symbols;

    let mut next_id = start_id;

    // Distributed evenly
    let distributed: Vec<_> = symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, per_symbol))
        .zip(next_id..)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += distributed.len() as u64;

    // Remainder randomly assigned
    let remainder_agents: Vec<_> = (0..remainder)
        .map(|_| symbols.choose(rng).unwrap())
        .zip(next_id..)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += remainder_agents.len() as u64;

    let agents = distributed.into_iter().chain(remainder_agents).collect();
    (agents, next_id)
}

/// Spawn quant strategy agents (momentum, trend, MACD, etc.) randomly across symbols.
/// Returns (agents, next_id).
fn spawn_quant_agents(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let agent_counts = [
        (Tier1AgentType::MomentumTrader, config.num_momentum_traders),
        (Tier1AgentType::TrendFollower, config.num_trend_followers),
        (Tier1AgentType::MacdTrader, config.num_macd_traders),
        (
            Tier1AgentType::BollingerTrader,
            config.num_bollinger_traders,
        ),
        (Tier1AgentType::VwapExecutor, config.num_vwap_executors),
    ];

    let specs: Vec<_> = agent_counts
        .iter()
        .flat_map(|(agent_type, count)| std::iter::repeat_n(*agent_type, *count))
        .map(|agent_type| (agent_type, symbols.choose(rng).unwrap()))
        .collect();

    let agents: Vec<_> = specs
        .iter()
        .zip(start_id..)
        .map(|((agent_type, spec), id)| {
            create_agent(*agent_type, id, config, &spec.symbol, spec.initial_price)
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}

/// Spawn pairs trading agents (requires at least 2 symbols).
/// Returns (agents, next_id).
fn spawn_pairs_traders(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
) -> (Vec<Box<dyn Agent>>, u64) {
    if symbols.len() < 2 {
        return (Vec::new(), start_id);
    }

    let num_symbols = symbols.len();
    let agents: Vec<_> = (0..config.num_pairs_traders)
        .zip(start_id..)
        .map(|(i, id)| {
            let idx_a = i % num_symbols;
            let idx_b = (i + 1) % num_symbols;
            let spec_a = &symbols[idx_a];
            let spec_b = &symbols[idx_b];

            let pairs_config = PairsTradingConfig::new(&spec_a.symbol, &spec_b.symbol)
                .with_initial_cash(config.quant_initial_cash)
                .with_max_position(config.quant_max_position);

            Box::new(PairsTrading::new(AgentId(id), pairs_config)) as Box<dyn Agent>
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}

/// Spawn random Tier 1 agents to fill to minimum count.
/// Returns (agents, next_id).
fn spawn_random_tier1_agents(
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    rng: &mut rand::prelude::ThreadRng,
) -> (Vec<Box<dyn Agent>>, u64) {
    let count = config.random_tier1_count();
    let agents: Vec<_> = (0..count)
        .map(|_| (Tier1AgentType::random(rng), symbols.choose(rng).unwrap()))
        .zip(start_id..)
        .map(|((agent_type, spec), id)| {
            create_agent(agent_type, id, config, &spec.symbol, spec.initial_price)
        })
        .collect();

    let next_id = start_id + agents.len() as u64;
    (agents, next_id)
}

/// Spawn all Tier 1 agents and add them to the simulation.
/// Returns next available agent ID.
fn spawn_all_tier1_agents(
    sim: &mut Simulation,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    rng: &mut rand::prelude::ThreadRng,
) -> u64 {
    let mut next_id = 1u64;

    let (mm_agents, id) = spawn_market_makers(config, symbols, next_id, rng);
    next_id = id;

    let (nt_agents, id) = spawn_noise_traders(config, symbols, next_id, rng);
    next_id = id;

    let (quant_agents, id) = spawn_quant_agents(config, symbols, next_id, rng);
    next_id = id;

    let (pairs_agents, id) = spawn_pairs_traders(config, symbols, next_id);
    next_id = id;

    let (random_agents, id) = spawn_random_tier1_agents(config, symbols, next_id, rng);
    next_id = id;

    // Add all agents to simulation
    for agent in mm_agents
        .into_iter()
        .chain(nt_agents)
        .chain(quant_agents)
        .chain(pairs_agents)
        .chain(random_agents)
    {
        sim.add_agent(agent);
    }

    next_id
}

// ─────────────────────────────────────────────────────────────────────────────
// Background Pool Setup
// ─────────────────────────────────────────────────────────────────────────────

/// Setup the Tier 3 background pool if enabled.
fn setup_background_pool(sim: &mut Simulation, config: &SimConfig, symbols: &[SymbolSpec]) {
    if !config.enable_background_pool {
        return;
    }

    let pool_symbols: Vec<String> = symbols.iter().map(|s| s.symbol.clone()).collect();
    let symbol_sectors: std::collections::HashMap<String, types::Sector> = symbols
        .iter()
        .map(|s| (s.symbol.clone(), s.sector))
        .collect();

    let pool_config = BackgroundPoolConfig {
        pool_size: config.background_pool_size,
        regime: config.background_regime,
        symbols: pool_symbols,
        mean_order_size: config.t3_mean_order_size,
        order_size_stddev: config.t3_order_size_stddev,
        max_order_size: config.t3_max_order_size,
        min_order_size: 1,
        price_spread_lambda: config.t3_price_spread_lambda,
        max_price_deviation: config.t3_max_price_deviation,
        sentiment_decay: 0.995,
        max_sentiment: 0.8,
        news_sentiment_scale: 0.5,
        enable_sanity_check: true,
        max_pnl_loss_fraction: 0.05,
        base_activity_override: config.t3_base_activity,
    };

    let mut pool = BackgroundAgentPool::new(pool_config, 42);
    pool.init_sectors(&symbol_sectors);
    sim.set_background_pool(pool);
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation Loop Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// State for the simulation event loop.
struct SimulationLoopState {
    running: bool,
    tick: u64,
    total_ticks: u64,
    tier1_count: usize,
    tier2_count: usize,
    symbols: Vec<Symbol>,
    price_history: HashMap<Symbol, Vec<f64>>,
    max_price_history: usize,
    tick_delay_ms: u64,
}

impl SimulationLoopState {
    fn new(config: &SimConfig) -> Self {
        let symbols: Vec<Symbol> = config
            .get_symbols()
            .iter()
            .map(|s| s.symbol.clone())
            .collect();

        let price_history: HashMap<Symbol, Vec<f64>> = config
            .get_symbols()
            .iter()
            .map(|spec| {
                let mut history = Vec::with_capacity(config.max_price_history);
                history.push(spec.initial_price.to_float());
                (spec.symbol.clone(), history)
            })
            .collect();

        Self {
            running: false,
            tick: 0,
            total_ticks: config.total_ticks,
            tier1_count: config.total_tier1_agents(),
            tier2_count: config.num_tier2_agents + config.num_sector_rotators,
            symbols,
            price_history,
            max_price_history: config.max_price_history,
            tick_delay_ms: config.tick_delay_ms,
        }
    }

    /// Update price history from current simulation state.
    fn update_price_history(&mut self, sim: &Simulation) {
        for symbol in &self.symbols {
            if let Some(book) = sim.get_book(symbol)
                && let Some(price) = book.last_price()
            {
                let history = self.price_history.entry(symbol.clone()).or_default();
                history.push(price.to_float());
                if history.len() > self.max_price_history {
                    history.remove(0);
                }
            }
        }
    }

    /// Build a SimUpdate from current state.
    fn build_update(&self, sim: &Simulation, finished: bool) -> SimUpdate {
        build_update(
            sim,
            &self.price_history,
            finished,
            self.tier1_count,
            self.tier2_count,
        )
    }
}

/// Process incoming commands, returning whether to continue the loop.
fn process_commands(
    cmd_rx: &Receiver<SimCommand>,
    state: &mut SimulationLoopState,
    sim: &Simulation,
    tx: &Sender<SimUpdate>,
) -> bool {
    while let Ok(cmd) = cmd_rx.try_recv() {
        match cmd {
            SimCommand::Start => state.running = true,
            SimCommand::Pause => state.running = false,
            SimCommand::Toggle => state.running = !state.running,
            SimCommand::Quit => {
                let _ = tx.send(state.build_update(sim, true));
                return false;
            }
        }
    }
    true
}

/// Wait for quit command after simulation finishes.
fn wait_for_quit(cmd_rx: &Receiver<SimCommand>) {
    loop {
        match cmd_rx.recv() {
            Ok(SimCommand::Quit) | Err(_) => return,
            _ => {}
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Simulation Entry Point
// ─────────────────────────────────────────────────────────────────────────────

/// Run the simulation, sending updates to the TUI via channel.
///
/// The simulation starts **paused** and waits for a Start or Toggle command.
/// Use the command receiver to control start/stop/quit.
fn run_simulation(
    tx: Sender<SimUpdate>,
    cmd_rx: Receiver<SimCommand>,
    config: SimConfig,
    args: Args,
) {
    // Phase 1: Build simulation
    let sim_config = build_simulation_config(&config, &args);
    let mut sim = Simulation::new(sim_config);

    let all_symbols: Vec<_> = config.get_symbols().to_vec();
    let mut rng = rand::thread_rng();

    // Phase 2: Spawn Tier 1 agents
    let mut next_id = spawn_all_tier1_agents(&mut sim, &config, &all_symbols, &mut rng);

    // Phase 3: Spawn Tier 2 agents
    spawn_tier2_agents(&mut sim, &mut next_id, &config, &all_symbols, &mut rng);
    spawn_sector_rotators(&mut sim, &mut next_id, &config, &all_symbols);

    // Phase 4: Setup background pool
    setup_background_pool(&mut sim, &config, &all_symbols);

    // Phase 5: Initialize loop state and send initial update
    let mut state = SimulationLoopState::new(&config);
    let _ = tx.send(state.build_update(&sim, false));

    // Phase 6: Main event loop
    loop {
        if !process_commands(&cmd_rx, &mut state, &sim, &tx) {
            return;
        }

        if !state.running {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        if state.tick >= state.total_ticks {
            let _ = tx.send(state.build_update(&sim, true));
            wait_for_quit(&cmd_rx);
            return;
        }

        sim.step();
        state.tick += 1;
        state.update_price_history(&sim);

        if tx.send(state.build_update(&sim, false)).is_err() {
            break;
        }

        if state.tick_delay_ms > 0 {
            thread::sleep(Duration::from_millis(state.tick_delay_ms));
        }
    }
}

fn main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Parse CLI arguments (V3.7)
    // ─────────────────────────────────────────────────────────────────────────
    let args = Args::parse();

    // Build config with CLI/env overrides
    let mut config = SimConfig::default();

    if let Some(ticks) = args.ticks {
        config.total_ticks = ticks;
    }
    if let Some(tier1) = args.tier1 {
        config.min_tier1_agents = tier1;
    }
    if let Some(tier2) = args.tier2 {
        config.num_tier2_agents = tier2;
    }
    if let Some(pool_size) = args.pool_size {
        config.background_pool_size = pool_size;
    }
    if let Some(delay) = args.tick_delay {
        config.tick_delay_ms = delay;
    }

    // In headless mode, ensure we have a finite tick count
    if args.headless && config.total_ticks == 0 {
        config.total_ticks = 10_000; // Default for headless
    }

    // Print config summary
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!(
        "║  Quant Trading Gym - {}                          ║",
        if args.headless {
            "Headless Mode"
        } else {
            "TUI Mode     "
        }
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Symbol: {:6}  │  Initial Price: ${:<8.2}                      ║",
        config.primary_symbol(),
        config.primary_initial_price().to_float()
    );
    eprintln!(
        "║  Ticks:  {:6}  │  Tick Delay: {:3}ms                              ║",
        config.total_ticks, config.tick_delay_ms
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!("║  Tier 1 Agents (specified minimums):                                  ║");
    eprintln!(
        "║    Market Makers:   {:2}  │  Noise Traders:    {:2}                      ║",
        config.num_market_makers, config.num_noise_traders
    );
    eprintln!(
        "║    Momentum (RSI):  {:2}  │  Trend Followers:  {:2}                      ║",
        config.num_momentum_traders, config.num_trend_followers
    );
    eprintln!(
        "║    MACD Crossover:  {:2}  │  Bollinger:        {:2}                      ║",
        config.num_macd_traders, config.num_bollinger_traders
    );
    eprintln!(
        "║    VWAP Executors:  {:2}  │                                              ║",
        config.num_vwap_executors
    );
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Tier 1 Min: {:2}  │  Random Fill: {:2}  │  Total T1: {:2}               ║",
        config.min_tier1_agents,
        config.random_tier1_count(),
        config.total_tier1_agents()
    );
    eprintln!(
        "║  Tier 2 Agents: {:4}                                                 ║",
        config.num_tier2_agents
    );
    if config.enable_background_pool {
        eprintln!(
            "║  Tier 3 Background Pool: {:5} (statistical)                           ║",
            config.background_pool_size
        );
    }
    eprintln!(
        "║  Total Agents: {:5}  │  Total Cash: ${:>14.2}          ║",
        config.total_agents(),
        config.total_starting_cash().to_float()
    );
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    if args.headless {
        // ─────────────────────────────────────────────────────────────────────
        // Headless mode: run simulation without TUI
        // ─────────────────────────────────────────────────────────────────────
        run_headless(config, args);
    } else {
        // ─────────────────────────────────────────────────────────────────────
        // TUI mode: interactive visualization
        // ─────────────────────────────────────────────────────────────────────
        eprintln!("  Press Space to start simulation...");
        eprintln!();
        run_with_tui(config, args);
    }
}

/// Run simulation in headless mode (no TUI).
fn run_headless(config: SimConfig, args: Args) {
    use std::time::Instant;

    let total_ticks = config.total_ticks;
    let tick_delay_ms = config.tick_delay_ms;

    // Build simulation (same as run_simulation)
    let sim_config = build_simulation_config(&config, &args);
    let mut sim = Simulation::new(sim_config);

    let all_symbols: Vec<_> = config.get_symbols().to_vec();
    let mut rng = rand::thread_rng();

    // Spawn all agents (same as run_simulation)
    let mut next_id = spawn_all_tier1_agents(&mut sim, &config, &all_symbols, &mut rng);
    spawn_tier2_agents(&mut sim, &mut next_id, &config, &all_symbols, &mut rng);
    spawn_sector_rotators(&mut sim, &mut next_id, &config, &all_symbols);
    setup_background_pool(&mut sim, &config, &all_symbols);

    eprintln!("Running {} ticks...", total_ticks);
    let start = Instant::now();

    for tick in 0..total_ticks {
        sim.step();

        if tick_delay_ms > 0 {
            thread::sleep(Duration::from_millis(tick_delay_ms));
        }

        // Progress every 10%
        if tick > 0 && tick % (total_ticks / 10).max(1) == 0 {
            let pct = (tick * 100) / total_ticks;
            eprintln!("  {}% ({}/{} ticks)", pct, tick, total_ticks);
        }
    }

    let elapsed = start.elapsed();
    let stats = sim.stats();

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Simulation Complete                                                  ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Ticks: {:8}  │  Elapsed: {:6.2}s  │  Rate: {:8.0} ticks/s     ║",
        total_ticks,
        elapsed.as_secs_f64(),
        total_ticks as f64 / elapsed.as_secs_f64()
    );
    eprintln!(
        "║  Total Orders: {:8}  │  Total Trades: {:8}                   ║",
        stats.total_orders, stats.total_trades
    );
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
}

/// Run simulation with TUI visualization.
fn run_with_tui(config: SimConfig, args: Args) {
    // Create bounded channel for updates (backpressure if TUI falls behind)
    let (tx, rx) = bounded::<SimUpdate>(100);

    // Create unbounded channel for commands (TUI → simulation)
    let (cmd_tx, cmd_rx) = bounded::<SimCommand>(10);

    // Spawn simulation thread
    let tui_frame_rate = config.tui_frame_rate;
    let sim_handle = thread::spawn(move || {
        run_simulation(tx, cmd_rx, config, args);
    });

    // Run TUI in main thread (required for terminal control)
    let app = TuiApp::new(rx)
        .with_command_sender(cmd_tx)
        .frame_rate(tui_frame_rate);
    if let Err(e) = app.run() {
        eprintln!("TUI error: {}", e);
    }

    // Wait for simulation to finish
    let _ = sim_handle.join();
}
