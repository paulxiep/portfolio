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

mod config;

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use agents::{
    Agent, BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig,
    MarketMaker, MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    ReactiveAgent, ReactiveStrategyType, TrendFollower, TrendFollowerConfig, VwapExecutor,
    VwapExecutorConfig,
};
use crossbeam_channel::{Receiver, Sender, bounded};
use rand::Rng;
use rand::prelude::IndexedRandom;
use simulation::{Simulation, SimulationConfig};
use tui::{AgentInfo, RiskInfo, SimCommand, SimUpdate, TuiApp};
use types::{AgentId, Cash, Price, Quantity, ShortSellingConfig, Symbol, SymbolConfig};

pub use config::{SimConfig, SymbolSpec, Tier1AgentType};

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
        .map(|(i, (name, positions, cash, realized_pnl))| {
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
                realized_pnl: *realized_pnl,
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
        total_trades: stats.total_trades,
        total_orders: stats.total_orders,
        agents_called: stats.agents_called_this_tick,
        t2_triggered: stats.t2_triggered_this_tick,
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
            + rng.random::<f64>() * (config.t2_buy_threshold_max - config.t2_buy_threshold_min);
        let buy_price = Price((buy_dollars * PRICE_SCALE as f64) as i64);
        let entry_size = 0.3 + rng.random::<f64>() * 0.4;

        let stop_pct = config.t2_stop_loss_min
            + rng.random::<f64>() * (config.t2_stop_loss_max - config.t2_stop_loss_min);

        let mut strategies = vec![
            ReactiveStrategyType::ThresholdBuyer {
                buy_price,
                size_fraction: entry_size,
            },
            ReactiveStrategyType::StopLoss { stop_pct },
        ];

        // TakeProfit or ThresholdSeller based on config probability
        if rng.random::<f64>() < config.t2_take_profit_prob {
            let target_pct = config.t2_take_profit_min
                + rng.random::<f64>() * (config.t2_take_profit_max - config.t2_take_profit_min);
            strategies.push(ReactiveStrategyType::TakeProfit { target_pct });
        } else {
            let sell_dollars = config.t2_sell_threshold_min
                + rng.random::<f64>()
                    * (config.t2_sell_threshold_max - config.t2_sell_threshold_min);
            strategies.push(ReactiveStrategyType::ThresholdSeller {
                sell_price: Price((sell_dollars * PRICE_SCALE as f64) as i64),
                size_fraction: 1.0,
            });
        }

        // NewsReactor based on config probability
        if rng.random::<f64>() < config.t2_news_reactor_prob {
            strategies.push(ReactiveStrategyType::NewsReactor {
                min_magnitude: 0.3 + rng.random::<f64>() * 0.4,
                sentiment_multiplier: 1.0 + rng.random::<f64>() * 2.0,
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

/// Run the simulation, sending updates to the TUI via channel.
///
/// The simulation starts **paused** and waits for a Start or Toggle command.
/// Use the command receiver to control start/stop/quit.
fn run_simulation(tx: Sender<SimUpdate>, cmd_rx: Receiver<SimCommand>, config: SimConfig) {
    // Build symbol configs from SimConfig symbols (V2.3)
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
    // Most agents have max_position of 200, MMs have max_inventory of 200
    let short_config = ShortSellingConfig::enabled_default().with_max_short(Quantity(500)); // Max 500 short per agent

    // Create simulation with position limits and short selling enabled (V2.3: multi-symbol)
    let sim_config = SimulationConfig::with_symbols(symbol_configs)
        .with_short_selling(short_config)
        .with_verbose(config.verbose);

    let mut sim = Simulation::new(sim_config);
    let mut next_id: u64 = 1;

    // Get all symbols for multi-symbol agent spawning
    let all_symbols: Vec<_> = config.get_symbols().to_vec();
    let num_symbols = all_symbols.len();

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: Spawn specified minimum agents for each type
    // Agents are distributed across symbols (round-robin for guaranteed coverage,
    // then random for remainder)
    // ─────────────────────────────────────────────────────────────────────────

    let mut rng = rand::rng();

    // Market Makers (infrastructure, distributed across symbols)
    let mm_per_symbol = config.num_market_makers / num_symbols;
    let mm_remainder = config.num_market_makers % num_symbols;
    let mm_distributed_count = mm_per_symbol * num_symbols;

    // Helper to create MM config
    let make_mm_config = |spec: &SymbolSpec| MarketMakerConfig {
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
    };

    // Distributed MMs across all symbols
    let distributed_mms: Vec<_> = all_symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, mm_per_symbol))
        .zip(next_id..next_id + mm_distributed_count as u64)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec))) as Box<dyn Agent>
        })
        .collect();
    next_id += mm_distributed_count as u64;

    // Remainder MMs (randomly assigned)
    let remainder_mm_specs: Vec<_> = (0..mm_remainder)
        .map(|_| all_symbols.choose(&mut rng).unwrap())
        .collect();
    let remainder_mms: Vec<_> = remainder_mm_specs
        .iter()
        .zip(next_id..next_id + mm_remainder as u64)
        .map(|(spec, id)| {
            Box::new(MarketMaker::new(AgentId(id), make_mm_config(spec))) as Box<dyn Agent>
        })
        .collect();
    next_id += mm_remainder as u64;

    // Noise Traders (distributed across symbols)
    let nt_per_symbol = config.num_noise_traders / num_symbols;
    let nt_remainder = config.num_noise_traders % num_symbols;
    let nt_distributed_count = nt_per_symbol * num_symbols;

    let distributed_nts: Vec<_> = all_symbols
        .iter()
        .flat_map(|spec| std::iter::repeat_n(spec, nt_per_symbol))
        .zip(next_id..next_id + nt_distributed_count as u64)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                &config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += nt_distributed_count as u64;

    // Remainder noise traders
    let remainder_nt_specs: Vec<_> = (0..nt_remainder)
        .map(|_| all_symbols.choose(&mut rng).unwrap())
        .collect();
    let remainder_nts: Vec<_> = remainder_nt_specs
        .iter()
        .zip(next_id..next_id + nt_remainder as u64)
        .map(|(spec, id)| {
            create_agent(
                Tier1AgentType::NoiseTrader,
                id,
                &config,
                &spec.symbol,
                spec.initial_price,
            )
        })
        .collect();
    next_id += nt_remainder as u64;

    // Quant strategies: randomly assigned to symbols
    let quant_agent_counts = [
        (Tier1AgentType::MomentumTrader, config.num_momentum_traders),
        (Tier1AgentType::TrendFollower, config.num_trend_followers),
        (Tier1AgentType::MacdTrader, config.num_macd_traders),
        (
            Tier1AgentType::BollingerTrader,
            config.num_bollinger_traders,
        ),
        (Tier1AgentType::VwapExecutor, config.num_vwap_executors),
    ];

    let quant_specs: Vec<_> = quant_agent_counts
        .iter()
        .flat_map(|(agent_type, count)| std::iter::repeat_n(*agent_type, *count))
        .map(|agent_type| (agent_type, all_symbols.choose(&mut rng).unwrap()))
        .collect();

    let quant_agents: Vec<_> = quant_specs
        .iter()
        .zip(next_id..next_id + quant_specs.len() as u64)
        .map(|((agent_type, spec), id)| {
            create_agent(*agent_type, id, &config, &spec.symbol, spec.initial_price)
        })
        .collect();
    next_id += quant_specs.len() as u64;

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Fill to Tier 1 minimum with random agents (random symbol)
    // ─────────────────────────────────────────────────────────────────────────

    let random_count = config.random_tier1_count();
    let random_specs: Vec<_> = (0..random_count)
        .map(|_| {
            (
                Tier1AgentType::random(&mut rng),
                all_symbols.choose(&mut rng).unwrap(),
            )
        })
        .collect();

    let random_agents: Vec<_> = random_specs
        .iter()
        .zip(next_id..next_id + random_count as u64)
        .map(|((agent_type, spec), id)| {
            create_agent(*agent_type, id, &config, &spec.symbol, spec.initial_price)
        })
        .collect();
    next_id += random_count as u64;

    // Add all Tier 1 agents to simulation (imperative: sim.add_agent is side-effectful)
    let all_tier1_agents: Vec<_> = distributed_mms
        .into_iter()
        .chain(remainder_mms)
        .chain(distributed_nts)
        .chain(remainder_nts)
        .chain(quant_agents)
        .chain(random_agents)
        .collect();

    for agent in all_tier1_agents {
        sim.add_agent(agent);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 3: Spawn Tier 2 Reactive Agents (V3.2)
    // ─────────────────────────────────────────────────────────────────────────

    spawn_tier2_agents(&mut sim, &mut next_id, &config, &all_symbols, &mut rng);

    // Price history for chart (V2.3: per-symbol)
    let symbols: Vec<Symbol> = config
        .get_symbols()
        .iter()
        .map(|s| s.symbol.clone())
        .collect();

    // Initialize price history with initial prices
    let mut price_history: HashMap<Symbol, Vec<f64>> = config
        .get_symbols()
        .iter()
        .map(|spec| {
            let mut history = Vec::with_capacity(config.max_price_history);
            history.push(spec.initial_price.to_float());
            (spec.symbol.clone(), history)
        })
        .collect();

    // Tier counts for TUI display
    let tier1_count = config.total_tier1_agents();
    let tier2_count = config.num_tier2_agents;

    // Send initial state before starting (so TUI has something to display)
    let _ = tx.send(build_update(
        &sim,
        &price_history,
        false,
        tier1_count,
        tier2_count,
    ));

    // Simulation control state - starts paused, press Space to run
    let mut running = false;
    let mut tick = 0u64;
    let total_ticks = config.total_ticks;

    // Main simulation loop with start/stop control
    loop {
        // Check for commands (non-blocking)
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                SimCommand::Start => running = true,
                SimCommand::Pause => running = false,
                SimCommand::Toggle => running = !running,
                SimCommand::Quit => {
                    // Send final update and exit
                    let _ = tx.send(build_update(
                        &sim,
                        &price_history,
                        true,
                        tier1_count,
                        tier2_count,
                    ));
                    return;
                }
            }
        }

        // If paused, sleep briefly and continue (don't burn CPU)
        if !running {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Check if we've reached max ticks
        if tick >= total_ticks {
            // Send final update
            let _ = tx.send(build_update(
                &sim,
                &price_history,
                true,
                tier1_count,
                tier2_count,
            ));
            // Keep checking for quit command
            loop {
                match cmd_rx.recv() {
                    Ok(SimCommand::Quit) | Err(_) => return,
                    _ => {}
                }
            }
        }

        // Step simulation
        sim.step();
        tick += 1;

        // Update price history for each symbol (V2.3)
        // (imperative: complex conditional mutation of price_history)
        for symbol in &symbols {
            if let Some(book) = sim.get_book(symbol)
                && let Some(price) = book.last_price()
            {
                let history = price_history.entry(symbol.clone()).or_default();
                history.push(price.to_float());
                if history.len() > config.max_price_history {
                    history.remove(0);
                }
            }
        }

        // Send update to TUI
        let update = build_update(&sim, &price_history, false, tier1_count, tier2_count);
        if tx.send(update).is_err() {
            // TUI closed, exit
            break;
        }

        // Delay for visualization
        if config.tick_delay_ms > 0 {
            thread::sleep(Duration::from_millis(config.tick_delay_ms));
        }
    }
}

fn main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Configuration - Edit here or use presets!
    // ─────────────────────────────────────────────────────────────────────────

    // Default configuration
    let config = SimConfig::default();

    // Or use a preset:
    // let config = SimConfig::demo();           // Quick 1000-tick demo
    // let config = SimConfig::stress_test();    // 100K ticks, 55 agents
    // let config = SimConfig::low_activity();   // Calm market
    // let config = SimConfig::high_volatility(); // Wild swings

    // Or customize with builder pattern:
    // let config = SimConfig::new()
    //     .symbol("GOOG")
    //     .initial_price(150.0)
    //     .total_ticks(10_000)
    //     .market_makers(3)
    //     .noise_traders(30)
    //     .mm_cash(2_000_000.0)
    //     .nt_cash(25_000.0);

    // Print config summary
    eprintln!("╔═══════════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Quant Trading Gym - Simulation Config                                ║");
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
    eprintln!(
        "║  Total Agents: {:5}  │  Total Cash: ${:>14.2}          ║",
        config.total_agents(),
        config.total_starting_cash().to_float()
    );
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Press Space to start simulation...");
    eprintln!();

    // Create bounded channel for updates (backpressure if TUI falls behind)
    let (tx, rx) = bounded::<SimUpdate>(100);

    // Create unbounded channel for commands (TUI → simulation)
    let (cmd_tx, cmd_rx) = bounded::<SimCommand>(10);

    // Spawn simulation thread
    let tui_frame_rate = config.tui_frame_rate;
    let sim_handle = thread::spawn(move || {
        run_simulation(tx, cmd_rx, config);
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
