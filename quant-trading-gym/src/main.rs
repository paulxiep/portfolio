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
//! │   (Thread A)   │    (channel)       │   (Thread B)   │
//! └────────────────┘                    └────────────────┘
//! ```

mod config;

use std::collections::HashMap;
use std::thread;
use std::time::Duration;

use agents::{
    BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig, MarketMaker,
    MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    TrendFollower, TrendFollowerConfig, VwapExecutor, VwapExecutorConfig,
};
use crossbeam_channel::{Sender, bounded};
use rand::prelude::IndexedRandom;
use simulation::{Simulation, SimulationConfig};
use tui::{AgentInfo, RiskInfo, SimUpdate, TuiApp};
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
fn build_update(
    sim: &Simulation,
    price_history: &HashMap<Symbol, Vec<f64>>,
    finished: bool,
) -> SimUpdate {
    let stats = sim.stats();
    let symbols: Vec<Symbol> = sim.config().symbols();
    let primary_symbol = sim.config().symbol().to_string();

    // Get mark price from primary symbol for equity calculation
    let mark_price = sim
        .book()
        .last_price()
        .map(|p| p.to_float())
        .unwrap_or(100.0);

    // Calculate digit width for agent numbering
    let agent_summaries = sim.agent_summaries();
    let num_agents = agent_summaries.len();
    let width = digit_width(num_agents);

    // Build agent info from simulation (V2.3: per-symbol positions)
    let agents: Vec<AgentInfo> = agent_summaries
        .iter()
        .enumerate()
        .map(|(i, (name, position, cash))| {
            let is_mm = name.contains("Market");
            let equity = cash.to_float() + (*position as f64 * mark_price);

            // Build positions HashMap (single symbol for now - agents are single-symbol)
            let mut positions = HashMap::new();
            positions.insert(primary_symbol.clone(), *position);

            AgentInfo {
                name: format!("{:0width$}-{}", i + 1, name, width = width),
                positions,
                realized_pnl: Cash::ZERO, // TODO: track realized P&L
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
        .filter_map(|(i, (name, _, _))| {
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
    let mut bids_map = HashMap::new();
    let mut asks_map = HashMap::new();
    let mut last_price_map = HashMap::new();

    for symbol in &symbols {
        if let Some(book) = sim.get_book(symbol) {
            let snapshot = book.snapshot(sim.timestamp(), sim.tick(), 10);
            bids_map.insert(symbol.clone(), snapshot.bids);
            asks_map.insert(symbol.clone(), snapshot.asks);
            if let Some(price) = book.last_price() {
                last_price_map.insert(symbol.clone(), price);
            }
        }
    }

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
        total_trades: stats.total_trades,
        total_orders: stats.total_orders,
        finished,
        risk_metrics,
    }
}

/// Spawn a single agent of the given type with the next available ID.
///
/// For multi-symbol support: symbol and initial_price are passed explicitly
/// so agents can be spawned per-symbol (market makers, noise traders) or
/// assigned to the primary symbol (quant strategies).
fn spawn_agent(
    sim: &mut Simulation,
    agent_type: Tier1AgentType,
    next_id: &mut u64,
    config: &SimConfig,
    symbol: &str,
    initial_price: Price,
) {
    let id = AgentId(*next_id);
    *next_id += 1;

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
            sim.add_agent(Box::new(NoiseTrader::new(id, nt_config)));
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
            sim.add_agent(Box::new(MomentumTrader::new(id, momentum_config)));
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
            sim.add_agent(Box::new(TrendFollower::new(id, trend_config)));
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
            sim.add_agent(Box::new(MacdCrossover::new(id, macd_config)));
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
            sim.add_agent(Box::new(BollingerReversion::new(id, bollinger_config)));
        }
        Tier1AgentType::VwapExecutor => {
            let vwap_config = VwapExecutorConfig {
                symbol: symbol.to_string(),
                initial_price,
                initial_cash: config.quant_initial_cash,
                ..Default::default()
            };
            sim.add_agent(Box::new(VwapExecutor::new(id, vwap_config)));
        }
    }
}

/// Run the simulation, sending updates to the TUI via channel.
fn run_simulation(tx: Sender<SimUpdate>, config: SimConfig) {
    // Build symbol configs from SimConfig symbols (V2.3)
    let symbol_configs: Vec<SymbolConfig> = config
        .get_symbols()
        .iter()
        .map(|spec| {
            SymbolConfig::new(
                &spec.symbol,
                Quantity(10_000_000), // 10M shares outstanding
                spec.initial_price,
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

    // Helper to pick a random symbol
    let pick_symbol = |rng: &mut rand::prelude::ThreadRng| -> &SymbolSpec {
        all_symbols.choose(rng).unwrap()
    };

    // Market Makers (infrastructure, distributed across symbols)
    // First ensure each symbol gets at least num_market_makers / num_symbols
    // Then randomly assign remainder
    let mm_per_symbol = config.num_market_makers / num_symbols;
    let mm_remainder = config.num_market_makers % num_symbols;

    for spec in &all_symbols {
        for _ in 0..mm_per_symbol {
            let mm_config = MarketMakerConfig {
                symbol: spec.symbol.clone(),
                initial_price: spec.initial_price,
                half_spread: config.mm_half_spread,
                quote_size: config.mm_quote_size,
                refresh_interval: config.mm_refresh_interval,
                max_inventory: config.mm_max_inventory,
                inventory_skew: config.mm_inventory_skew,
                initial_cash: config.mm_initial_cash,
                initial_position: 500, // Start with inventory from the float
            };
            sim.add_agent(Box::new(MarketMaker::new(AgentId(next_id), mm_config)));
            next_id += 1;
        }
    }
    // Randomly assign remainder
    for _ in 0..mm_remainder {
        let spec = pick_symbol(&mut rng);
        let mm_config = MarketMakerConfig {
            symbol: spec.symbol.clone(),
            initial_price: spec.initial_price,
            half_spread: config.mm_half_spread,
            quote_size: config.mm_quote_size,
            refresh_interval: config.mm_refresh_interval,
            max_inventory: config.mm_max_inventory,
            inventory_skew: config.mm_inventory_skew,
            initial_cash: config.mm_initial_cash,
            initial_position: 500,
        };
        sim.add_agent(Box::new(MarketMaker::new(AgentId(next_id), mm_config)));
        next_id += 1;
    }

    // Noise Traders (distributed across symbols)
    let nt_per_symbol = config.num_noise_traders / num_symbols;
    let nt_remainder = config.num_noise_traders % num_symbols;

    for spec in &all_symbols {
        for _ in 0..nt_per_symbol {
            spawn_agent(
                &mut sim,
                Tier1AgentType::NoiseTrader,
                &mut next_id,
                &config,
                &spec.symbol,
                spec.initial_price,
            );
        }
    }
    // Randomly assign remainder
    for _ in 0..nt_remainder {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::NoiseTrader,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // Quant strategies: randomly assigned to symbols (equal chance per symbol)

    // Momentum Traders (RSI)
    for _ in 0..config.num_momentum_traders {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::MomentumTrader,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // Trend Followers (SMA crossover)
    for _ in 0..config.num_trend_followers {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::TrendFollower,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // MACD Traders
    for _ in 0..config.num_macd_traders {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::MacdTrader,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // Bollinger Traders
    for _ in 0..config.num_bollinger_traders {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::BollingerTrader,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // VWAP Executors
    for _ in 0..config.num_vwap_executors {
        let spec = pick_symbol(&mut rng);
        spawn_agent(
            &mut sim,
            Tier1AgentType::VwapExecutor,
            &mut next_id,
            &config,
            &spec.symbol,
            spec.initial_price,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Fill to Tier 1 minimum with random agents (random symbol)
    // ─────────────────────────────────────────────────────────────────────────

    let random_count = config.random_tier1_count();
    if random_count > 0 {
        for _ in 0..random_count {
            let agent_type = Tier1AgentType::random(&mut rng);
            let spec = pick_symbol(&mut rng);
            spawn_agent(
                &mut sim,
                agent_type,
                &mut next_id,
                &config,
                &spec.symbol,
                spec.initial_price,
            );
        }
    }

    // Price history for chart (V2.3: per-symbol)
    let symbols: Vec<Symbol> = config
        .get_symbols()
        .iter()
        .map(|s| s.symbol.clone())
        .collect();
    let mut price_history: HashMap<Symbol, Vec<f64>> = HashMap::new();
    for symbol in &symbols {
        price_history.insert(symbol.clone(), Vec::with_capacity(config.max_price_history));
    }

    // Run simulation loop
    for _tick in 0..config.total_ticks {
        // Step simulation
        sim.step();

        // Update price history for each symbol (V2.3)
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
        let update = build_update(&sim, &price_history, false);
        if tx.send(update).is_err() {
            // TUI closed, exit
            break;
        }

        // Delay for visualization
        if config.tick_delay_ms > 0 {
            thread::sleep(Duration::from_millis(config.tick_delay_ms));
        }
    }

    // Send final update
    let _ = tx.send(build_update(&sim, &price_history, true));

    // Keep channel open briefly so TUI can display final state
    thread::sleep(Duration::from_secs(1));
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
        "║  Tier 1 Minimum: {:2}  │  Random Fill: {:2}  │  Total: {:2}              ║",
        config.min_tier1_agents,
        config.random_tier1_count(),
        config.total_agents()
    );
    eprintln!(
        "║  Total Starting Cash: ${:>14.2}                              ║",
        config.total_starting_cash().to_float()
    );
    eprintln!("╚═══════════════════════════════════════════════════════════════════════╝");

    // Create bounded channel (backpressure if TUI falls behind)
    let (tx, rx) = bounded::<SimUpdate>(100);

    // Spawn simulation thread
    let tui_frame_rate = config.tui_frame_rate;
    let sim_handle = thread::spawn(move || {
        run_simulation(tx, config);
    });

    // Run TUI in main thread (required for terminal control)
    let app = TuiApp::new(rx).frame_rate(tui_frame_rate);
    if let Err(e) = app.run() {
        eprintln!("TUI error: {}", e);
    }

    // Wait for simulation to finish
    let _ = sim_handle.join();
}
