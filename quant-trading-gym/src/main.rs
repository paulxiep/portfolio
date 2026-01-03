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

use std::thread;
use std::time::Duration;

use agents::{
    BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig, MarketMaker,
    MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    TrendFollower, TrendFollowerConfig, VwapExecutor, VwapExecutorConfig,
};
use crossbeam_channel::{Sender, bounded};
use simulation::{Simulation, SimulationConfig};
use tui::{AgentInfo, RiskInfo, SimUpdate, TuiApp};
use types::{AgentId, Cash};

pub use config::{SimConfig, Tier1AgentType};

/// Calculate the number of digits needed to display a number.
fn digit_width(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as usize + 1
    }
}

/// Build a SimUpdate from current simulation state.
fn build_update(sim: &Simulation, price_history: &[f64], finished: bool) -> SimUpdate {
    let stats = sim.stats();
    let book = sim.book();
    let snapshot = book.snapshot(sim.timestamp(), sim.tick(), 10);

    // Get mark price for equity calculation
    let mark_price = book.last_price().map(|p| p.to_float()).unwrap_or(100.0);

    // Calculate digit width for agent numbering
    let agent_summaries = sim.agent_summaries();
    let num_agents = agent_summaries.len();
    let width = digit_width(num_agents);

    // Build agent info from simulation
    let agents: Vec<AgentInfo> = agent_summaries
        .iter()
        .enumerate()
        .map(|(i, (name, position, cash))| {
            let is_mm = name.contains("Market");
            let equity = cash.to_float() + (*position as f64 * mark_price);
            AgentInfo {
                name: format!("{:0width$}-{}", i + 1, name, width = width),
                position: *position,
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

    SimUpdate {
        tick: sim.tick(),
        trades: sim.recent_trades().to_vec(),
        last_price: book.last_price(),
        price_history: price_history.to_vec(),
        bids: snapshot.bids,
        asks: snapshot.asks,
        agents,
        total_trades: stats.total_trades,
        total_orders: stats.total_orders,
        finished,
        risk_metrics,
    }
}

/// Spawn a single agent of the given type with the next available ID.
fn spawn_agent(
    sim: &mut Simulation,
    agent_type: Tier1AgentType,
    next_id: &mut u64,
    config: &SimConfig,
) {
    let id = AgentId(*next_id);
    *next_id += 1;

    match agent_type {
        Tier1AgentType::NoiseTrader => {
            let nt_config = NoiseTraderConfig {
                symbol: config.symbol.clone(),
                order_probability: config.nt_order_probability,
                initial_price: config.initial_price,
                price_deviation: config.nt_price_deviation,
                min_quantity: config.nt_min_quantity,
                max_quantity: config.nt_max_quantity,
                initial_cash: config.nt_initial_cash,
            };
            sim.add_agent(Box::new(NoiseTrader::new(id, nt_config)));
        }
        Tier1AgentType::MomentumTrader => {
            let momentum_config = MomentumConfig {
                symbol: config.symbol.clone(),
                initial_price: config.initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            sim.add_agent(Box::new(MomentumTrader::new(id, momentum_config)));
        }
        Tier1AgentType::TrendFollower => {
            let trend_config = TrendFollowerConfig {
                symbol: config.symbol.clone(),
                initial_price: config.initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            sim.add_agent(Box::new(TrendFollower::new(id, trend_config)));
        }
        Tier1AgentType::MacdTrader => {
            let macd_config = MacdCrossoverConfig {
                symbol: config.symbol.clone(),
                initial_price: config.initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            sim.add_agent(Box::new(MacdCrossover::new(id, macd_config)));
        }
        Tier1AgentType::BollingerTrader => {
            let bollinger_config = BollingerReversionConfig {
                symbol: config.symbol.clone(),
                initial_price: config.initial_price,
                initial_cash: config.quant_initial_cash,
                order_size: config.quant_order_size,
                max_position: config.quant_max_position,
                ..Default::default()
            };
            sim.add_agent(Box::new(BollingerReversion::new(id, bollinger_config)));
        }
        Tier1AgentType::VwapExecutor => {
            let vwap_config = VwapExecutorConfig {
                symbol: config.symbol.clone(),
                initial_price: config.initial_price,
                initial_cash: config.quant_initial_cash,
                ..Default::default()
            };
            sim.add_agent(Box::new(VwapExecutor::new(id, vwap_config)));
        }
    }
}

/// Run the simulation, sending updates to the TUI via channel.
fn run_simulation(tx: Sender<SimUpdate>, config: SimConfig) {
    // Create simulation from central config
    let sim_config = SimulationConfig::new(&config.symbol)
        .with_initial_price(config.initial_price)
        .with_verbose(config.verbose);

    let mut sim = Simulation::new(sim_config);
    let mut next_id: u64 = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 1: Spawn specified minimum agents for each type
    // ─────────────────────────────────────────────────────────────────────────

    // Market Makers (infrastructure, always spawn first)
    for _ in 0..config.num_market_makers {
        let mm_config = MarketMakerConfig {
            symbol: config.symbol.clone(),
            initial_price: config.initial_price,
            half_spread: config.mm_half_spread,
            quote_size: config.mm_quote_size,
            refresh_interval: config.mm_refresh_interval,
            max_inventory: config.mm_max_inventory,
            inventory_skew: config.mm_inventory_skew,
            initial_cash: config.mm_initial_cash,
        };
        sim.add_agent(Box::new(MarketMaker::new(AgentId(next_id), mm_config)));
        next_id += 1;
    }

    // Noise Traders
    for _ in 0..config.num_noise_traders {
        spawn_agent(&mut sim, Tier1AgentType::NoiseTrader, &mut next_id, &config);
    }

    // Momentum Traders (RSI)
    for _ in 0..config.num_momentum_traders {
        spawn_agent(
            &mut sim,
            Tier1AgentType::MomentumTrader,
            &mut next_id,
            &config,
        );
    }

    // Trend Followers (SMA crossover)
    for _ in 0..config.num_trend_followers {
        spawn_agent(
            &mut sim,
            Tier1AgentType::TrendFollower,
            &mut next_id,
            &config,
        );
    }

    // MACD Traders
    for _ in 0..config.num_macd_traders {
        spawn_agent(&mut sim, Tier1AgentType::MacdTrader, &mut next_id, &config);
    }

    // Bollinger Traders
    for _ in 0..config.num_bollinger_traders {
        spawn_agent(
            &mut sim,
            Tier1AgentType::BollingerTrader,
            &mut next_id,
            &config,
        );
    }

    // VWAP Executors
    for _ in 0..config.num_vwap_executors {
        spawn_agent(
            &mut sim,
            Tier1AgentType::VwapExecutor,
            &mut next_id,
            &config,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Phase 2: Fill to Tier 1 minimum with random agents
    // ─────────────────────────────────────────────────────────────────────────

    let random_count = config.random_tier1_count();
    if random_count > 0 {
        let mut rng = rand::rng();
        for _ in 0..random_count {
            let agent_type = Tier1AgentType::random(&mut rng);
            spawn_agent(&mut sim, agent_type, &mut next_id, &config);
        }
    }

    // Price history for chart
    let mut price_history: Vec<f64> = Vec::with_capacity(config.max_price_history);

    // Run simulation loop
    for _tick in 0..config.total_ticks {
        // Step simulation
        sim.step();

        // Update price history
        if let Some(price) = sim.book().last_price() {
            price_history.push(price.to_float());
            if price_history.len() > config.max_price_history {
                price_history.remove(0);
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
        config.symbol,
        config.initial_price.to_float()
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
