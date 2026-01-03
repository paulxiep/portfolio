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

use agents::{MarketMaker, MarketMakerConfig, NoiseTrader, NoiseTraderConfig};
use crossbeam_channel::{Sender, bounded};
use simulation::{Simulation, SimulationConfig};
use tui::{AgentInfo, SimUpdate, TuiApp};
use types::{AgentId, Cash};

pub use config::SimConfig;

/// Build a SimUpdate from current simulation state.
fn build_update(sim: &Simulation, price_history: &[f64], finished: bool) -> SimUpdate {
    let stats = sim.stats();
    let book = sim.book();
    let snapshot = book.snapshot(sim.timestamp(), sim.tick(), 10);

    // Build agent info from simulation
    let agents: Vec<AgentInfo> = sim
        .agent_summaries()
        .into_iter()
        .enumerate()
        .map(|(i, (name, position, cash))| AgentInfo {
            name: format!("{}-{}", name, i + 1),
            position,
            realized_pnl: Cash::ZERO, // TODO: track realized P&L
            cash,
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
    }
}

/// Run the simulation, sending updates to the TUI via channel.
fn run_simulation(tx: Sender<SimUpdate>, config: SimConfig) {
    // Create simulation from central config
    let sim_config = SimulationConfig::new(&config.symbol)
        .with_initial_price(config.initial_price)
        .with_verbose(config.verbose);

    let mut sim = Simulation::new(sim_config);

    // Add market makers (provide liquidity)
    for i in 0..config.num_market_makers {
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
        let mm = MarketMaker::new(AgentId(i as u64 + 1), mm_config);
        sim.add_agent(Box::new(mm));
    }

    // Add noise traders (generate activity)
    for i in 0..config.num_noise_traders {
        let nt_config = NoiseTraderConfig {
            symbol: config.symbol.clone(),
            order_probability: config.nt_order_probability,
            initial_price: config.initial_price,
            price_deviation: config.nt_price_deviation,
            min_quantity: config.nt_min_quantity,
            max_quantity: config.nt_max_quantity,
            initial_cash: config.nt_initial_cash,
        };
        let nt = NoiseTrader::new(
            AgentId((config.num_market_makers + i + 1) as u64),
            nt_config,
        );
        sim.add_agent(Box::new(nt));
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
    eprintln!("╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║  Quant Trading Gym - Simulation Config                         ║");
    eprintln!("╠════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Symbol: {:6}  │  Initial Price: ${:<8.2}               ║",
        config.symbol,
        config.initial_price.to_float()
    );
    eprintln!(
        "║  Ticks:  {:6}  │  Tick Delay: {}ms                         ║",
        config.total_ticks, config.tick_delay_ms
    );
    eprintln!("╠════════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Market Makers: {:2}  │  Cash: ${:>12.2} each              ║",
        config.num_market_makers,
        config.mm_initial_cash.to_float()
    );
    eprintln!(
        "║  Noise Traders: {:2}  │  Cash: ${:>12.2} each              ║",
        config.num_noise_traders,
        config.nt_initial_cash.to_float()
    );
    eprintln!(
        "║  Total Agents:  {:2}  │  Total Cash: ${:>12.2}             ║",
        config.total_agents(),
        config.total_starting_cash().to_float()
    );
    eprintln!("╚════════════════════════════════════════════════════════════════╝");

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
