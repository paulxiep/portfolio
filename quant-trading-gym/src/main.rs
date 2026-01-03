//! Quant Trading Gym - Main binary
//!
//! Runs a simulation to verify the Week 2 implementation.

use agents::{Agent, AgentAction, MarketData};
use simulation::{Simulation, SimulationConfig};
use types::AgentId;

/// A simple passive agent for testing the simulation loop.
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

fn main() {
    println!("=== Quant Trading Gym - Week 2: The Loop ===\n");

    // Create simulation with default config
    let config = SimulationConfig::new("SIM")
        .with_initial_price(types::Price::from_float(100.0))
        .with_verbose(true);

    let mut sim = Simulation::new(config);

    // Add some passive agents
    for i in 1..=10 {
        sim.add_agent(Box::new(PassiveAgent { id: AgentId(i) }));
    }

    println!("Created simulation with {} agents", sim.agent_count());
    println!("Running 1000 ticks...\n");

    // Run the simulation
    let trades = sim.run(1000);

    // Report results
    let stats = sim.stats();
    println!("=== Simulation Complete ===");
    println!("Final tick: {}", stats.tick);
    println!("Total trades: {}", stats.total_trades);
    println!("Total orders: {}", stats.total_orders);
    println!("Filled orders: {}", stats.filled_orders);
    println!("Resting orders: {}", stats.resting_orders);
    println!("Trades returned: {}", trades.len());

    println!("\n✓ Empty simulation ran 1000 ticks without panic!");
    println!("✓ Week 2 exit criteria met.");
}
