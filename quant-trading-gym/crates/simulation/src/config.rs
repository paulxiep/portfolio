//! Simulation configuration options.

use types::Price;

/// Configuration for the simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Symbol being traded in this simulation.
    pub symbol: String,

    /// Initial/reference price for the market.
    pub initial_price: Price,

    /// Number of book levels to include in snapshots.
    pub snapshot_depth: usize,

    /// Maximum number of recent trades to keep in market data.
    pub max_recent_trades: usize,

    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            symbol: "SIM".to_string(),
            initial_price: Price::from_float(100.0),
            snapshot_depth: 10,
            max_recent_trades: 100,
            verbose: false,
        }
    }
}

impl SimulationConfig {
    /// Create a new configuration with the given symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            ..Default::default()
        }
    }

    /// Set the initial price.
    pub fn with_initial_price(mut self, price: Price) -> Self {
        self.initial_price = price;
        self
    }

    /// Set the snapshot depth.
    pub fn with_snapshot_depth(mut self, depth: usize) -> Self {
        self.snapshot_depth = depth;
        self
    }

    /// Set the maximum recent trades.
    pub fn with_max_recent_trades(mut self, max: usize) -> Self {
        self.max_recent_trades = max;
        self
    }

    /// Enable verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
