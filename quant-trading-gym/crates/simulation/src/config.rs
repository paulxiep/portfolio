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

    /// Number of ticks per candle (for OHLCV aggregation).
    pub candle_interval: u64,

    /// Maximum number of candles to keep in history.
    pub max_candles: usize,

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
            candle_interval: 10, // Candle every 10 ticks
            max_candles: 200,    // Keep 200 candles (~2000 ticks of history)
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

    /// Set the candle interval (ticks per candle).
    pub fn with_candle_interval(mut self, interval: u64) -> Self {
        self.candle_interval = interval;
        self
    }

    /// Set the maximum number of candles to keep.
    pub fn with_max_candles(mut self, max: usize) -> Self {
        self.max_candles = max;
        self
    }

    /// Enable verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
