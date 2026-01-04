//! Simulation configuration options.

use types::{Price, Quantity, ShortSellingConfig, SymbolConfig};

/// Configuration for the simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Symbol configuration with shares_outstanding and borrow pool settings.
    pub symbol_config: SymbolConfig,

    /// Short-selling configuration (enabled, limits, rates).
    pub short_selling: ShortSellingConfig,

    /// Number of book levels to include in snapshots.
    pub snapshot_depth: usize,

    /// Maximum number of recent trades to keep in market data.
    pub max_recent_trades: usize,

    /// Number of ticks per candle (for OHLCV aggregation).
    pub candle_interval: u64,

    /// Maximum number of candles to keep in history.
    pub max_candles: usize,

    /// Whether to validate orders against position limits.
    /// When disabled, orders are processed without constraint checks.
    pub enforce_position_limits: bool,

    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            symbol_config: SymbolConfig::default(),
            short_selling: ShortSellingConfig::disabled(),
            snapshot_depth: 10,
            max_recent_trades: 100,
            candle_interval: 10, // Candle every 10 ticks
            max_candles: 200,    // Keep 200 candles (~2000 ticks of history)
            enforce_position_limits: true,
            verbose: false,
        }
    }
}

impl SimulationConfig {
    /// Create a new configuration with the given symbol.
    pub fn new(symbol: impl Into<String>) -> Self {
        let symbol_config = SymbolConfig {
            symbol: symbol.into(),
            ..SymbolConfig::default()
        };
        Self {
            symbol_config,
            ..Default::default()
        }
    }

    /// Get the symbol name.
    pub fn symbol(&self) -> &str {
        &self.symbol_config.symbol
    }

    /// Get the initial/reference price.
    pub fn initial_price(&self) -> Price {
        self.symbol_config.initial_price
    }

    /// Set the symbol configuration.
    pub fn with_symbol_config(mut self, config: SymbolConfig) -> Self {
        self.symbol_config = config;
        self
    }

    /// Set the initial price.
    pub fn with_initial_price(mut self, price: Price) -> Self {
        self.symbol_config.initial_price = price;
        self
    }

    /// Set shares outstanding.
    pub fn with_shares_outstanding(mut self, shares: Quantity) -> Self {
        self.symbol_config.shares_outstanding = shares;
        self
    }

    /// Set short-selling configuration.
    pub fn with_short_selling(mut self, config: ShortSellingConfig) -> Self {
        self.short_selling = config;
        self
    }

    /// Enable short selling with default settings.
    pub fn with_short_selling_enabled(mut self) -> Self {
        self.short_selling = ShortSellingConfig::enabled_default();
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

    /// Enable or disable position limit enforcement.
    pub fn with_position_limits(mut self, enforce: bool) -> Self {
        self.enforce_position_limits = enforce;
        self
    }

    /// Enable verbose mode.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}
