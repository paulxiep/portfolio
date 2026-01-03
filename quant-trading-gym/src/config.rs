//! Central configuration for the Quant Trading Gym simulation.
//!
//! All simulation parameters are defined here for easy tuning.

use types::{Cash, Price};

/// Master configuration for the entire simulation.
#[derive(Debug, Clone)]
pub struct SimConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Simulation Control
    // ─────────────────────────────────────────────────────────────────────────
    /// Symbol being traded.
    pub symbol: String,
    /// Initial price of the asset.
    pub initial_price: Price,
    /// Total ticks to run (0 = infinite).
    pub total_ticks: u64,
    /// Delay between ticks in milliseconds (0 = fastest).
    pub tick_delay_ms: u64,
    /// Enable verbose logging.
    pub verbose: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Agent Counts
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of market makers.
    pub num_market_makers: usize,
    /// Number of noise traders.
    pub num_noise_traders: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Market Maker Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each market maker.
    pub mm_initial_cash: Cash,
    /// Half-spread as a fraction (e.g., 0.0025 = 0.25%).
    pub mm_half_spread: f64,
    /// Number of shares to quote on each side.
    pub mm_quote_size: u64,
    /// Ticks between quote refreshes.
    pub mm_refresh_interval: u64,
    /// Maximum inventory before reducing quotes.
    pub mm_max_inventory: i64,
    /// Price adjustment per unit of inventory.
    pub mm_inventory_skew: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Noise Trader Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each noise trader.
    pub nt_initial_cash: Cash,
    /// Probability of placing an order each tick (0.0 - 1.0).
    pub nt_order_probability: f64,
    /// Maximum price deviation from mid price as a fraction.
    pub nt_price_deviation: f64,
    /// Minimum order quantity.
    pub nt_min_quantity: u64,
    /// Maximum order quantity.
    pub nt_max_quantity: u64,

    // ─────────────────────────────────────────────────────────────────────────
    // TUI Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum price history points to display.
    pub max_price_history: usize,
    /// TUI frame rate (frames per second).
    pub tui_frame_rate: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            // Simulation Control
            symbol: "ACME".to_string(),
            initial_price: Price::from_float(100.0),
            total_ticks: 5000,
            tick_delay_ms: 10, // ~100 ticks/sec for watchable visualization
            verbose: false,

            // Agent Counts
            num_market_makers: 3,
            num_noise_traders: 30,

            // Market Maker Parameters
            mm_initial_cash: Cash::from_float(1_000_000.0),
            mm_half_spread: 0.0025, // 0.25% half-spread = $0.25 on $100
            mm_quote_size: 50,
            mm_refresh_interval: 10,
            mm_max_inventory: 200,
            mm_inventory_skew: 0.001,

            // Noise Trader Parameters
            nt_initial_cash: Cash::from_float(10_000.0),
            nt_order_probability: 0.3, // 30% chance each tick
            nt_price_deviation: 0.01,  // 1% from mid price
            nt_min_quantity: 5,
            nt_max_quantity: 30,

            // TUI Parameters
            max_price_history: 200,
            tui_frame_rate: 30,
        }
    }
}

impl SimConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Builder-style setters for fluent configuration
    // ─────────────────────────────────────────────────────────────────────────

    /// Set the trading symbol.
    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = symbol.into();
        self
    }

    /// Set the initial asset price.
    pub fn initial_price(mut self, price: f64) -> Self {
        self.initial_price = Price::from_float(price);
        self
    }

    /// Set total ticks to run.
    pub fn total_ticks(mut self, ticks: u64) -> Self {
        self.total_ticks = ticks;
        self
    }

    /// Set tick delay in milliseconds.
    pub fn tick_delay_ms(mut self, ms: u64) -> Self {
        self.tick_delay_ms = ms;
        self
    }

    /// Set number of market makers.
    pub fn market_makers(mut self, count: usize) -> Self {
        self.num_market_makers = count;
        self
    }

    /// Set number of noise traders.
    pub fn noise_traders(mut self, count: usize) -> Self {
        self.num_noise_traders = count;
        self
    }

    /// Set market maker initial cash.
    pub fn mm_cash(mut self, cash: f64) -> Self {
        self.mm_initial_cash = Cash::from_float(cash);
        self
    }

    /// Set noise trader initial cash.
    pub fn nt_cash(mut self, cash: f64) -> Self {
        self.nt_initial_cash = Cash::from_float(cash);
        self
    }

    /// Set market maker half-spread.
    pub fn mm_spread(mut self, half_spread: f64) -> Self {
        self.mm_half_spread = half_spread;
        self
    }

    /// Set noise trader order probability.
    pub fn nt_probability(mut self, prob: f64) -> Self {
        self.nt_order_probability = prob;
        self
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Computed Properties
    // ─────────────────────────────────────────────────────────────────────────

    /// Total number of agents.
    pub fn total_agents(&self) -> usize {
        self.num_market_makers + self.num_noise_traders
    }

    /// Total starting cash in the system.
    pub fn total_starting_cash(&self) -> Cash {
        let mm_total =
            Cash::from_float(self.mm_initial_cash.to_float() * self.num_market_makers as f64);
        let nt_total =
            Cash::from_float(self.nt_initial_cash.to_float() * self.num_noise_traders as f64);
        mm_total + nt_total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset Configurations
// ─────────────────────────────────────────────────────────────────────────────

impl SimConfig {
    /// Quick demo: fewer ticks, faster visualization.
    pub fn demo() -> Self {
        Self::default().total_ticks(1000).tick_delay_ms(5)
    }

    /// Stress test: many agents, many ticks, no delay.
    pub fn stress_test() -> Self {
        Self::default()
            .total_ticks(100_000)
            .tick_delay_ms(0)
            .market_makers(5)
            .noise_traders(50)
    }

    /// Low activity: conservative parameters.
    pub fn low_activity() -> Self {
        Self::default().noise_traders(5).nt_probability(0.1)
    }

    /// High volatility: aggressive noise traders.
    pub fn high_volatility() -> Self {
        Self::default()
            .nt_probability(0.5)
            .nt_cash(50_000.0)
            .mm_spread(0.005) // Wider spread to compensate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_consistency() {
        // Test that default config is internally consistent.
        // Don't check specific values - those may change.
        let config = SimConfig::default();

        // total_agents() should match sum of individual counts
        assert_eq!(
            config.total_agents(),
            config.num_market_makers + config.num_noise_traders
        );

        // Sanity checks - defaults should be reasonable, not exact values
        assert!(config.num_market_makers >= 1, "Should have at least 1 MM");
        assert!(config.num_noise_traders >= 1, "Should have at least 1 NT");
        assert!(config.total_ticks > 0, "Should run at least 1 tick");
        assert!(
            config.initial_price > Price::ZERO,
            "Initial price should be positive"
        );
    }

    #[test]
    fn test_builder_pattern() {
        let config = SimConfig::new()
            .market_makers(7)
            .noise_traders(42)
            .total_ticks(9999);

        // Builder should set the values we specified
        assert_eq!(config.num_market_makers, 7);
        assert_eq!(config.num_noise_traders, 42);
        assert_eq!(config.total_ticks, 9999);
        assert_eq!(config.total_agents(), 49);
    }

    #[test]
    fn test_total_starting_cash() {
        let config = SimConfig::new()
            .market_makers(2)
            .noise_traders(20)
            .mm_cash(1_000_000.0)
            .nt_cash(10_000.0);

        // 2 * 1M + 20 * 10K = 2M + 200K = 2.2M
        assert_eq!(config.total_starting_cash(), Cash::from_float(2_200_000.0));
    }

    #[test]
    fn test_preset_configs_differ_from_default() {
        let default = SimConfig::default();
        let demo = SimConfig::demo();
        let stress = SimConfig::stress_test();
        let low = SimConfig::low_activity();
        let high = SimConfig::high_volatility();

        // Presets should modify at least one parameter from default
        assert_ne!(demo.total_ticks, default.total_ticks);
        assert_ne!(stress.tick_delay_ms, default.tick_delay_ms);
        assert_ne!(low.num_noise_traders, default.num_noise_traders);
        assert_ne!(high.nt_order_probability, default.nt_order_probability);
    }
}
