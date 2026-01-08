//! Central configuration for the Quant Trading Gym simulation.
//!
//! All simulation parameters are defined here for easy tuning.
//!
//! # Agent Tiers
//! - **Tier 1**: Full agents that run every tick (MarketMaker, NoiseTrader, Momentum, etc.)
//! - **Tier 2**: Reactive agents that wake on conditions (not yet implemented)
//! - **Tier 3**: Statistical background pool (not yet implemented)
//!
//! # Configuration Strategy
//! 1. Specify minimum count for each specific agent type
//! 2. Specify minimum total for each tier
//! 3. If tier minimum not met by specific agents, random tier agents are spawned
//!
//! # Multi-Symbol Support (V2.3)
//!
//! The `symbols` field allows configuring multiple symbols. Currently the simulation
//! runs the first symbol only (single-symbol mode), but the TUI infrastructure
//! supports displaying multiple symbols once the simulation is upgraded.
//!
//! **Recommended limits:**
//! - Development/testing: 1-3 symbols
//! - Production: Up to 10 symbols (performance depends on agent count per symbol)
//! - Hard limit: None, but memory grows linearly with symbols × price history

use agents::tier3::MarketRegime;
use rand::Rng;
use rand::prelude::SliceRandom;
use types::{Cash, Price, Sector};

/// Configuration for a single symbol.
#[derive(Debug, Clone)]
pub struct SymbolSpec {
    /// Symbol ticker (e.g., "AAPL", "GOOG").
    pub symbol: String,
    /// Initial price of the asset.
    pub initial_price: Price,
    /// Industry sector for news events and grouping (V2.4).
    pub sector: Sector,
}

impl SymbolSpec {
    /// Create a new symbol specification.
    pub fn new(symbol: impl Into<String>, initial_price: f64) -> Self {
        Self {
            symbol: symbol.into(),
            initial_price: Price::from_float(initial_price),
            sector: Sector::Tech, // Default sector
        }
    }

    /// Create a symbol specification with explicit sector.
    pub fn with_sector(symbol: impl Into<String>, initial_price: f64, sector: Sector) -> Self {
        Self {
            symbol: symbol.into(),
            initial_price: Price::from_float(initial_price),
            sector,
        }
    }
}

/// Master configuration for the entire simulation.
#[derive(Debug, Clone)]
pub struct SimConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Simulation Control
    // ─────────────────────────────────────────────────────────────────────────
    /// Symbols to trade (first symbol is primary for single-symbol simulation).
    pub symbols: Vec<SymbolSpec>,
    /// Total ticks to run (0 = infinite).
    pub total_ticks: u64,
    /// Delay between ticks in milliseconds (0 = fastest).
    pub tick_delay_ms: u64,
    /// Enable verbose logging.
    pub verbose: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 1 Agent Counts (minimum for each type)
    // ─────────────────────────────────────────────────────────────────────────
    /// Minimum number of market makers.
    pub num_market_makers: usize,
    /// Minimum number of noise traders.
    pub num_noise_traders: usize,
    /// Minimum number of momentum (RSI) traders.
    pub num_momentum_traders: usize,
    /// Minimum number of trend followers (SMA crossover).
    pub num_trend_followers: usize,
    /// Minimum number of MACD crossover traders.
    pub num_macd_traders: usize,
    /// Minimum number of Bollinger reversion traders.
    pub num_bollinger_traders: usize,
    /// Minimum number of VWAP executors.
    pub num_vwap_executors: usize,
    /// Minimum number of pairs trading agents (V3.3 multi-symbol).
    pub num_pairs_traders: usize,
    /// Minimum number of sector rotator agents (V3.3 multi-symbol, special Tier 2).
    pub num_sector_rotators: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier Minimums
    // ─────────────────────────────────────────────────────────────────────────
    /// Minimum total Tier 1 agents. If specific agent types don't reach this,
    /// random Tier 1 agents are spawned to fill the gap.
    pub min_tier1_agents: usize,

    /// Number of Tier 2 reactive agents (V3.2).
    /// These are lightweight event-driven agents.
    pub num_tier2_agents: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 2 Reactive Agent Parameters (V3.2)
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for each Tier 2 reactive agent.
    pub t2_initial_cash: Cash,
    /// Maximum position size for Tier 2 agents.
    pub t2_max_position: u64,
    /// ThresholdBuyer: minimum buy price (dollars).
    pub t2_buy_threshold_min: f64,
    /// ThresholdBuyer: maximum buy price (dollars).
    pub t2_buy_threshold_max: f64,
    /// StopLoss: minimum stop percentage (e.g., 0.02 = 2%).
    pub t2_stop_loss_min: f64,
    /// StopLoss: maximum stop percentage (e.g., 0.08 = 8%).
    pub t2_stop_loss_max: f64,
    /// TakeProfit: minimum target percentage (e.g., 0.10 = 10%).
    pub t2_take_profit_min: f64,
    /// TakeProfit: maximum target percentage (e.g., 0.30 = 30%).
    pub t2_take_profit_max: f64,
    /// ThresholdSeller: minimum sell price (dollars).
    pub t2_sell_threshold_min: f64,
    /// ThresholdSeller: maximum sell price (dollars).
    pub t2_sell_threshold_max: f64,
    /// Probability of having TakeProfit vs ThresholdSeller (0.0 - 1.0).
    pub t2_take_profit_prob: f64,
    /// Probability of having NewsReactor strategy (0.0 - 1.0).
    pub t2_news_reactor_prob: f64,
    /// ThresholdBuyer: minimum order size fraction (0.0 - 1.0).
    pub t2_order_size_min: f64,
    /// ThresholdBuyer: maximum order size fraction (0.0 - 1.0).
    pub t2_order_size_max: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Tier 3 Background Pool Parameters (V3.4)
    // ─────────────────────────────────────────────────────────────────────────
    /// Enable background pool (statistical order generation).
    pub enable_background_pool: bool,
    /// Number of simulated background agents (scales order rate).
    pub background_pool_size: usize,
    /// Market regime for background pool behavior.
    pub background_regime: MarketRegime,
    /// Mean order size for background pool.
    pub t3_mean_order_size: f64,
    /// Maximum order size for background pool.
    pub t3_max_order_size: u64,
    /// Order size standard deviation for background pool.
    pub t3_order_size_stddev: f64,
    /// Base activity rate override (None = use regime default).
    /// Fraction of pool that trades each tick (0.0-1.0).
    pub t3_base_activity: Option<f64>,
    /// Price spread lambda (higher = tighter around mid price).
    pub t3_price_spread_lambda: f64,
    /// Maximum price deviation from mid (as fraction).
    pub t3_max_price_deviation: f64,

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
    /// Starting position in shares (allows balanced buy/sell from start).
    pub nt_initial_position: i64,
    /// Probability of placing an order each tick (0.0 - 1.0).
    pub nt_order_probability: f64,
    /// Maximum price deviation from mid price as a fraction.
    pub nt_price_deviation: f64,
    /// Minimum order quantity.
    pub nt_min_quantity: u64,
    /// Maximum order quantity.
    pub nt_max_quantity: u64,

    // ─────────────────────────────────────────────────────────────────────────
    // Quant Strategy Parameters (shared defaults)
    // ─────────────────────────────────────────────────────────────────────────
    /// Starting cash for quant strategies.
    pub quant_initial_cash: Cash,
    /// Order size for quant strategies.
    pub quant_order_size: u64,
    /// Maximum position for quant strategies.
    pub quant_max_position: i64,

    // ─────────────────────────────────────────────────────────────────────────
    // TUI Parameters
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum price history points to display.
    pub max_price_history: usize,
    /// TUI frame rate (frames per second).
    pub tui_frame_rate: u64,

    // ─────────────────────────────────────────────────────────────────────────
    // Event/News Generation Parameters (V2.4)
    // ─────────────────────────────────────────────────────────────────────────
    /// Enable event/news generation.
    pub events_enabled: bool,
    /// Earnings event probability per tick (e.g., 0.002 = ~1 per 500 ticks).
    pub event_earnings_prob: f64,
    /// Minimum ticks between earnings events.
    pub event_earnings_interval: u64,
    /// Guidance event probability per tick (e.g., 0.001 = ~1 per 1000 ticks).
    pub event_guidance_prob: f64,
    /// Minimum ticks between guidance events.
    pub event_guidance_interval: u64,
    /// Rate decision probability per tick (e.g., 0.0005 = ~1 per 2000 ticks).
    pub event_rate_decision_prob: f64,
    /// Minimum ticks between rate decisions.
    pub event_rate_decision_interval: u64,
    /// Sector news probability per tick (e.g., 0.003 = ~1 per 333 ticks).
    pub event_sector_news_prob: f64,
    /// Minimum ticks between sector news.
    pub event_sector_news_interval: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            // Simulation Control - default multi-symbol with different sectors (V2.4)
            symbols: vec![
                SymbolSpec::with_sector("Duck Delish", 100.0, Sector::Consumer),
                SymbolSpec::with_sector("Zephyr Zap", 100.0, Sector::Utilities),
                SymbolSpec::with_sector("Vraiment Villa", 100.0, Sector::RealEstate),
                SymbolSpec::with_sector("Quant Quotation", 100.0, Sector::Finance),
                SymbolSpec::with_sector("Hello Handy", 100.0, Sector::Communications),
                SymbolSpec::with_sector("Nubes Nexus", 100.0, Sector::Tech),
            ],
            total_ticks: 10000,
            tick_delay_ms: 0, // ~100 ticks/sec for watchable visualization
            verbose: false,

            // V3.5: Doubled agent counts for parallel execution benchmarking
            // Tier 1 Agent Counts (minimums per type)
            num_market_makers: 300,
            num_noise_traders: 1500,
            num_momentum_traders: 400,
            num_trend_followers: 400,
            num_macd_traders: 400,
            num_bollinger_traders: 400,
            num_vwap_executors: 400,
            num_pairs_traders: 1200,   // V3.3: multi-symbol pairs traders
            num_sector_rotators: 2000, // V3.3: sector rotation agents (special T2)
            // Tier Minimums
            min_tier1_agents: 5000, // Random agents fill the gap

            // Tier 2 Reactive Agents (V3.2)
            num_tier2_agents: 18000,

            // Tier 2 Reactive Agent Parameters (V3.2)
            // Equal starting cash to noise traders for fair comparison
            t2_initial_cash: Cash::from_float(100_000.0),
            t2_max_position: 1000,
            t2_buy_threshold_min: 60.0, // Buy when price drops to $50-75
            t2_buy_threshold_max: 90.0,
            t2_stop_loss_min: 0.25, // StopLoss 25-50% (wider to avoid noise)
            t2_stop_loss_max: 0.5,
            t2_take_profit_min: 0.25, // TakeProfit 25-50% (aggressive targets)
            t2_take_profit_max: 0.5,
            t2_sell_threshold_min: 105.0, // ThresholdSeller $110-140
            t2_sell_threshold_max: 130.0,
            t2_take_profit_prob: 0.5,  // 50% TakeProfit, 50% ThresholdSeller
            t2_news_reactor_prob: 0.1, // 10% have NewsReactor
            t2_order_size_min: 0.2,    // Min 30% of max position per order
            t2_order_size_max: 0.5,    // Max 70% of max position per order

            // Tier 3 Background Pool (V3.4)
            // V3.5: Doubled pool size for parallel execution benchmarking
            enable_background_pool: true,
            background_pool_size: 75000,
            background_regime: MarketRegime::Normal,
            t3_mean_order_size: 25.0,
            t3_max_order_size: 100,
            t3_order_size_stddev: 10.0,
            t3_base_activity: Some(0.003), // Use regime default
            t3_price_spread_lambda: 10.0,
            t3_max_price_deviation: 0.05, // 5% from mid

            // Market Maker Parameters
            mm_initial_cash: Cash::from_float(1_000_000.0),
            mm_half_spread: 0.005, // 0.25% half-spread = $0.25 on $100
            mm_quote_size: 100,
            mm_refresh_interval: 1, // Quote every tick (required for IOC mode)
            mm_max_inventory: 200,
            mm_inventory_skew: 0.001,

            // Noise Trader Parameters
            // Noise traders start flat (0 position) to avoid adding to long imbalance
            // from market makers. Cash equals quant_initial_cash for equal net worth.
            nt_initial_cash: Cash::from_float(100_000.0),
            nt_initial_position: 0,
            nt_order_probability: 0.3, // 30% chance each tick
            nt_price_deviation: 0.02,  // 1% from mid price
            nt_min_quantity: 15,
            nt_max_quantity: 50,

            // Quant Strategy Parameters
            quant_initial_cash: Cash::from_float(100_000.0),
            quant_order_size: 35,
            quant_max_position: 200,

            // TUI Parameters
            max_price_history: 500,
            tui_frame_rate: 100,

            // Event/News Generation Parameters (V2.4)
            // Defaults match crates/news/src/config.rs
            events_enabled: true,
            event_earnings_prob: 0.006, // ~1 per 500 ticks per symbol
            event_earnings_interval: 25,
            event_guidance_prob: 0.003, // ~1 per 1000 ticks
            event_guidance_interval: 50,
            event_rate_decision_prob: 0.0015, // ~1 per 2000 ticks (rare)
            event_rate_decision_interval: 125,
            event_sector_news_prob: 0.009, // ~1 per 333 ticks
            event_sector_news_interval: 12,
        }
    }
}

/// Types of Tier 1 agents that can be randomly spawned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier1AgentType {
    NoiseTrader,
    MomentumTrader,
    TrendFollower,
    MacdTrader,
    BollingerTrader,
    VwapExecutor,
    PairsTrading, // V3.3: multi-symbol pairs trading
}

impl Tier1AgentType {
    /// All spawnable Tier 1 agent types (excludes MarketMaker as it's infrastructure).
    /// Note: PairsTrading excluded from random spawn (requires 2-symbol pairs config).
    pub const SPAWNABLE: &'static [Tier1AgentType] = &[
        Tier1AgentType::NoiseTrader,
        Tier1AgentType::MomentumTrader,
        Tier1AgentType::TrendFollower,
        Tier1AgentType::MacdTrader,
        Tier1AgentType::BollingerTrader,
        Tier1AgentType::VwapExecutor,
    ];

    /// Pick a random spawnable agent type.
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        *Self::SPAWNABLE.choose(rng).unwrap()
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

    /// Set the complete list of symbols to trade.
    pub fn symbols(mut self, symbols: Vec<SymbolSpec>) -> Self {
        self.symbols = symbols;
        self
    }

    /// Add a symbol to the trading list.
    pub fn add_symbol(mut self, symbol: impl Into<String>, initial_price: f64) -> Self {
        self.symbols.push(SymbolSpec::new(symbol, initial_price));
        self
    }

    /// Convenience: Set a single trading symbol (replaces all existing).
    /// For multi-symbol, use `symbols()` or `add_symbol()`.
    pub fn symbol(mut self, symbol: impl Into<String>) -> Self {
        if self.symbols.is_empty() {
            self.symbols.push(SymbolSpec::new(symbol, 100.0));
        } else {
            self.symbols[0].symbol = symbol.into();
        }
        self
    }

    /// Convenience: Set initial price for the first symbol.
    /// For multi-symbol, configure via `symbols()` or `SymbolSpec`.
    pub fn initial_price(mut self, price: f64) -> Self {
        if self.symbols.is_empty() {
            self.symbols.push(SymbolSpec::new("ACME", price));
        } else {
            self.symbols[0].initial_price = Price::from_float(price);
        }
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

    /// Set minimum number of market makers.
    pub fn market_makers(mut self, count: usize) -> Self {
        self.num_market_makers = count;
        self
    }

    /// Set minimum number of noise traders.
    pub fn noise_traders(mut self, count: usize) -> Self {
        self.num_noise_traders = count;
        self
    }

    /// Set minimum number of momentum traders.
    pub fn momentum_traders(mut self, count: usize) -> Self {
        self.num_momentum_traders = count;
        self
    }

    /// Set minimum number of trend followers.
    pub fn trend_followers(mut self, count: usize) -> Self {
        self.num_trend_followers = count;
        self
    }

    /// Set minimum number of MACD traders.
    pub fn macd_traders(mut self, count: usize) -> Self {
        self.num_macd_traders = count;
        self
    }

    /// Set minimum number of Bollinger traders.
    pub fn bollinger_traders(mut self, count: usize) -> Self {
        self.num_bollinger_traders = count;
        self
    }

    /// Set minimum number of VWAP executors.
    pub fn vwap_executors(mut self, count: usize) -> Self {
        self.num_vwap_executors = count;
        self
    }

    /// Set minimum number of pairs trading agents (V3.3).
    pub fn pairs_traders(mut self, count: usize) -> Self {
        self.num_pairs_traders = count;
        self
    }

    /// Set minimum number of sector rotator agents (V3.3).
    pub fn sector_rotators(mut self, count: usize) -> Self {
        self.num_sector_rotators = count;
        self
    }

    /// Set minimum total Tier 1 agents.
    pub fn min_tier1(mut self, count: usize) -> Self {
        self.min_tier1_agents = count;
        self
    }

    /// Set number of Tier 2 reactive agents (V3.2).
    pub fn tier2_agents(mut self, count: usize) -> Self {
        self.num_tier2_agents = count;
        self
    }

    /// Set Tier 2 agent initial cash (V3.2).
    pub fn t2_cash(mut self, cash: f64) -> Self {
        self.t2_initial_cash = Cash::from_float(cash);
        self
    }

    /// Set Tier 2 agent max position (V3.2).
    pub fn t2_max_position(mut self, max_pos: u64) -> Self {
        self.t2_max_position = max_pos;
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

    /// Set quant strategy initial cash.
    pub fn quant_cash(mut self, cash: f64) -> Self {
        self.quant_initial_cash = Cash::from_float(cash);
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
    // Symbol Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Get all symbol specifications.
    pub fn get_symbols(&self) -> &[SymbolSpec] {
        &self.symbols
    }

    /// Get the primary (first) symbol name.
    /// Panics if no symbols are configured.
    pub fn primary_symbol(&self) -> &str {
        &self.symbols[0].symbol
    }

    /// Get the primary (first) symbol's initial price.
    /// Panics if no symbols are configured.
    pub fn primary_initial_price(&self) -> Price {
        self.symbols[0].initial_price
    }

    /// Number of configured symbols.
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Computed Properties
    // ─────────────────────────────────────────────────────────────────────────

    /// Total number of specified (non-random) Tier 1 agents.
    pub fn specified_tier1_agents(&self) -> usize {
        self.num_market_makers
            + self.num_noise_traders
            + self.num_momentum_traders
            + self.num_trend_followers
            + self.num_macd_traders
            + self.num_bollinger_traders
            + self.num_vwap_executors
            + self.num_pairs_traders
    }

    /// Number of random Tier 1 agents to spawn to meet minimum.
    pub fn random_tier1_count(&self) -> usize {
        let specified = self.specified_tier1_agents();
        self.min_tier1_agents.saturating_sub(specified)
    }

    /// Total number of Tier 1 agents (specified + random fill).
    pub fn total_tier1_agents(&self) -> usize {
        self.specified_tier1_agents() + self.random_tier1_count()
    }

    /// Total number of agents (Tier 1 + Tier 2 + SectorRotators).
    pub fn total_agents(&self) -> usize {
        self.total_tier1_agents() + self.num_tier2_agents + self.num_sector_rotators
    }

    /// Total starting cash in the system (estimate, doesn't include random agents).
    pub fn total_starting_cash(&self) -> Cash {
        let mm_total =
            Cash::from_float(self.mm_initial_cash.to_float() * self.num_market_makers as f64);
        let nt_total =
            Cash::from_float(self.nt_initial_cash.to_float() * self.num_noise_traders as f64);
        let quant_count = self.num_momentum_traders
            + self.num_trend_followers
            + self.num_macd_traders
            + self.num_bollinger_traders;
        let quant_total = Cash::from_float(self.quant_initial_cash.to_float() * quant_count as f64);
        mm_total + nt_total + quant_total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preset Configurations
// ─────────────────────────────────────────────────────────────────────────────

impl SimConfig {
    /// Quick demo: 10% of default agents, fewer ticks, faster visualization.
    pub fn demo() -> Self {
        Self::default()
            .total_ticks(1000)
            .tick_delay_ms(5)
            .market_makers(10)
            .noise_traders(40)
            .momentum_traders(5)
            .trend_followers(5)
            .macd_traders(5)
            .bollinger_traders(5)
            .vwap_executors(5)
            .min_tier1(100)
    }

    /// Stress test: 2x default agents, many ticks, no delay.
    pub fn stress_test() -> Self {
        Self::default()
            .total_ticks(100_000)
            .tick_delay_ms(0)
            .min_tier1(2000)
    }

    /// Low activity: 20% of default agents, conservative parameters.
    pub fn low_activity() -> Self {
        Self::default()
            .market_makers(20)
            .noise_traders(80)
            .momentum_traders(10)
            .trend_followers(10)
            .macd_traders(10)
            .bollinger_traders(10)
            .vwap_executors(10)
            .nt_probability(0.1)
            .min_tier1(200)
    }

    /// High volatility: aggressive noise traders, wider spreads.
    pub fn high_volatility() -> Self {
        Self::default()
            .noise_traders(600) // 1.5x noise traders
            .nt_probability(0.5)
            .nt_cash(50_000.0)
            .mm_spread(0.005) // Wider spread to compensate
    }

    /// Quant-heavy: More algorithmic traders, fewer noise traders.
    pub fn quant_heavy() -> Self {
        Self::default()
            .noise_traders(100) // 25% of default noise
            .momentum_traders(150) // 3x quant strategies
            .trend_followers(150)
            .macd_traders(150)
            .bollinger_traders(150)
            .vwap_executors(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_consistency() {
        // Test that default config is internally consistent.
        let config = SimConfig::default();

        // specified_tier1_agents should match sum of individual counts
        let expected_specified = config.num_market_makers
            + config.num_noise_traders
            + config.num_momentum_traders
            + config.num_trend_followers
            + config.num_macd_traders
            + config.num_bollinger_traders
            + config.num_vwap_executors
            + config.num_pairs_traders;
        assert_eq!(config.specified_tier1_agents(), expected_specified);

        // Sanity checks - defaults should be reasonable
        assert!(config.num_market_makers >= 1, "Should have at least 1 MM");
        assert!(config.total_ticks > 0, "Should run at least 1 tick");
        assert!(
            config.primary_initial_price() > Price::ZERO,
            "Initial price should be positive"
        );
    }

    #[test]
    fn test_builder_pattern() {
        let config = SimConfig::new()
            .market_makers(7)
            .noise_traders(10)
            .momentum_traders(3)
            .min_tier1(25);

        assert_eq!(config.num_market_makers, 7);
        assert_eq!(config.num_noise_traders, 10);
        assert_eq!(config.num_momentum_traders, 3);
        assert_eq!(config.min_tier1_agents, 25);
    }

    #[test]
    fn test_random_tier1_fill() {
        let config = SimConfig::new()
            .market_makers(2)
            .noise_traders(3)
            .momentum_traders(0)
            .trend_followers(0)
            .macd_traders(0)
            .bollinger_traders(0)
            .vwap_executors(0)
            .pairs_traders(0)
            .min_tier1(10)
            .tier2_agents(0); // Explicitly set tier2 to 0 for this test

        assert_eq!(config.specified_tier1_agents(), 5);
        assert_eq!(config.random_tier1_count(), 5); // Need 5 more to reach 10
        assert_eq!(config.total_tier1_agents(), 10);
    }

    #[test]
    fn test_no_random_fill_when_specified_meets_minimum() {
        let config = SimConfig::new()
            .market_makers(5)
            .noise_traders(10)
            .min_tier1(10);

        assert!(config.specified_tier1_agents() >= config.min_tier1_agents);
        assert_eq!(config.random_tier1_count(), 0);
    }

    #[test]
    fn test_total_starting_cash() {
        let config = SimConfig::new()
            .market_makers(2)
            .noise_traders(10)
            .momentum_traders(1)
            .trend_followers(1)
            .macd_traders(0)
            .bollinger_traders(0)
            .mm_cash(1_000_000.0)
            .nt_cash(10_000.0)
            .quant_cash(100_000.0);

        // 2 * 1M + 10 * 10K + 2 * 100K = 2M + 100K + 200K = 2.3M
        assert_eq!(config.total_starting_cash(), Cash::from_float(2_300_000.0));
    }

    #[test]
    fn test_preset_configs_differ_from_default() {
        let default = SimConfig::default();
        let demo = SimConfig::demo();
        let stress = SimConfig::stress_test();
        let low = SimConfig::low_activity();
        let high = SimConfig::high_volatility();
        let quant = SimConfig::quant_heavy();

        // Presets should modify at least one parameter from default
        assert_ne!(demo.total_ticks, default.total_ticks);
        assert_ne!(stress.total_ticks, default.total_ticks);
        assert_ne!(low.nt_order_probability, default.nt_order_probability);
        assert_ne!(high.nt_order_probability, default.nt_order_probability);
        assert_ne!(quant.num_noise_traders, default.num_noise_traders);
    }

    #[test]
    fn test_tier1_agent_type_random() {
        let mut rng = rand::thread_rng();
        // Just verify it doesn't panic and returns valid types
        for _ in 0..10 {
            let agent_type = Tier1AgentType::random(&mut rng);
            assert!(Tier1AgentType::SPAWNABLE.contains(&agent_type));
        }
    }
}
