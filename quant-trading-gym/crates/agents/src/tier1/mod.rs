//! Tier 1: Smart Agents - Full-context agents that run every tick.
//!
//! Tier 1 agents receive complete market context via `StrategyContext` and
//! can perform complex computations including indicator calculations.
//!
//! # Characteristics
//! - Run `on_tick()` every simulation tick
//! - Full access to `StrategyContext` (candles, indicators, trades, events)
//! - Can compute rolling indicators (SMA, EMA, RSI, MACD, Bollinger)
//! - Suitable for 10-100 agents per simulation
//! - ~3KB memory per agent
//!
//! # Module Structure
//! - `strategies/` - Concrete strategy implementations
//! - `ml/` - Tree-based ML model agents (V5.5)

pub mod ml;
pub mod strategies;

// Re-export all strategies at tier1 level for convenience
pub use strategies::{
    BollingerReversion, BollingerReversionConfig, MacdCrossover, MacdCrossoverConfig, MarketMaker,
    MarketMakerConfig, MomentumConfig, MomentumTrader, NoiseTrader, NoiseTraderConfig,
    PairsTrading, PairsTradingConfig, TrendFollower, TrendFollowerConfig, VwapExecutor,
    VwapExecutorConfig,
};

// Re-export ML model types (V5.5)
// V5.6: Added ModelRegistry and extract_features for centralized prediction caching
pub use ml::{
    ClassProbabilities, DecisionTree, DecisionTreeAgent, GradientBoosted, GradientBoostedAgent,
    MlModel, ModelRegistry, RandomForest, RandomForestAgent, TreeAgent, TreeAgentConfig,
    extract_features,
};
