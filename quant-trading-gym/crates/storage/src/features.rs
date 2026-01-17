//! Feature extraction traits and context for ML recording.
//!
//! V5.3: Defines the feature extraction interface used by RecordingHook.
//! These traits are designed to be reusable for ML agent inference in V5.5.

use quant::AgentRiskSnapshot;
use simulation::{AgentSummary, EnrichedData, MarketSnapshot};
use types::{Symbol, Tick};

use crate::price_history::PriceHistory;

/// Context for feature extraction.
///
/// Contains all the data needed to compute features for a single agent at a single tick.
/// Uses references to avoid data duplication.
#[derive(Debug)]
pub struct FeatureContext<'a> {
    /// Current simulation tick.
    pub tick: Tick,
    /// Symbol being traded.
    pub symbol: &'a Symbol,
    /// Market state snapshot (bid/ask, mid prices).
    pub market: &'a MarketSnapshot,
    /// Enriched data (candles, indicators, fair values, news).
    pub enriched: Option<&'a EnrichedData>,
    /// Agent summary (positions, cash, pnl).
    pub agent_summary: &'a AgentSummary,
    /// Risk metrics (sharpe, drawdown, volatility).
    pub risk_snapshot: Option<&'a AgentRiskSnapshot>,
    /// Rolling price history for computing returns.
    pub price_history: &'a PriceHistory,
    /// Initial cash for normalization.
    pub initial_cash: f64,
    /// Position limit for normalization.
    pub position_limit: i64,
}

/// Trait for feature extraction strategies.
///
/// Implement this trait to define a set of features to extract from market state.
/// Used by both RecordingHook (V5.3) and ML agents (V5.5).
pub trait FeatureExtractor: Send + Sync {
    /// Return ordered list of feature names (for Parquet schema).
    ///
    /// Names should use the `f_` prefix convention for pre-tick features.
    fn feature_names(&self) -> &[&'static str];

    /// Extract features as f64 vector.
    ///
    /// The returned vector must have the same length as `feature_names()`.
    /// Use NaN for missing values.
    fn extract(&self, ctx: &FeatureContext) -> Vec<f64>;
}

/// Trait for ML model inference (V5.5 forward compatibility).
///
/// Implemented by DecisionTree, RandomForest, etc.
pub trait MlModel: Send + Sync {
    /// Predict action score from features.
    ///
    /// Returns a continuous score that can be thresholded for action selection.
    fn predict(&self, features: &[f64]) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use types::{AgentId, Cash};

    struct DummyExtractor;

    impl FeatureExtractor for DummyExtractor {
        fn feature_names(&self) -> &[&'static str] {
            &["f_dummy"]
        }

        fn extract(&self, _ctx: &FeatureContext) -> Vec<f64> {
            vec![1.0]
        }
    }

    #[test]
    fn test_dummy_extractor() {
        let extractor = DummyExtractor;
        assert_eq!(extractor.feature_names().len(), 1);
    }
}
