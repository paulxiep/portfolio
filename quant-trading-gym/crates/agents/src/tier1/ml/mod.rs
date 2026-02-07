//! Tree-based ML model inference for Tier 1 agents (V5.5).
//!
//! This module provides Rust inference for trained sklearn tree-based models:
//! - [`DecisionTree`] - single classification tree
//! - [`RandomForest`] - ensemble of classification trees, predictions averaged
//! - [`GradientBoosted`] - staged ensemble of regression trees with learning rate
//!
//! All models implement the [`MlModel`] trait and produce class probabilities
//! for 3-class classification: sell (-1), hold (0), buy (1).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    TreeAgent<M: MlModel>                    │
//! │  ┌─────────────────┐  ┌─────────────────┐                   │
//! │  │ AgentState      │  │ M: MlModel      │                   │
//! │  └─────────────────┘  └─────────────────┘                   │
//! │                                                              │
//! │  on_tick():                                                  │
//! │    1. Extract 42 features from StrategyContext               │
//! │    2. model.predict(features) → [p_sell, p_hold, p_buy]      │
//! │    3. Compare to thresholds → generate order                 │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use agents::tier1::ml::{DecisionTree, TreeAgent, TreeAgentConfig};
//!
//! let model = DecisionTree::from_json("models/shallow_decision_tree.json")?;
//! let config = TreeAgentConfig {
//!     symbols: vec!["ACME".into()],
//!     buy_threshold: 0.55,
//!     sell_threshold: 0.55,
//!     order_size: 50,
//!     max_position_per_symbol: 500,
//!     initial_cash: Cash::from_float(100_000.0),
//! };
//! let agent = TreeAgent::new(AgentId(100), model, config);
//! ```

mod decision_tree;
mod feature_extractor;
mod full_features;
mod gradient_boosted;
pub mod group_extractors;
mod model_registry;
mod random_forest;
mod tree_agent;

pub use decision_tree::DecisionTree;
pub use feature_extractor::{MinimalFeatures, extract_features, extract_features_raw};
pub use full_features::FullFeatures;
pub use gradient_boosted::GradientBoosted;
pub use model_registry::ModelRegistry;
pub use random_forest::RandomForest;
pub use tree_agent::{TreeAgent, TreeAgentConfig};

/// Class probabilities: [p_sell, p_hold, p_buy] for classes [-1, 0, 1].
pub type ClassProbabilities = [f64; 3];

/// Trait for ML models that produce class probabilities.
///
/// Implementors must be `Send + Sync` for parallel agent execution.
pub trait MlModel: Send + Sync {
    /// Predict class probabilities from 42 market features.
    ///
    /// # Arguments
    /// * `features` - Array of 42 features in canonical order (see comprehensive_features.rs)
    ///
    /// # Returns
    /// `[p_sell, p_hold, p_buy]` probabilities that sum to 1.0.
    /// Returns `[0.0, 1.0, 0.0]` (hold) if prediction fails.
    fn predict(&self, features: &[f64]) -> ClassProbabilities;

    /// Model name for logging and debugging.
    fn name(&self) -> &str;

    /// Number of features expected (should be 42).
    fn n_features(&self) -> usize {
        42
    }
}

/// Trait for extracting market-level features from simulation state.
///
/// Implementors produce a feature vector from `StrategyContext` for a given symbol.
/// Market features are shared across agents (cacheable). Per-agent features
/// (portfolio state) are NOT part of this trait — they are computed locally.
///
/// # Pipeline
///
/// Extraction is **pure** — `extract_market()` returns raw features with NaN
/// for missing values. Imputation is a separate step using `neutral_values()`.
/// The runner applies imputation before caching:
///
/// ```ignore
/// let raw = extractor.extract_market(symbol, ctx);
/// let neutrals = extractor.neutral_values();
/// let imputed: FeatureVec = raw.iter().zip(neutrals).map(|(f, n)| {
///     if f.is_nan() { *n } else { *f }
/// }).collect();
/// cache.insert_features(symbol, imputed);
/// ```
pub trait FeatureExtractor: Send + Sync {
    /// Number of features this extractor produces.
    fn n_features(&self) -> usize;

    /// Extract raw market features for a symbol. NaN values preserved.
    fn extract_market(
        &self,
        symbol: &types::Symbol,
        ctx: &crate::StrategyContext<'_>,
    ) -> crate::ml_cache::FeatureVec;

    /// Feature names in extraction order (for Parquet schema, logging).
    fn feature_names(&self) -> &[&str];

    /// Per-feature neutral values for NaN imputation.
    ///
    /// Length must equal `n_features()`. Each value is the "no signal" default
    /// for that feature when data is missing (e.g. RSI → 50, vol_ratio → 1.0).
    fn neutral_values(&self) -> &[f64];

    /// Feature registry providing metadata (groups, valid ranges, descriptors).
    ///
    /// Used by downstream consumers: V6.2 SHAP analysis (group names),
    /// V6.3 gym (observation space bounds), V7.2 deep RL (normalization ranges).
    fn registry(&self) -> &'static types::FeatureRegistry;
}

/// Apply per-feature NaN imputation using neutral values from an extractor.
///
/// Replaces NaN values in `features` with the corresponding neutral value.
/// This is the single imputation point in the pipeline — called by the runner
/// after extraction and before cache insertion.
#[inline]
pub fn impute_features(features: &mut crate::ml_cache::FeatureVec, neutrals: &[f64]) {
    features.iter_mut().zip(neutrals).for_each(|(f, n)| {
        if f.is_nan() {
            *f = *n;
        }
    });
}

/// Type aliases for convenience.
pub type DecisionTreeAgent = TreeAgent<DecisionTree>;
pub type RandomForestAgent = TreeAgent<RandomForest>;
pub type GradientBoostedAgent = TreeAgent<GradientBoosted>;

/// Compute numerically stable softmax.
///
/// Subtracts max value before exponentiation to prevent overflow.
#[inline]
pub(crate) fn softmax(scores: &[f64]) -> ClassProbabilities {
    // Find max for numerical stability
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max) for each score
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();

    // Sum for normalization
    let sum: f64 = exp_scores.iter().sum();

    // Normalize to probabilities
    if sum > 0.0 && sum.is_finite() {
        [
            exp_scores[0] / sum,
            exp_scores[1] / sum,
            exp_scores[2] / sum,
        ]
    } else {
        // Fallback to uniform hold if softmax fails
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let scores = [1.0, 2.0, 3.0];
        let probs = softmax(&scores);

        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Higher score should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large scores that would overflow naive exp()
        let scores = [1000.0, 1001.0, 1002.0];
        let probs = softmax(&scores);

        // Should still sum to 1 and not be NaN/Inf
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p.is_finite()));
    }

    #[test]
    fn test_softmax_equal_scores() {
        let scores = [0.0, 0.0, 0.0];
        let probs = softmax(&scores);

        // Equal scores should give ~equal probabilities
        for p in &probs {
            assert!((*p - 1.0 / 3.0).abs() < 1e-10);
        }
    }
}
