//! Centralized ML prediction cache for V5.6 optimization.
//!
//! This module provides [`MlPredictionCache`] which stores extracted features
//! and model predictions for the current tick. This enables O(S) feature extractions
//! and O(M × S) predictions instead of O(N) per-agent computations.
//!
//! # Architecture
//!
//! ```text
//! Phase 3 (StrategyContext Builder)
//!     │
//!     ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │            MlPredictionCache                                 │
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │ features: HashMap<Symbol, [f64; 42]>                     ││
//! │  │   - Extract ONCE per symbol per tick                     ││
//! │  └─────────────────────────────────────────────────────────┘│
//! │  ┌─────────────────────────────────────────────────────────┐│
//! │  │ predictions: HashMap<(ModelName, Symbol), [f64; 3]>      ││
//! │  │   - Predict ONCE per (model, symbol) per tick            ││
//! │  └─────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut cache = MlPredictionCache::new(tick);
//!
//! // Insert features for a symbol
//! cache.insert_features("ACME".into(), features);
//!
//! // Insert prediction for a model-symbol pair
//! cache.insert_prediction("decision_tree_1", &"ACME".into(), [0.1, 0.2, 0.7]);
//!
//! // Retrieve cached prediction
//! if let Some(probs) = cache.get_prediction("decision_tree_1", &"ACME".into()) {
//!     // Use cached prediction
//! }
//! ```

use std::collections::HashMap;
use types::{N_MARKET_FEATURES, Symbol, Tick};

use crate::tier1::ml::ClassProbabilities;

/// Key for prediction cache: (model_name, symbol).
pub type PredictionKey = (String, Symbol);

/// Centralized cache for ML features and predictions.
///
/// Created once per tick and populated during Phase 3 of the tick loop.
/// Agents can then retrieve cached predictions instead of computing them locally.
#[derive(Debug, Clone, Default)]
pub struct MlPredictionCache {
    /// Tick this cache was built for (for staleness detection).
    pub tick: Tick,

    /// Extracted features per symbol, computed once.
    features: HashMap<Symbol, [f64; N_MARKET_FEATURES]>,

    /// Cached predictions per (model_name, symbol) pair.
    predictions: HashMap<PredictionKey, ClassProbabilities>,
}

impl MlPredictionCache {
    /// Create a new empty cache for the given tick.
    pub fn new(tick: Tick) -> Self {
        Self {
            tick,
            features: HashMap::new(),
            predictions: HashMap::new(),
        }
    }

    /// Insert extracted features for a symbol.
    pub fn insert_features(&mut self, symbol: Symbol, features: [f64; N_MARKET_FEATURES]) {
        self.features.insert(symbol, features);
    }

    /// Get cached features for a symbol.
    pub fn get_features(&self, symbol: &Symbol) -> Option<&[f64; N_MARKET_FEATURES]> {
        self.features.get(symbol)
    }

    /// Insert a prediction for a model-symbol pair.
    pub fn insert_prediction(
        &mut self,
        model_name: &str,
        symbol: &Symbol,
        probs: ClassProbabilities,
    ) {
        let key = (model_name.to_string(), symbol.clone());
        self.predictions.insert(key, probs);
    }

    /// Get cached prediction for a model-symbol pair.
    pub fn get_prediction(&self, model_name: &str, symbol: &Symbol) -> Option<ClassProbabilities> {
        let key = (model_name.to_string(), symbol.clone());
        self.predictions.get(&key).copied()
    }

    /// Check if features are cached for a symbol.
    pub fn has_features(&self, symbol: &Symbol) -> bool {
        self.features.contains_key(symbol)
    }

    /// Check if prediction is cached for a model-symbol pair.
    pub fn has_prediction(&self, model_name: &str, symbol: &Symbol) -> bool {
        let key = (model_name.to_string(), symbol.clone());
        self.predictions.contains_key(&key)
    }

    /// Get the number of cached symbols.
    pub fn symbol_count(&self) -> usize {
        self.features.len()
    }

    /// Get the number of cached predictions.
    pub fn prediction_count(&self) -> usize {
        self.predictions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = MlPredictionCache::new(100);
        assert_eq!(cache.tick, 100);
        assert_eq!(cache.symbol_count(), 0);
        assert_eq!(cache.prediction_count(), 0);
    }

    #[test]
    fn test_features_insert_and_get() {
        let mut cache = MlPredictionCache::new(100);
        let features = [1.0; N_MARKET_FEATURES];
        let symbol = "ACME".to_string();

        assert!(!cache.has_features(&symbol));

        cache.insert_features(symbol.clone(), features);

        assert!(cache.has_features(&symbol));
        assert_eq!(cache.get_features(&symbol), Some(&features));
        assert_eq!(cache.symbol_count(), 1);
    }

    #[test]
    fn test_predictions_insert_and_get() {
        let mut cache = MlPredictionCache::new(100);
        let probs: ClassProbabilities = [0.1, 0.2, 0.7];
        let symbol = "ACME".to_string();
        let model = "test_model";

        assert!(!cache.has_prediction(model, &symbol));

        cache.insert_prediction(model, &symbol, probs);

        assert!(cache.has_prediction(model, &symbol));
        assert_eq!(cache.get_prediction(model, &symbol), Some(probs));
        assert_eq!(cache.prediction_count(), 1);
    }

    #[test]
    fn test_multiple_models_same_symbol() {
        let mut cache = MlPredictionCache::new(100);
        let symbol = "ACME".to_string();

        cache.insert_prediction("model_a", &symbol, [0.3, 0.3, 0.4]);
        cache.insert_prediction("model_b", &symbol, [0.1, 0.1, 0.8]);

        assert_eq!(
            cache.get_prediction("model_a", &symbol),
            Some([0.3, 0.3, 0.4])
        );
        assert_eq!(
            cache.get_prediction("model_b", &symbol),
            Some([0.1, 0.1, 0.8])
        );
        assert_eq!(cache.prediction_count(), 2);
    }

    #[test]
    fn test_same_model_multiple_symbols() {
        let mut cache = MlPredictionCache::new(100);
        let model = "test_model";

        cache.insert_prediction(model, &"SYM_A".to_string(), [0.3, 0.3, 0.4]);
        cache.insert_prediction(model, &"SYM_B".to_string(), [0.1, 0.1, 0.8]);

        assert_eq!(
            cache.get_prediction(model, &"SYM_A".to_string()),
            Some([0.3, 0.3, 0.4])
        );
        assert_eq!(
            cache.get_prediction(model, &"SYM_B".to_string()),
            Some([0.1, 0.1, 0.8])
        );
    }
}
