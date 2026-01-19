//! Unified feature schema for ML training and inference.
//!
//! This module provides a single source of truth for feature extraction,
//! eliminating training-serving skew risk by sharing constants and pure
//! computation functions between storage (training) and agents (inference).
//!
//! # Design Philosophy
//!
//! - **Declarative**: Feature schema defined via constants, not imperative code
//! - **Pure Functions**: All computations are side-effect-free for testability
//! - **Type-Safe Indices**: Named constants prevent magic number bugs
//!
//! # Usage
//!
//! ```ignore
//! use types::features::{idx, LOOKBACKS, N_MARKET_FEATURES, price_change_from_candles};
//!
//! let mut features = [f64::NAN; N_MARKET_FEATURES];
//! features[idx::MID_PRICE] = mid_price;
//! features[idx::PRICE_CHANGE_START + i] = price_change_from_candles(candles, LOOKBACKS[i]);
//! ```

use crate::{Candle, IndicatorType};

// =============================================================================
// Constants
// =============================================================================

/// Lookback periods for price changes and log returns.
/// Geometric spread optimized for batch auction tick intervals.
pub const LOOKBACKS: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/// Number of lookback periods.
pub const N_LOOKBACKS: usize = 12;

/// Total number of market-level features.
pub const N_MARKET_FEATURES: usize = 42;

// =============================================================================
// Feature Indices (Type-Safe Access)
// =============================================================================

/// Named feature indices for type-safe array access.
///
/// Use these instead of magic numbers to prevent training-serving skew.
pub mod idx {
    /// Mid price at current tick.
    pub const MID_PRICE: usize = 0;

    /// Start of price change features (12 values for each lookback).
    pub const PRICE_CHANGE_START: usize = 1;

    /// Start of log return features (12 values for each lookback).
    pub const LOG_RETURN_START: usize = 13;

    /// SMA with 8-tick period.
    pub const SMA_8: usize = 25;
    /// SMA with 16-tick period.
    pub const SMA_16: usize = 26;
    /// EMA with 8-tick period.
    pub const EMA_8: usize = 27;
    /// EMA with 16-tick period.
    pub const EMA_16: usize = 28;
    /// RSI with 8-tick period.
    pub const RSI_8: usize = 29;

    /// MACD line (fast EMA - slow EMA).
    pub const MACD_LINE: usize = 30;
    /// MACD signal line (EMA of MACD line).
    pub const MACD_SIGNAL: usize = 31;
    /// MACD histogram (line - signal).
    pub const MACD_HISTOGRAM: usize = 32;

    /// Bollinger upper band.
    pub const BB_UPPER: usize = 33;
    /// Bollinger middle band (SMA).
    pub const BB_MIDDLE: usize = 34;
    /// Bollinger lower band.
    pub const BB_LOWER: usize = 35;
    /// Bollinger %B (normalized position within bands).
    pub const BB_PERCENT_B: usize = 36;

    /// ATR with 8-tick period.
    pub const ATR_8: usize = 37;

    /// Binary indicator for active news event.
    pub const HAS_ACTIVE_NEWS: usize = 38;
    /// News sentiment (-1 to +1).
    pub const NEWS_SENTIMENT: usize = 39;
    /// News magnitude (impact strength).
    pub const NEWS_MAGNITUDE: usize = 40;
    /// Ticks remaining in news event.
    pub const NEWS_TICKS_REMAINING: usize = 41;
}

// =============================================================================
// Feature Names (For Parquet Schema & Model Introspection)
// =============================================================================

/// Market feature names (42 total).
///
/// Order matches feature indices for direct array indexing.
pub const MARKET_FEATURE_NAMES: &[&str] = &[
    // Price features (25) - geometric lookbacks: 1,2,3,4,6,8,12,16,24,32,48,64
    "f_mid_price",
    "f_price_change_1",
    "f_price_change_2",
    "f_price_change_3",
    "f_price_change_4",
    "f_price_change_6",
    "f_price_change_8",
    "f_price_change_12",
    "f_price_change_16",
    "f_price_change_24",
    "f_price_change_32",
    "f_price_change_48",
    "f_price_change_64",
    "f_log_return_1",
    "f_log_return_2",
    "f_log_return_3",
    "f_log_return_4",
    "f_log_return_6",
    "f_log_return_8",
    "f_log_return_12",
    "f_log_return_16",
    "f_log_return_24",
    "f_log_return_32",
    "f_log_return_48",
    "f_log_return_64",
    // Technical indicators (13) - from quant crate, 8/16 spread
    "f_sma_8",
    "f_sma_16",
    "f_ema_8",
    "f_ema_16",
    "f_rsi_8",
    "f_macd_line",
    "f_macd_signal",
    "f_macd_histogram",
    "f_bb_upper",
    "f_bb_middle",
    "f_bb_lower",
    "f_bb_percent_b",
    "f_atr_8",
    // News/sentiment features (4)
    "f_has_active_news",
    "f_news_sentiment",
    "f_news_magnitude",
    "f_news_ticks_remaining",
];

// =============================================================================
// Pure Computation Functions
// =============================================================================

/// Compute price change percentage.
///
/// Returns `(current - past) / past * 100`, or NaN if past <= 0.
#[inline]
pub fn price_change_pct(current: f64, past: f64) -> f64 {
    if past > 0.0 {
        (current - past) / past * 100.0
    } else {
        f64::NAN
    }
}

/// Compute log return.
///
/// Returns `ln(current / past)`, or NaN if either price is non-positive.
#[inline]
pub fn log_return(current: f64, past: f64) -> f64 {
    if current > 0.0 && past > 0.0 {
        (current / past).ln()
    } else {
        f64::NAN
    }
}

/// Compute Bollinger %B (normalized position within bands).
///
/// Returns `(price - lower) / (upper - lower)`, or NaN if inputs invalid.
#[inline]
pub fn bollinger_percent_b(price: f64, upper: f64, lower: f64) -> f64 {
    if upper.is_finite() && lower.is_finite() && price.is_finite() {
        let width = upper - lower;
        if width > 0.0 {
            (price - lower) / width
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

/// Compute price change from candle history.
///
/// Looks back `lookback` candles from the most recent and computes percentage change.
/// Returns NaN if insufficient history.
pub fn price_change_from_candles(candles: &[Candle], lookback: usize) -> f64 {
    if candles.len() < lookback + 1 {
        return f64::NAN;
    }
    let current = candles[candles.len() - 1].close.to_float();
    let past = candles[candles.len() - 1 - lookback].close.to_float();
    price_change_pct(current, past)
}

/// Compute log return from candle history.
///
/// Looks back `lookback` candles from the most recent and computes log return.
/// Returns NaN if insufficient history.
pub fn log_return_from_candles(candles: &[Candle], lookback: usize) -> f64 {
    if candles.len() < lookback + 1 {
        return f64::NAN;
    }
    let current = candles[candles.len() - 1].close.to_float();
    let past = candles[candles.len() - 1 - lookback].close.to_float();
    log_return(current, past)
}

// =============================================================================
// Required Indicators
// =============================================================================

/// Returns the set of technical indicators required for feature extraction.
///
/// Use this to configure the indicator engine with exactly what's needed.
pub fn required_indicators() -> [IndicatorType; 12] {
    [
        IndicatorType::Sma(8),
        IndicatorType::Sma(16),
        IndicatorType::Ema(8),
        IndicatorType::Ema(16),
        IndicatorType::Rsi(8),
        IndicatorType::MACD_LINE_STANDARD,
        IndicatorType::MACD_SIGNAL_STANDARD,
        IndicatorType::MACD_HISTOGRAM_STANDARD,
        IndicatorType::BOLLINGER_UPPER_STANDARD,
        IndicatorType::BOLLINGER_MIDDLE_STANDARD,
        IndicatorType::BOLLINGER_LOWER_STANDARD,
        IndicatorType::Atr(8),
    ]
}

// =============================================================================
// Compile-Time Assertions
// =============================================================================

/// Static assertions to catch schema mismatches at compile time.
const _: () = {
    assert!(MARKET_FEATURE_NAMES.len() == N_MARKET_FEATURES);
    assert!(LOOKBACKS.len() == N_LOOKBACKS);
    // Verify final index matches count
    assert!(idx::NEWS_TICKS_REMAINING == N_MARKET_FEATURES - 1);
};

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Price;

    #[test]
    fn test_price_change_pct() {
        // 10% increase
        assert!((price_change_pct(110.0, 100.0) - 10.0).abs() < 1e-10);
        // 10% decrease
        assert!((price_change_pct(90.0, 100.0) - (-10.0)).abs() < 1e-10);
        // Zero past price returns NaN
        assert!(price_change_pct(100.0, 0.0).is_nan());
        // Negative past price returns NaN
        assert!(price_change_pct(100.0, -100.0).is_nan());
    }

    #[test]
    fn test_log_return() {
        // ln(1.1) ≈ 0.0953
        assert!((log_return(110.0, 100.0) - 0.1_f64.ln_1p()).abs() < 1e-10);
        // ln(0.9) ≈ -0.1054
        assert!((log_return(90.0, 100.0) - (-0.1_f64).ln_1p()).abs() < 1e-10);
        // Invalid inputs return NaN
        assert!(log_return(0.0, 100.0).is_nan());
        assert!(log_return(100.0, 0.0).is_nan());
        assert!(log_return(-100.0, 100.0).is_nan());
    }

    #[test]
    fn test_bollinger_percent_b() {
        // Price at lower band = 0.0
        assert!((bollinger_percent_b(100.0, 120.0, 100.0) - 0.0).abs() < 1e-10);
        // Price at upper band = 1.0
        assert!((bollinger_percent_b(120.0, 120.0, 100.0) - 1.0).abs() < 1e-10);
        // Price at middle = 0.5
        assert!((bollinger_percent_b(110.0, 120.0, 100.0) - 0.5).abs() < 1e-10);
        // Invalid bands return NaN
        assert!(bollinger_percent_b(100.0, 100.0, 100.0).is_nan()); // zero width
        assert!(bollinger_percent_b(f64::NAN, 120.0, 100.0).is_nan());
    }

    #[test]
    fn test_price_change_from_candles() {
        use crate::Quantity;

        let candles: Vec<Candle> = (0..10)
            .map(|i| Candle {
                symbol: "TEST".to_string(),
                timestamp: i as u64,
                tick: i as u64,
                open: Price::from_float(100.0 + i as f64),
                high: Price::from_float(101.0 + i as f64),
                low: Price::from_float(99.0 + i as f64),
                close: Price::from_float(100.0 + i as f64),
                volume: Quantity(1000),
            })
            .collect();

        // Latest close = 109, lookback 1 = 108 => (109-108)/108 * 100
        let expected = (109.0 - 108.0) / 108.0 * 100.0;
        assert!((price_change_from_candles(&candles, 1) - expected).abs() < 1e-10);

        // Insufficient history
        assert!(price_change_from_candles(&candles, 100).is_nan());
    }

    #[test]
    fn test_feature_count_consistency() {
        assert_eq!(MARKET_FEATURE_NAMES.len(), N_MARKET_FEATURES);
        assert_eq!(LOOKBACKS.len(), N_LOOKBACKS);
    }

    #[test]
    fn test_feature_name_prefix() {
        for name in MARKET_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Feature name '{}' should start with 'f_'",
                name
            );
        }
    }

    #[test]
    fn test_required_indicators() {
        let indicators = required_indicators();
        assert_eq!(indicators.len(), 12);

        // Check we have all expected types
        assert!(indicators.contains(&IndicatorType::Sma(8)));
        assert!(indicators.contains(&IndicatorType::Rsi(8)));
        assert!(indicators.contains(&IndicatorType::MACD_LINE_STANDARD));
        assert!(indicators.contains(&IndicatorType::BOLLINGER_UPPER_STANDARD));
    }
}
