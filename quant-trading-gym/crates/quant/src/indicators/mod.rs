//! Technical indicators for market analysis.
//!
//! This module provides a trait-based interface for computing technical indicators
//! on price candle data. All indicators are designed to work with [`Candle`] slices
//! and produce f64 values suitable for statistical analysis.
//!
//! # Supported Indicators
//! - **SMA** - Simple Moving Average
//! - **EMA** - Exponential Moving Average
//! - **RSI** - Relative Strength Index
//! - **MACD** - Moving Average Convergence Divergence
//! - **Bollinger Bands** - Volatility bands around SMA
//! - **ATR** - Average True Range
//!
//! # Example
//! ```
//! use quant::indicators::{Indicator, Sma};
//! use types::{Candle, Price, Quantity};
//!
//! let candles: Vec<Candle> = vec![/* ... */];
//! let sma = Sma::new(20);
//! if let Some(value) = sma.calculate(&candles) {
//!     println!("SMA(20) = {:.2}", value);
//! }
//! ```

use types::{Candle, IndicatorType};

// =============================================================================
// Indicator Modules
// =============================================================================

mod atr;
mod bollinger;
mod ema;
mod macd;
mod rsi;
mod sma;

// =============================================================================
// Re-exports
// =============================================================================

pub use atr::Atr;
pub use bollinger::BollingerBands;
pub use ema::Ema;
pub use macd::Macd;
pub use rsi::Rsi;
pub use sma::Sma;

// =============================================================================
// Indicator Trait
// =============================================================================

/// Trait for technical indicators.
///
/// Indicators consume candle data and produce a single f64 value.
/// They declare their type (for caching) and minimum required data periods.
pub trait Indicator: Send + Sync {
    /// The type of this indicator (for caching and identification).
    fn indicator_type(&self) -> IndicatorType;

    /// Calculate the indicator value from candle data.
    ///
    /// Returns `None` if there's insufficient data.
    /// Candles are expected to be ordered from oldest to newest.
    fn calculate(&self, candles: &[Candle]) -> Option<f64>;

    /// Minimum number of candles required for a valid calculation.
    fn required_periods(&self) -> usize;
}

// =============================================================================
// Factory Function
// =============================================================================

/// Create an indicator from its type specification.
pub fn create_indicator(indicator_type: IndicatorType) -> Box<dyn Indicator> {
    match indicator_type {
        IndicatorType::Sma(p) => Box::new(Sma::new(p)),
        IndicatorType::Ema(p) => Box::new(Ema::new(p)),
        IndicatorType::Rsi(p) => Box::new(Rsi::new(p)),
        IndicatorType::Macd { fast, slow, signal } => Box::new(Macd::new(fast, slow, signal)),
        IndicatorType::BollingerBands { period, std_dev_bp } => {
            Box::new(BollingerBands::new(period, std_dev_bp as f64 / 100.0))
        }
        IndicatorType::Atr(p) => Box::new(Atr::new(p)),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use types::{Price, Quantity};

    /// Helper to create test candles with given close prices.
    fn make_candles(closes: &[f64]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(close),
                high: Price::from_float(close + 1.0),
                low: Price::from_float(close - 1.0),
                close: Price::from_float(close),
                volume: Quantity(1000),
                timestamp: i as u64,
                tick: i as u64,
            })
            .collect()
    }

    #[test]
    fn test_sma_calculation() {
        let candles = make_candles(&[10.0, 11.0, 12.0, 13.0, 14.0]);
        let sma = Sma::new(3);

        // SMA(3) of last 3 values: (12 + 13 + 14) / 3 = 13
        let result = sma.calculate(&candles);
        assert!((result.unwrap() - 13.0).abs() < 0.001);
    }

    #[test]
    fn test_sma_insufficient_data() {
        let candles = make_candles(&[10.0, 11.0]);
        let sma = Sma::new(5);
        assert!(sma.calculate(&candles).is_none());
    }

    #[test]
    fn test_ema_calculation() {
        let candles = make_candles(&[
            22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
        ]);
        let ema = Ema::new(10);

        let result = ema.calculate(&candles);
        // EMA(10) with these values should be around 22.22
        assert!(result.is_some());
        assert!((result.unwrap() - 22.221).abs() < 0.01);
    }

    #[test]
    fn test_rsi_calculation() {
        // Test data that should produce RSI around 70
        let prices: Vec<f64> = (0..20)
            .map(|i| 44.0 + i as f64 * 0.2 + (i % 3) as f64 * 0.1)
            .collect();
        let candles = make_candles(&prices);
        let rsi = Rsi::new(14);

        let result = rsi.calculate(&candles);
        assert!(result.is_some());
        // RSI should be positive and <= 100
        let rsi_val = result.unwrap();
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
    }

    #[test]
    fn test_rsi_boundaries() {
        // Test with only gains (RSI should be 100)
        let increasing = make_candles(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let rsi = Rsi::new(14);
        let result = rsi.calculate(&increasing);
        assert!(result.is_some());
        assert!((result.unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_macd_standard() {
        // Need at least 26 + 9 = 35 candles
        let prices: Vec<f64> = (0..40)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let candles = make_candles(&prices);
        let macd = Macd::standard();

        let result = macd.calculate_full(&candles);
        assert!(result.is_some());

        let output = result.unwrap();
        // MACD line should be the difference of fast and slow EMAs
        // Histogram should be MACD - Signal
        assert!((output.histogram - (output.macd_line - output.signal_line)).abs() < 0.0001);
    }

    #[test]
    fn test_bollinger_bands() {
        let candles = make_candles(&[
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
            45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
        ]);
        let bb = BollingerBands::standard();

        let result = bb.calculate_full(&candles);
        assert!(result.is_some());

        let output = result.unwrap();
        // Upper band should be > middle > lower
        assert!(output.upper > output.middle);
        assert!(output.middle > output.lower);
        // %B should be between 0 and 1 for price within bands
        assert!(output.percent_b >= -0.5 && output.percent_b <= 1.5);
    }

    #[test]
    fn test_atr_calculation() {
        // Create candles with varying ranges
        let candles: Vec<Candle> = (0..20)
            .map(|i| {
                let base = 100.0 + i as f64;
                Candle {
                    symbol: "TEST".to_string(),
                    open: Price::from_float(base),
                    high: Price::from_float(base + 2.0),
                    low: Price::from_float(base - 1.0),
                    close: Price::from_float(base + 0.5),
                    volume: Quantity(1000),
                    timestamp: i as u64,
                    tick: i as u64,
                }
            })
            .collect();

        let atr = Atr::new(14);
        let result = atr.calculate(&candles);
        assert!(result.is_some());

        // ATR should be positive
        let atr_val = result.unwrap();
        assert!(atr_val > 0.0);
    }

    #[test]
    fn test_indicator_factory() {
        let sma = create_indicator(IndicatorType::Sma(20));
        assert_eq!(sma.required_periods(), 20);

        // V5.3: MACD_STANDARD is (8, 16, 4) → required = 16 + 4 = 20
        let macd = create_indicator(IndicatorType::MACD_STANDARD);
        assert_eq!(macd.required_periods(), 20);

        // V5.3: BOLLINGER_STANDARD is period=12
        let bb = create_indicator(IndicatorType::BOLLINGER_STANDARD);
        assert_eq!(bb.required_periods(), 12);
    }
}
