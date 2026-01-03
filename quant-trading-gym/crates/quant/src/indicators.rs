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

use types::{BollingerOutput, Candle, IndicatorType, MacdOutput};

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
// Simple Moving Average (SMA)
// =============================================================================

/// Simple Moving Average indicator.
///
/// Computes the arithmetic mean of the closing prices over a specified period.
#[derive(Debug, Clone)]
pub struct Sma {
    period: usize,
}

impl Sma {
    /// Create a new SMA indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "SMA period must be > 0");
        Self { period }
    }

    /// Calculate SMA from a slice of closing prices (f64).
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period || period == 0 {
            return None;
        }
        let sum: f64 = prices.iter().rev().take(period).sum();
        Some(sum / period as f64)
    }
}

impl Indicator for Sma {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Sma(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        if candles.len() < self.period {
            return None;
        }

        let sum: f64 = candles
            .iter()
            .rev()
            .take(self.period)
            .map(|c| c.close.to_float())
            .sum();

        Some(sum / self.period as f64)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}

// =============================================================================
// Exponential Moving Average (EMA)
// =============================================================================

/// Exponential Moving Average indicator.
///
/// Gives more weight to recent prices using exponential smoothing.
/// Multiplier = 2 / (period + 1)
#[derive(Debug, Clone)]
pub struct Ema {
    period: usize,
    multiplier: f64,
}

impl Ema {
    /// Create a new EMA indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "EMA period must be > 0");
        Self {
            period,
            multiplier: 2.0 / (period as f64 + 1.0),
        }
    }

    /// Calculate EMA from a slice of prices.
    /// Uses SMA of first `period` values as the initial EMA value.
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        if prices.len() < period || period == 0 {
            return None;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initial EMA is SMA of first `period` values
        let initial_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;

        // Apply EMA formula to remaining values
        let ema = prices
            .iter()
            .skip(period)
            .fold(initial_sma, |prev_ema, price| {
                (price - prev_ema) * multiplier + prev_ema
            });

        Some(ema)
    }
}

impl Indicator for Ema {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Ema(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        if candles.len() < self.period {
            return None;
        }

        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();

        // Initial EMA is SMA of first `period` values
        let initial_sma: f64 = prices.iter().take(self.period).sum::<f64>() / self.period as f64;

        // Apply EMA formula to remaining values
        let ema = prices
            .iter()
            .skip(self.period)
            .fold(initial_sma, |prev_ema, price| {
                (price - prev_ema) * self.multiplier + prev_ema
            });

        Some(ema)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}

// =============================================================================
// Relative Strength Index (RSI)
// =============================================================================

/// Relative Strength Index indicator.
///
/// Measures the speed and change of price movements on a 0-100 scale.
/// RSI > 70 is typically considered overbought, < 30 oversold.
#[derive(Debug, Clone)]
pub struct Rsi {
    period: usize,
}

impl Rsi {
    /// Create a new RSI indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "RSI period must be > 0");
        Self { period }
    }

    /// Calculate RSI from a slice of prices using Wilder's smoothing method.
    pub fn calculate_from_prices(prices: &[f64], period: usize) -> Option<f64> {
        // Need at least period + 1 prices for period changes
        if prices.len() < period + 1 || period == 0 {
            return None;
        }

        // Calculate price changes
        let changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        // Separate gains and losses
        let (mut avg_gain, mut avg_loss) =
            changes
                .iter()
                .take(period)
                .fold((0.0, 0.0), |(g, l), &change| {
                    if change > 0.0 {
                        (g + change, l)
                    } else {
                        (g, l - change)
                    }
                });

        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // Wilder's smoothing for remaining periods
        for &change in changes.iter().skip(period) {
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };
            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
        }

        // Calculate RSI
        if avg_loss == 0.0 {
            Some(100.0) // No losses = max RSI
        } else {
            let rs = avg_gain / avg_loss;
            Some(100.0 - (100.0 / (1.0 + rs)))
        }
    }
}

impl Indicator for Rsi {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Rsi(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Need period + 1 candles for period price changes
        if candles.len() < self.period + 1 {
            return None;
        }

        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();
        Rsi::calculate_from_prices(&prices, self.period)
    }

    fn required_periods(&self) -> usize {
        self.period + 1
    }
}

// =============================================================================
// MACD (Moving Average Convergence Divergence)
// =============================================================================

/// MACD indicator.
///
/// Shows the relationship between two EMAs and includes a signal line.
/// Standard configuration is (12, 26, 9).
#[derive(Debug, Clone)]
pub struct Macd {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl Macd {
    /// Create a new MACD indicator with custom periods.
    ///
    /// # Panics
    /// Panics if any period is 0 or if fast_period >= slow_period.
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        assert!(fast_period > 0, "MACD fast period must be > 0");
        assert!(slow_period > 0, "MACD slow period must be > 0");
        assert!(signal_period > 0, "MACD signal period must be > 0");
        assert!(
            fast_period < slow_period,
            "MACD fast period must be < slow period"
        );
        Self {
            fast_period,
            slow_period,
            signal_period,
        }
    }

    /// Create MACD with standard (12, 26, 9) configuration.
    pub fn standard() -> Self {
        Self::new(12, 26, 9)
    }

    /// Calculate full MACD output including signal line and histogram.
    pub fn calculate_full(&self, candles: &[Candle]) -> Option<MacdOutput> {
        let prices: Vec<f64> = candles.iter().map(|c| c.close.to_float()).collect();
        self.calculate_full_from_prices(&prices)
    }

    /// Calculate full MACD output from price data.
    pub fn calculate_full_from_prices(&self, prices: &[f64]) -> Option<MacdOutput> {
        // Need enough data for slow EMA + signal EMA
        if prices.len() < self.slow_period + self.signal_period {
            return None;
        }

        // Calculate MACD line at each point after slow_period
        let mut macd_values: Vec<f64> = Vec::with_capacity(prices.len() - self.slow_period + 1);

        for i in self.slow_period..=prices.len() {
            let slice = &prices[..i];
            let fast_ema = Ema::calculate_from_prices(slice, self.fast_period)?;
            let slow_ema = Ema::calculate_from_prices(slice, self.slow_period)?;
            macd_values.push(fast_ema - slow_ema);
        }

        if macd_values.len() < self.signal_period {
            return None;
        }

        // Calculate signal line (EMA of MACD values)
        let signal_line = Ema::calculate_from_prices(&macd_values, self.signal_period)?;
        let macd_line = *macd_values.last()?;
        let histogram = macd_line - signal_line;

        Some(MacdOutput {
            macd_line,
            signal_line,
            histogram,
        })
    }
}

impl Indicator for Macd {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Macd {
            fast: self.fast_period,
            slow: self.slow_period,
            signal: self.signal_period,
        }
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Returns just the MACD line value for trait compatibility
        self.calculate_full(candles).map(|m| m.macd_line)
    }

    fn required_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }
}

// =============================================================================
// Bollinger Bands
// =============================================================================

/// Bollinger Bands indicator.
///
/// Volatility bands placed above and below a moving average.
/// Default is 20-period SMA with 2 standard deviations.
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    std_dev_multiplier: f64,
}

impl BollingerBands {
    /// Create new Bollinger Bands with custom parameters.
    ///
    /// # Arguments
    /// * `period` - SMA period for middle band
    /// * `std_dev_multiplier` - Number of standard deviations for bands (typically 2.0)
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize, std_dev_multiplier: f64) -> Self {
        assert!(period > 0, "Bollinger period must be > 0");
        Self {
            period,
            std_dev_multiplier,
        }
    }

    /// Create Bollinger Bands with standard (20, 2.0) configuration.
    pub fn standard() -> Self {
        Self::new(20, 2.0)
    }

    /// Calculate full Bollinger Bands output.
    pub fn calculate_full(&self, candles: &[Candle]) -> Option<BollingerOutput> {
        if candles.len() < self.period {
            return None;
        }

        let prices: Vec<f64> = candles
            .iter()
            .rev()
            .take(self.period)
            .map(|c| c.close.to_float())
            .collect();

        let mean = prices.iter().sum::<f64>() / self.period as f64;

        let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.period as f64;
        let std_dev = variance.sqrt();

        let upper = mean + (std_dev * self.std_dev_multiplier);
        let lower = mean - (std_dev * self.std_dev_multiplier);
        let current_price = candles.last()?.close.to_float();

        // Band width as percentage of middle band
        let bandwidth = if mean != 0.0 {
            (upper - lower) / mean * 100.0
        } else {
            0.0
        };

        // %B: where is price relative to bands (0 = lower, 1 = upper)
        let percent_b = if upper != lower {
            (current_price - lower) / (upper - lower)
        } else {
            0.5
        };

        Some(BollingerOutput {
            upper,
            middle: mean,
            lower,
            bandwidth,
            percent_b,
        })
    }
}

impl Indicator for BollingerBands {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::BollingerBands {
            period: self.period,
            std_dev_bp: (self.std_dev_multiplier * 100.0) as u32,
        }
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Returns middle band (SMA) for trait compatibility
        self.calculate_full(candles).map(|b| b.middle)
    }

    fn required_periods(&self) -> usize {
        self.period
    }
}

// =============================================================================
// Average True Range (ATR)
// =============================================================================

/// Average True Range indicator.
///
/// Measures market volatility by decomposing the entire range of price movement.
/// True Range is max of: High-Low, |High-PrevClose|, |Low-PrevClose|
#[derive(Debug, Clone)]
pub struct Atr {
    period: usize,
}

impl Atr {
    /// Create a new ATR indicator with the given period.
    ///
    /// # Panics
    /// Panics if period is 0.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "ATR period must be > 0");
        Self { period }
    }

    /// Calculate True Range for a candle given the previous close.
    fn true_range(candle: &Candle, prev_close: f64) -> f64 {
        let high = candle.high.to_float();
        let low = candle.low.to_float();

        let hl = high - low;
        let hpc = (high - prev_close).abs();
        let lpc = (low - prev_close).abs();

        hl.max(hpc).max(lpc)
    }
}

impl Indicator for Atr {
    fn indicator_type(&self) -> IndicatorType {
        IndicatorType::Atr(self.period)
    }

    fn calculate(&self, candles: &[Candle]) -> Option<f64> {
        // Need period + 1 candles for period true ranges
        if candles.len() < self.period + 1 {
            return None;
        }

        // Calculate true ranges
        let mut true_ranges: Vec<f64> = Vec::with_capacity(candles.len() - 1);
        for i in 1..candles.len() {
            let prev_close = candles[i - 1].close.to_float();
            true_ranges.push(Self::true_range(&candles[i], prev_close));
        }

        // Calculate initial ATR as simple average
        let initial_atr: f64 =
            true_ranges.iter().take(self.period).sum::<f64>() / self.period as f64;

        // Apply Wilder's smoothing (same as RSI)
        let atr = true_ranges
            .iter()
            .skip(self.period)
            .fold(initial_atr, |prev_atr, &tr| {
                (prev_atr * (self.period as f64 - 1.0) + tr) / self.period as f64
            });

        Some(atr)
    }

    fn required_periods(&self) -> usize {
        self.period + 1
    }
}

// =============================================================================
// Factory Functions
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

        let macd = create_indicator(IndicatorType::MACD_STANDARD);
        assert_eq!(macd.required_periods(), 35);

        let bb = create_indicator(IndicatorType::BOLLINGER_STANDARD);
        assert_eq!(bb.required_periods(), 20);
    }
}
