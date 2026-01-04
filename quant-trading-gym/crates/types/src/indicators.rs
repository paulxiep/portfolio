//! Technical indicator types for the quant module.
//!
//! This module defines the types used for technical analysis indicators
//! including moving averages, RSI, MACD, Bollinger Bands, and ATR.

use crate::ids::{Symbol, Tick};
use serde::{Deserialize, Serialize};

// =============================================================================
// Indicator Type Enum
// =============================================================================

/// Type of technical indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndicatorType {
    /// Simple Moving Average with period.
    Sma(usize),
    /// Exponential Moving Average with period.
    Ema(usize),
    /// Relative Strength Index with period.
    Rsi(usize),
    /// MACD with fast, slow, and signal periods.
    Macd {
        fast: usize,
        slow: usize,
        signal: usize,
    },
    /// Bollinger Bands with period and standard deviation multiplier (stored as basis points for precision).
    BollingerBands {
        period: usize,
        /// Standard deviation multiplier * 100 (e.g., 200 = 2.0 std devs).
        std_dev_bp: u32,
    },
    /// Average True Range with period.
    Atr(usize),
}

impl IndicatorType {
    /// Standard MACD configuration (12, 26, 9).
    pub const MACD_STANDARD: Self = Self::Macd {
        fast: 12,
        slow: 26,
        signal: 9,
    };

    /// Standard Bollinger Bands (20 period, 2 std devs).
    pub const BOLLINGER_STANDARD: Self = Self::BollingerBands {
        period: 20,
        std_dev_bp: 200,
    };

    /// Get the number of periods required for this indicator to produce valid output.
    pub fn required_periods(&self) -> usize {
        match self {
            Self::Sma(p) | Self::Ema(p) | Self::Rsi(p) | Self::Atr(p) => *p,
            Self::Macd { slow, signal, .. } => slow + signal,
            Self::BollingerBands { period, .. } => *period,
        }
    }
}

// =============================================================================
// Indicator Value
// =============================================================================

/// Computed indicator value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndicatorValue {
    /// Type of indicator.
    pub indicator_type: IndicatorType,
    /// Stock symbol.
    pub symbol: Symbol,
    /// Computed value (f64 for statistical precision).
    pub value: f64,
    /// Tick when computed.
    pub tick: Tick,
}

// =============================================================================
// MACD Output
// =============================================================================

/// MACD output values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct MacdOutput {
    /// MACD line (fast EMA - slow EMA).
    pub macd_line: f64,
    /// Signal line (EMA of MACD line).
    pub signal_line: f64,
    /// Histogram (MACD - Signal).
    pub histogram: f64,
}

// =============================================================================
// Bollinger Bands Output
// =============================================================================

/// Bollinger Bands output values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BollingerOutput {
    /// Upper band.
    pub upper: f64,
    /// Middle band (SMA).
    pub middle: f64,
    /// Lower band.
    pub lower: f64,
    /// Band width as percentage of middle.
    pub bandwidth: f64,
    /// %B: where price is relative to bands (0 = lower, 1 = upper).
    pub percent_b: f64,
}
