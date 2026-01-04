//! Fundamental valuation types for the trading simulation (V2.4).
//!
//! This module provides:
//! - [`Fundamentals`]: Per-symbol financial data (EPS, growth, payout ratio)
//! - [`MacroEnvironment`]: Market-wide rates (risk-free rate, equity risk premium)
//! - [`SymbolFundamentals`]: Container for all symbol fundamentals
//! - [`fair_value()`]: Gordon Growth Model valuation
//!
//! # Fair Value Calculation
//!
//! The Gordon Growth Model computes intrinsic value as:
//!
//! ```text
//! fair_value = D1 / (r - g)
//!
//! where:
//!   D1 = EPS × payout_ratio × (1 + growth)  // Next year's dividend
//!   r  = risk_free_rate + equity_risk_premium // Required return
//!   g  = growth_estimate                      // Perpetual growth rate
//! ```
//!
//! When r ≤ g (model undefined), we fall back to a P/E multiple.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use types::{Price, Symbol};

// =============================================================================
// Fundamentals
// =============================================================================

/// Per-symbol fundamental financial data.
///
/// These values drive the Gordon Growth Model fair value calculation.
/// Events can permanently modify these values (e.g., earnings surprises).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fundamentals {
    /// Earnings per share (fixed-point, same scale as Price).
    pub eps: Price,

    /// Expected annual growth rate (e.g., 0.05 = 5%).
    /// Used as perpetual growth rate in Gordon Growth Model.
    pub growth_estimate: f64,

    /// Dividend payout ratio (0.0 to 1.0).
    /// Fraction of earnings paid as dividends.
    pub payout_ratio: f64,
}

impl Fundamentals {
    /// Create new fundamentals.
    ///
    /// # Arguments
    /// * `eps` - Earnings per share
    /// * `growth_estimate` - Annual growth rate (e.g., 0.05 for 5%)
    /// * `payout_ratio` - Dividend payout ratio (0.0 to 1.0)
    pub fn new(eps: Price, growth_estimate: f64, payout_ratio: f64) -> Self {
        Self {
            eps,
            growth_estimate,
            payout_ratio: payout_ratio.clamp(0.0, 1.0),
        }
    }

    /// Calculate fair value using the Gordon Growth Model.
    ///
    /// # Formula
    /// ```text
    /// fair_value = D1 / (r - g)
    /// D1 = EPS × payout_ratio × (1 + growth)
    /// r = risk_free_rate + equity_risk_premium
    /// ```
    ///
    /// # Fallback
    /// When r ≤ g (model undefined), returns EPS × fallback P/E multiple (15x).
    pub fn fair_value(&self, macro_env: &MacroEnvironment) -> Price {
        let eps_float = self.eps.to_float();

        // Next year's expected dividend
        let d1 = eps_float * self.payout_ratio * (1.0 + self.growth_estimate);

        // Required rate of return
        let r = macro_env.required_return();
        let g = self.growth_estimate;

        // Gordon Growth Model requires r > g
        if r <= g || d1 <= 0.0 {
            // Fallback: P/E multiple of 15
            return Price::from_float(eps_float * 15.0);
        }

        let value = d1 / (r - g);

        // Sanity check: clamp to reasonable range
        let clamped = value.clamp(eps_float * 5.0, eps_float * 100.0);
        Price::from_float(clamped)
    }

    /// Apply an earnings surprise (permanently modifies EPS).
    ///
    /// # Arguments
    /// * `surprise_pct` - Percentage change (e.g., 0.10 for +10%, -0.05 for -5%)
    pub fn apply_earnings_surprise(&mut self, surprise_pct: f64) {
        let multiplier = 1.0 + surprise_pct;
        let new_eps = self.eps.to_float() * multiplier;
        self.eps = Price::from_float(new_eps.max(0.01)); // Ensure positive EPS
    }

    /// Apply a guidance change (permanently modifies growth estimate).
    ///
    /// # Arguments
    /// * `new_growth` - New growth estimate (e.g., 0.08 for 8%)
    pub fn apply_guidance_change(&mut self, new_growth: f64) {
        self.growth_estimate = new_growth.clamp(-0.20, 0.50); // Clamp to reasonable range
    }
}

impl Default for Fundamentals {
    fn default() -> Self {
        Self {
            eps: Price::from_float(5.0), // $5.00 EPS
            growth_estimate: 0.05,       // 5% growth
            payout_ratio: 0.40,          // 40% dividend payout
        }
    }
}

// =============================================================================
// MacroEnvironment
// =============================================================================

/// Market-wide macroeconomic parameters.
///
/// These affect all symbols' fair value calculations.
/// Rate decisions permanently modify these values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroEnvironment {
    /// Risk-free rate (e.g., 0.04 = 4%).
    /// Typically based on government bond yields.
    pub risk_free_rate: f64,

    /// Equity risk premium (e.g., 0.05 = 5%).
    /// Additional return required for equity risk vs risk-free.
    pub equity_risk_premium: f64,
}

impl MacroEnvironment {
    /// Create a new macro environment.
    pub fn new(risk_free_rate: f64, equity_risk_premium: f64) -> Self {
        Self {
            risk_free_rate,
            equity_risk_premium,
        }
    }

    /// Get the required rate of return (r = risk_free + equity_premium).
    pub fn required_return(&self) -> f64 {
        self.risk_free_rate + self.equity_risk_premium
    }

    /// Apply a rate decision (permanently modifies risk-free rate).
    pub fn apply_rate_decision(&mut self, new_rate: f64) {
        self.risk_free_rate = new_rate.clamp(0.0, 0.20); // 0% to 20%
    }
}

impl Default for MacroEnvironment {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.04,      // 4%
            equity_risk_premium: 0.05, // 5%
        }
    }
}

// =============================================================================
// SymbolFundamentals
// =============================================================================

/// Container for all symbol fundamentals and macro environment.
///
/// This is the top-level struct passed to agents for fair value lookups.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolFundamentals {
    /// Per-symbol fundamental data.
    data: HashMap<Symbol, Fundamentals>,

    /// Market-wide macro environment.
    pub macro_env: MacroEnvironment,
}

impl SymbolFundamentals {
    /// Create a new container with the given macro environment.
    pub fn new(macro_env: MacroEnvironment) -> Self {
        Self {
            data: HashMap::new(),
            macro_env,
        }
    }

    /// Add or update fundamentals for a symbol.
    pub fn insert(&mut self, symbol: impl Into<Symbol>, fundamentals: Fundamentals) {
        self.data.insert(symbol.into(), fundamentals);
    }

    /// Get fundamentals for a symbol.
    pub fn get(&self, symbol: &Symbol) -> Option<&Fundamentals> {
        self.data.get(symbol)
    }

    /// Get mutable fundamentals for a symbol.
    pub fn get_mut(&mut self, symbol: &Symbol) -> Option<&mut Fundamentals> {
        self.data.get_mut(symbol)
    }

    /// Calculate fair value for a symbol.
    ///
    /// Returns `None` if no fundamentals exist for the symbol.
    pub fn fair_value(&self, symbol: &Symbol) -> Option<Price> {
        self.data.get(symbol).map(|f| f.fair_value(&self.macro_env))
    }

    /// Get all symbols with fundamentals.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.data.keys()
    }

    /// Get the number of symbols.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fair_value_gordon_growth() {
        // EPS = $5, growth = 5%, payout = 40%
        // D1 = 5 * 0.40 * 1.05 = $2.10
        // r = 4% + 5% = 9%
        // fair_value = 2.10 / (0.09 - 0.05) = 2.10 / 0.04 = $52.50
        let fundamentals = Fundamentals::new(Price::from_float(5.0), 0.05, 0.40);
        let macro_env = MacroEnvironment::new(0.04, 0.05);

        let fv = fundamentals.fair_value(&macro_env);
        let fv_float = fv.to_float();

        assert!(
            (fv_float - 52.50).abs() < 0.01,
            "Expected ~$52.50, got ${fv_float:.2}"
        );
    }

    #[test]
    fn test_fair_value_fallback_when_r_le_g() {
        // When r <= g, model is undefined, should use P/E fallback
        let fundamentals = Fundamentals::new(Price::from_float(5.0), 0.15, 0.40);
        let macro_env = MacroEnvironment::new(0.02, 0.05); // r = 7% < g = 15%

        let fv = fundamentals.fair_value(&macro_env);
        let fv_float = fv.to_float();

        // Fallback is EPS * 15 = $75
        assert!(
            (fv_float - 75.0).abs() < 0.01,
            "Expected P/E fallback of $75, got ${fv_float:.2}"
        );
    }

    #[test]
    fn test_earnings_surprise_positive() {
        let mut fundamentals = Fundamentals::new(Price::from_float(5.0), 0.05, 0.40);
        fundamentals.apply_earnings_surprise(0.10); // +10%

        let new_eps = fundamentals.eps.to_float();
        assert!(
            (new_eps - 5.50).abs() < 0.01,
            "Expected EPS of $5.50, got ${new_eps:.2}"
        );
    }

    #[test]
    fn test_earnings_surprise_negative() {
        let mut fundamentals = Fundamentals::new(Price::from_float(5.0), 0.05, 0.40);
        fundamentals.apply_earnings_surprise(-0.20); // -20%

        let new_eps = fundamentals.eps.to_float();
        assert!(
            (new_eps - 4.0).abs() < 0.01,
            "Expected EPS of $4.00, got ${new_eps:.2}"
        );
    }

    #[test]
    fn test_rate_decision() {
        let mut macro_env = MacroEnvironment::new(0.04, 0.05);
        macro_env.apply_rate_decision(0.05); // Rate hike to 5%

        assert!(
            (macro_env.risk_free_rate - 0.05).abs() < 1e-10,
            "Expected rate of 5%"
        );
        assert!(
            (macro_env.required_return() - 0.10).abs() < 1e-10,
            "Expected required return of 10%"
        );
    }

    #[test]
    fn test_symbol_fundamentals_container() {
        let mut sf = SymbolFundamentals::new(MacroEnvironment::default());
        sf.insert(
            "AAPL",
            Fundamentals::new(Price::from_float(6.0), 0.08, 0.25),
        );
        sf.insert(
            "MSFT",
            Fundamentals::new(Price::from_float(10.0), 0.10, 0.30),
        );

        assert_eq!(sf.len(), 2);
        assert!(sf.fair_value(&"AAPL".to_string()).is_some());
        assert!(sf.fair_value(&"GOOG".to_string()).is_none());
    }

    #[test]
    fn test_guidance_change() {
        let mut fundamentals = Fundamentals::default();
        fundamentals.apply_guidance_change(0.12); // New growth: 12%

        assert!(
            (fundamentals.growth_estimate - 0.12).abs() < 1e-10,
            "Expected growth of 12%"
        );
    }
}
