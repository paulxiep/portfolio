//! Configuration types for trading simulation rules.
//!
//! This module defines configuration structures for symbol constraints,
//! short-selling rules, and risk violation types.

use crate::ids::Symbol;
use crate::money::{Price, Quantity};
use serde::{Deserialize, Serialize};

// =============================================================================
// Symbol Configuration (V2.1)
// =============================================================================

/// Configuration for a single symbol's market constraints.
///
/// Defines the natural limits on position sizes based on shares outstanding
/// and any symbol-specific trading rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolConfig {
    /// The symbol this configuration applies to.
    pub symbol: Symbol,
    /// Total shares outstanding in the market.
    /// This sets the natural upper bound on aggregate long positions.
    pub shares_outstanding: Quantity,
    /// Fraction of shares available for borrowing (basis points, e.g., 1500 = 15%).
    /// The borrow pool size = shares_outstanding * borrow_pool_bps / 10000.
    pub borrow_pool_bps: u32,
    /// Initial/reference price for this symbol.
    pub initial_price: Price,
}

impl SymbolConfig {
    /// Create a new symbol configuration.
    pub fn new(
        symbol: impl Into<Symbol>,
        shares_outstanding: Quantity,
        initial_price: Price,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            shares_outstanding,
            borrow_pool_bps: 1500, // Default 15% borrow pool
            initial_price,
        }
    }

    /// Set the borrow pool fraction (in basis points).
    pub fn with_borrow_pool_bps(mut self, bps: u32) -> Self {
        self.borrow_pool_bps = bps;
        self
    }

    /// Calculate the total shares available for borrowing.
    pub fn borrow_pool_size(&self) -> Quantity {
        let pool = (self.shares_outstanding.raw() as u128 * self.borrow_pool_bps as u128) / 10_000;
        Quantity(pool as u64)
    }
}

impl Default for SymbolConfig {
    fn default() -> Self {
        Self {
            symbol: "SIM".to_string(),
            shares_outstanding: Quantity(1_000_000),
            borrow_pool_bps: 1500,
            initial_price: Price::from_float(100.0),
        }
    }
}

// =============================================================================
// Short-Selling Configuration (V2.1)
// =============================================================================

/// Configuration for short-selling rules.
///
/// Controls whether short-selling is allowed and under what constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortSellingConfig {
    /// Whether short selling is allowed at all.
    pub enabled: bool,
    /// Annual borrow rate in basis points (e.g., 50 = 0.5%/year).
    /// Used for calculating borrow costs over time.
    pub borrow_rate_bps: u32,
    /// Whether agents must locate shares before shorting.
    /// When true, shorts are rejected if no borrow is available.
    pub locate_required: bool,
    /// Maximum short position per agent (risk limit).
    /// Set to 0 for unlimited (within borrow availability).
    pub max_short_per_agent: Quantity,
}

impl ShortSellingConfig {
    /// Create a new short-selling configuration.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            borrow_rate_bps: 50, // Default 0.5% annual
            locate_required: true,
            max_short_per_agent: Quantity(10_000),
        }
    }

    /// Disable short selling entirely.
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Enable short selling with default settings.
    pub fn enabled_default() -> Self {
        Self::new(true)
    }

    /// Set the borrow rate (basis points per year).
    pub fn with_borrow_rate_bps(mut self, bps: u32) -> Self {
        self.borrow_rate_bps = bps;
        self
    }

    /// Set whether locate is required before shorting.
    pub fn with_locate_required(mut self, required: bool) -> Self {
        self.locate_required = required;
        self
    }

    /// Set the maximum short position per agent.
    pub fn with_max_short(mut self, max: Quantity) -> Self {
        self.max_short_per_agent = max;
        self
    }
}

impl Default for ShortSellingConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

// =============================================================================
// Risk Violation
// =============================================================================

/// Reasons why an order might be rejected due to position limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskViolation {
    /// Agent doesn't have enough cash to buy.
    InsufficientCash,
    /// Not enough shares exist in the market (long position would exceed shares outstanding).
    InsufficientShares,
    /// Short position would exceed agent's max short limit.
    ShortLimitExceeded,
    /// No shares available to borrow for shorting.
    NoBorrowAvailable,
    /// Short selling is disabled for this simulation.
    ShortSellingDisabled,
}

impl std::fmt::Display for RiskViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientCash => write!(f, "Insufficient cash"),
            Self::InsufficientShares => write!(f, "Insufficient shares in market"),
            Self::ShortLimitExceeded => write!(f, "Short position limit exceeded"),
            Self::NoBorrowAvailable => write!(f, "No shares available to borrow"),
            Self::ShortSellingDisabled => write!(f, "Short selling is disabled"),
        }
    }
}

impl std::error::Error for RiskViolation {}
