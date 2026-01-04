//! Core types for the Quant Trading Gym simulation.
//!
//! This crate provides all shared data types used across the simulation,
//! including order types, trade types, and fixed-point monetary values.

use derive_more::{Add, AddAssign, From, Into, Neg, Sub, SubAssign, Sum};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Mul;

// =============================================================================
// Constants
// =============================================================================

/// Fixed-point scale for Price and Cash types.
/// 10,000 = $1.00, 15,000 = $1.50, 100 = $0.01
pub const PRICE_SCALE: i64 = 10_000;

/// Sentinel AgentId for Tier 3 background pool trades.
pub const BACKGROUND_POOL_ID: AgentId = AgentId(0);

// =============================================================================
// Core ID Types (Newtypes for type safety)
// =============================================================================

/// Unique identifier for orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct OrderId(pub u64);

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Order({})", self.0)
    }
}

/// Unique identifier for agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct AgentId(pub u64);

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent({})", self.0)
    }
}

/// Unique identifier for trades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TradeId(pub u64);

impl fmt::Display for TradeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trade({})", self.0)
    }
}

// =============================================================================
// Symbol Type
// =============================================================================

/// Stock ticker symbol (e.g., "AAPL", "GOOGL").
pub type Symbol = String;

// =============================================================================
// Time Types
// =============================================================================

/// Wall clock timestamp in milliseconds since epoch.
pub type Timestamp = u64;

/// Simulation tick number (discrete time step).
pub type Tick = u64;

// =============================================================================
// Quantity Type (Newtype for shares)
// =============================================================================

/// Number of shares (newtype for type safety).
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    AddAssign,
    SubAssign,
    Sum,
    From,
    Into,
)]
pub struct Quantity(pub u64);

impl Quantity {
    pub const ZERO: Quantity = Quantity(0);

    /// Get raw value.
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Check if zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Saturating subtraction.
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Quantity(self.0.saturating_sub(rhs.0))
    }

    /// Minimum of two quantities.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Quantity(self.0.min(other.0))
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Qty({})", self.0)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Allow `quantity == 50` comparisons
impl PartialEq<u64> for Quantity {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

// =============================================================================
// Fixed-Point Monetary Types
// =============================================================================

/// Fixed-point price with 4 decimal places.
///
/// # Examples
/// - `Price(10000)` = $1.00
/// - `Price(15000)` = $1.50
/// - `Price(100)` = $0.01
/// - `Price(1)` = $0.0001
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    Neg,
    AddAssign,
    SubAssign,
    From,
    Into,
)]
pub struct Price(pub i64);

impl Price {
    pub const ZERO: Price = Price(0);

    /// Create a Price from a floating-point value.
    #[inline]
    pub fn from_float(v: f64) -> Self {
        Self((v * PRICE_SCALE as f64).round() as i64)
    }

    /// Convert to floating-point for display/calculations.
    #[inline]
    pub fn to_float(self) -> f64 {
        self.0 as f64 / PRICE_SCALE as f64
    }

    /// Raw internal value.
    #[inline]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Check if price is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Price(self.0.abs())
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price(${:.4})", self.to_float())
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.4}", self.to_float())
    }
}

/// Fixed-point cash/money with 4 decimal places.
///
/// Semantically identical to Price but represents account balances.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Default,
    Add,
    Sub,
    Neg,
    AddAssign,
    SubAssign,
    From,
    Into,
)]
pub struct Cash(pub i64);

impl Cash {
    pub const ZERO: Cash = Cash(0);

    /// Create Cash from a floating-point value.
    #[inline]
    pub fn from_float(v: f64) -> Self {
        Self((v * PRICE_SCALE as f64).round() as i64)
    }

    /// Convert to floating-point for display/calculations.
    #[inline]
    pub fn to_float(self) -> f64 {
        self.0 as f64 / PRICE_SCALE as f64
    }

    /// Raw internal value.
    #[inline]
    pub fn raw(self) -> i64 {
        self.0
    }

    /// Check if cash is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Check if cash is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }
}

impl fmt::Debug for Cash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cash(${:.4})", self.to_float())
    }
}

impl fmt::Display for Cash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:.4}", self.to_float())
    }
}

// =============================================================================
// Price-Quantity Operations
// =============================================================================

impl Mul<Quantity> for Price {
    type Output = Cash;

    /// Multiply price by quantity to get total cash value.
    fn mul(self, qty: Quantity) -> Cash {
        Cash(self.0 * qty.0 as i64)
    }
}

impl Mul<Price> for Quantity {
    type Output = Cash;

    fn mul(self, price: Price) -> Cash {
        Cash(price.0 * self.0 as i64)
    }
}

// =============================================================================
// Order Types
// =============================================================================

/// Which side of the market the order is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    /// Returns the opposite side.
    pub fn opposite(self) -> Self {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

impl fmt::Display for OrderSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Type of order determining execution rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Execute immediately at best available price.
    Market,
    /// Execute at specified price or better.
    Limit { price: Price },
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit { price } => write!(f, "LIMIT@{}", price),
        }
    }
}

/// Status of an order in the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum OrderStatus {
    /// Order created but not yet submitted.
    #[default]
    Pending,
    /// Order queued with latency, will execute at specified tick.
    Queued { execute_at: Tick },
    /// Order partially filled.
    PartialFill { filled: Quantity },
    /// Order completely filled.
    Filled,
    /// Order was cancelled.
    Cancelled,
}

/// A trading order submitted by an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Order {
    /// Unique order identifier (assigned by Market, use 0 as placeholder).
    pub id: OrderId,
    /// Agent who submitted the order.
    pub agent_id: AgentId,
    /// Symbol being traded.
    pub symbol: Symbol,
    /// Buy or Sell.
    pub side: OrderSide,
    /// Market or Limit order.
    pub order_type: OrderType,
    /// Number of shares.
    pub quantity: Quantity,
    /// Remaining quantity (for partial fills).
    pub remaining_quantity: Quantity,
    /// When order was created (wall clock).
    pub timestamp: Timestamp,
    /// Latency in ticks before order can be matched (0 = instant).
    pub latency_ticks: u64,
    /// Current status.
    pub status: OrderStatus,
}

impl Order {
    /// Create a new limit order.
    pub fn limit(
        agent_id: AgentId,
        symbol: impl Into<Symbol>,
        side: OrderSide,
        price: Price,
        quantity: Quantity,
    ) -> Self {
        Self {
            id: OrderId(0), // Placeholder, assigned by Market
            agent_id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit { price },
            quantity,
            remaining_quantity: quantity,
            timestamp: 0,
            latency_ticks: 0,
            status: OrderStatus::Pending,
        }
    }

    /// Create a new market order.
    pub fn market(
        agent_id: AgentId,
        symbol: impl Into<Symbol>,
        side: OrderSide,
        quantity: Quantity,
    ) -> Self {
        Self {
            id: OrderId(0),
            agent_id,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            remaining_quantity: quantity,
            timestamp: 0,
            latency_ticks: 0,
            status: OrderStatus::Pending,
        }
    }

    /// Get the limit price if this is a limit order.
    pub fn limit_price(&self) -> Option<Price> {
        match self.order_type {
            OrderType::Limit { price } => Some(price),
            OrderType::Market => None,
        }
    }

    /// Check if order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.remaining_quantity.is_zero()
    }

    /// Check if order is a buy order.
    pub fn is_buy(&self) -> bool {
        self.side == OrderSide::Buy
    }

    /// Check if order is a sell order.
    pub fn is_sell(&self) -> bool {
        self.side == OrderSide::Sell
    }
}

// =============================================================================
// Trade Types
// =============================================================================

/// A completed trade between two parties.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier.
    pub id: TradeId,
    /// Symbol traded.
    pub symbol: Symbol,
    /// Agent who bought.
    pub buyer_id: AgentId,
    /// Agent who sold.
    pub seller_id: AgentId,
    /// Order that was the buyer.
    pub buyer_order_id: OrderId,
    /// Order that was the seller.
    pub seller_order_id: OrderId,
    /// Execution price.
    pub price: Price,
    /// Number of shares traded.
    pub quantity: Quantity,
    /// When trade occurred (wall clock).
    pub timestamp: Timestamp,
    /// Simulation tick when trade occurred.
    pub tick: Tick,
}

impl Trade {
    /// Calculate the total value of this trade.
    pub fn value(&self) -> Cash {
        self.price * self.quantity
    }
}

impl fmt::Display for Trade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Trade[{}]: {} {} shares @ {} (buyer: {}, seller: {})",
            self.id, self.symbol, self.quantity, self.price, self.buyer_id, self.seller_id
        )
    }
}

// =============================================================================
// Market Data Types
// =============================================================================

/// OHLCV candle data for a single time period.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candle {
    /// Stock symbol.
    pub symbol: Symbol,
    /// Opening price.
    pub open: Price,
    /// Highest price during the period.
    pub high: Price,
    /// Lowest price during the period.
    pub low: Price,
    /// Closing price.
    pub close: Price,
    /// Trading volume during the period.
    pub volume: Quantity,
    /// Wall clock timestamp.
    pub timestamp: Timestamp,
    /// Simulation tick at period end.
    pub tick: Tick,
}

impl Candle {
    /// Create a new candle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: impl Into<Symbol>,
        open: Price,
        high: Price,
        low: Price,
        close: Price,
        volume: Quantity,
        timestamp: Timestamp,
        tick: Tick,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            open,
            high,
            low,
            close,
            volume,
            timestamp,
            tick,
        }
    }

    /// Get the typical price (HLC/3).
    #[inline]
    pub fn typical_price(&self) -> Price {
        Price((self.high.0 + self.low.0 + self.close.0) / 3)
    }

    /// Get the candle range (high - low).
    #[inline]
    pub fn range(&self) -> Price {
        self.high - self.low
    }

    /// Check if this is a bullish candle (close > open).
    #[inline]
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Check if this is a bearish candle (close < open).
    #[inline]
    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }
}

/// A single price level in the order book.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookLevel {
    /// Price at this level.
    pub price: Price,
    /// Total quantity available at this price.
    pub quantity: Quantity,
    /// Number of orders at this level.
    pub order_count: usize,
}

/// Snapshot of the order book at a point in time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BookSnapshot {
    /// Symbol this book is for.
    pub symbol: Symbol,
    /// Bid levels (highest first).
    pub bids: Vec<BookLevel>,
    /// Ask levels (lowest first).
    pub asks: Vec<BookLevel>,
    /// When snapshot was taken.
    pub timestamp: Timestamp,
    /// Simulation tick.
    pub tick: Tick,
}

impl BookSnapshot {
    /// Get the best bid price.
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.first().map(|l| l.price)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.first().map(|l| l.price)
    }

    /// Calculate the spread between best bid and ask.
    pub fn spread(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Calculate the mid price.
    pub fn mid_price(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(Price((bid.0 + ask.0) / 2)),
            _ => None,
        }
    }
}

// =============================================================================
// Quant Types
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
// Position Limits & Short-Selling Configuration (V2.1)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_from_float() {
        assert_eq!(Price::from_float(1.0), Price(10_000));
        assert_eq!(Price::from_float(1.50), Price(15_000));
        assert_eq!(Price::from_float(0.01), Price(100));
        assert_eq!(Price::from_float(100.0), Price(1_000_000));
    }

    #[test]
    fn test_price_to_float() {
        assert!((Price(10_000).to_float() - 1.0).abs() < 1e-10);
        assert!((Price(15_000).to_float() - 1.50).abs() < 1e-10);
        assert!((Price(100).to_float() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_price_arithmetic() {
        let p1 = Price::from_float(10.0);
        let p2 = Price::from_float(3.5);

        assert_eq!((p1 + p2).to_float(), 13.5);
        assert_eq!((p1 - p2).to_float(), 6.5);
    }

    #[test]
    fn test_price_quantity_multiplication() {
        let price = Price::from_float(50.0);
        let quantity = Quantity(100);

        let total = price * quantity;
        assert_eq!(total.to_float(), 5000.0);
    }

    #[test]
    fn test_cash_operations() {
        let c1 = Cash::from_float(1000.0);
        let c2 = Cash::from_float(250.0);

        assert_eq!((c1 - c2).to_float(), 750.0);
        assert!(c1.is_positive());
        assert!(!c1.is_negative());
    }

    #[test]
    fn test_order_side_opposite() {
        assert_eq!(OrderSide::Buy.opposite(), OrderSide::Sell);
        assert_eq!(OrderSide::Sell.opposite(), OrderSide::Buy);
    }

    #[test]
    fn test_limit_order_creation() {
        let order = Order::limit(
            AgentId(1),
            "AAPL",
            OrderSide::Buy,
            Price::from_float(150.0),
            Quantity(100),
        );

        assert_eq!(order.agent_id, AgentId(1));
        assert_eq!(order.symbol, "AAPL");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.limit_price(), Some(Price::from_float(150.0)));
        assert_eq!(order.quantity, 100);
        assert!(!order.is_filled());
    }

    #[test]
    fn test_market_order_creation() {
        let order = Order::market(AgentId(2), "GOOGL", OrderSide::Sell, Quantity(50));

        assert_eq!(order.agent_id, AgentId(2));
        assert_eq!(order.symbol, "GOOGL");
        assert_eq!(order.side, OrderSide::Sell);
        assert_eq!(order.limit_price(), None);
        assert!(order.is_sell());
    }

    #[test]
    fn test_trade_value() {
        let trade = Trade {
            id: TradeId(1),
            symbol: "AAPL".to_string(),
            buyer_id: AgentId(1),
            seller_id: AgentId(2),
            buyer_order_id: OrderId(1),
            seller_order_id: OrderId(2),
            price: Price::from_float(150.0),
            quantity: Quantity(100),
            timestamp: 0,
            tick: 0,
        };

        assert_eq!(trade.value().to_float(), 15000.0);
    }

    #[test]
    fn test_book_snapshot() {
        let snapshot = BookSnapshot {
            symbol: "AAPL".to_string(),
            bids: vec![
                BookLevel {
                    price: Price::from_float(99.0),
                    quantity: Quantity(100),
                    order_count: 2,
                },
                BookLevel {
                    price: Price::from_float(98.0),
                    quantity: Quantity(200),
                    order_count: 3,
                },
            ],
            asks: vec![
                BookLevel {
                    price: Price::from_float(101.0),
                    quantity: Quantity(150),
                    order_count: 1,
                },
                BookLevel {
                    price: Price::from_float(102.0),
                    quantity: Quantity(250),
                    order_count: 2,
                },
            ],
            timestamp: 0,
            tick: 0,
        };

        assert_eq!(snapshot.best_bid(), Some(Price::from_float(99.0)));
        assert_eq!(snapshot.best_ask(), Some(Price::from_float(101.0)));
        assert_eq!(snapshot.spread(), Some(Price::from_float(2.0)));
        assert_eq!(snapshot.mid_price(), Some(Price::from_float(100.0)));
    }

    #[test]
    fn test_symbol_config_defaults() {
        let config = SymbolConfig::default();
        assert_eq!(config.symbol, "SIM");
        assert_eq!(config.shares_outstanding, Quantity(1_000_000));
        assert_eq!(config.borrow_pool_bps, 1500);
    }

    #[test]
    fn test_symbol_config_borrow_pool_size() {
        let config = SymbolConfig::new("TEST", Quantity(10_000_000), Price::from_float(50.0))
            .with_borrow_pool_bps(1500); // 15%

        // 10,000,000 * 15% = 1,500,000
        assert_eq!(config.borrow_pool_size(), Quantity(1_500_000));
    }

    #[test]
    fn test_short_selling_config() {
        let disabled = ShortSellingConfig::disabled();
        assert!(!disabled.enabled);

        let enabled = ShortSellingConfig::enabled_default()
            .with_borrow_rate_bps(100)
            .with_max_short(Quantity(5_000));
        assert!(enabled.enabled);
        assert_eq!(enabled.borrow_rate_bps, 100);
        assert_eq!(enabled.max_short_per_agent, Quantity(5_000));
    }
}
