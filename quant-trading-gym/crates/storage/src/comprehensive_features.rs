//! Comprehensive feature extractor for ML training.
//!
//! V5.5.2: Uses unified feature schema from `types::features` to ensure
//! training-serving consistency. Only market features (42) are extracted;
//! agent features have been removed as tree agents only use market data.
//!
//! ## Warmup Requirements
//!
//! Features require minimum ticks before producing values.
//! Candle interval is 1 tick (1 candle = 1 tick).
//!
//! | Feature | Ticks Needed |
//! |---------|--------------|
//! | price_change_64 | 64 |
//! | SMA_8/16 | 8/16 |
//! | EMA_8/16 | ~16 |
//! | RSI_8 | 9 |
//! | MACD 8/16/4 | 20 |
//! | Bollinger 12 | 12 |
//! | ATR_8 | 8 |
//!
//! Use `--record-warmup 100` for complete coverage (64 for price + buffer).

use crate::price_history::PriceHistory;
use simulation::{EnrichedData, MarketSnapshot};
use types::{
    IndicatorType, LOOKBACKS, N_MARKET_FEATURES, Symbol, bollinger_percent_b, feature_idx as idx,
};

// ─────────────────────────────────────────────────────────────────────────────
// Market Features (42 total) - One row per tick per symbol
// ─────────────────────────────────────────────────────────────────────────────

/// Market-level features (42 total).
///
/// Stored once per tick - identical for all agents trading the same symbol.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Feature values in order of MARKET_FEATURE_NAMES.
    pub features: Vec<f64>,
    /// Mid price (convenience - also features[0]).
    pub mid_price: f64,
}

impl MarketFeatures {
    /// Number of market-level features.
    pub const COUNT: usize = N_MARKET_FEATURES;

    /// Feature names for market features.
    pub fn feature_names() -> &'static [&'static str] {
        types::MARKET_FEATURE_NAMES
    }

    /// Extract market features for a symbol.
    ///
    /// Uses the unified feature schema from `types::features` for index
    /// positions and computation functions to ensure training-serving parity.
    pub fn extract(
        symbol: &Symbol,
        tick: u64,
        market: &MarketSnapshot,
        enriched: Option<&EnrichedData>,
        price_history: &PriceHistory,
    ) -> Self {
        let mut features = vec![f64::NAN; Self::COUNT];

        // ─────────────────────────────────────────────────────────────────────
        // Price features (25) - idx::MID_PRICE through idx::LOG_RETURN_START+11
        // ─────────────────────────────────────────────────────────────────────

        let mid_price = market
            .mid_price(symbol)
            .map(|p: types::Price| p.to_float())
            .unwrap_or(f64::NAN);
        features[idx::MID_PRICE] = mid_price;

        // Price changes at lookback horizons (declarative iteration)
        LOOKBACKS.iter().enumerate().for_each(|(i, &period)| {
            features[idx::PRICE_CHANGE_START + i] = price_history
                .price_change(symbol, period)
                .unwrap_or(f64::NAN);
        });

        // Log returns at lookback horizons
        LOOKBACKS.iter().enumerate().for_each(|(i, &period)| {
            features[idx::LOG_RETURN_START + i] =
                price_history.log_return(symbol, period).unwrap_or(f64::NAN);
        });

        // ─────────────────────────────────────────────────────────────────────
        // Technical indicators (13) - idx::SMA_8 through idx::ATR_8
        // ─────────────────────────────────────────────────────────────────────

        let indicators = enriched.and_then(|e| e.indicators.get(symbol));

        let get_indicator = |itype: IndicatorType| -> f64 {
            indicators
                .and_then(|m| m.get(&itype))
                .copied()
                .unwrap_or(f64::NAN)
        };

        features[idx::SMA_8] = get_indicator(IndicatorType::Sma(8));
        features[idx::SMA_16] = get_indicator(IndicatorType::Sma(16));
        features[idx::EMA_8] = get_indicator(IndicatorType::Ema(8));
        features[idx::EMA_16] = get_indicator(IndicatorType::Ema(16));
        features[idx::RSI_8] = get_indicator(IndicatorType::Rsi(8));
        features[idx::MACD_LINE] = get_indicator(IndicatorType::MACD_LINE_STANDARD);
        features[idx::MACD_SIGNAL] = get_indicator(IndicatorType::MACD_SIGNAL_STANDARD);
        features[idx::MACD_HISTOGRAM] = get_indicator(IndicatorType::MACD_HISTOGRAM_STANDARD);

        let bb_upper = get_indicator(IndicatorType::BOLLINGER_UPPER_STANDARD);
        let bb_middle = get_indicator(IndicatorType::BOLLINGER_MIDDLE_STANDARD);
        let bb_lower = get_indicator(IndicatorType::BOLLINGER_LOWER_STANDARD);
        features[idx::BB_UPPER] = bb_upper;
        features[idx::BB_MIDDLE] = bb_middle;
        features[idx::BB_LOWER] = bb_lower;

        // Bollinger %B using unified computation function
        features[idx::BB_PERCENT_B] = bollinger_percent_b(mid_price, bb_upper, bb_lower);
        features[idx::ATR_8] = get_indicator(IndicatorType::Atr(8));

        // ─────────────────────────────────────────────────────────────────────
        // News features (4) - idx::HAS_ACTIVE_NEWS through idx::NEWS_TICKS_REMAINING
        // ─────────────────────────────────────────────────────────────────────

        let news_event = enriched.and_then(|e| {
            e.news_events
                .iter()
                .find(|n| n.event.symbol() == Some(symbol))
        });

        features[idx::HAS_ACTIVE_NEWS] = if news_event.is_some() { 1.0 } else { 0.0 };

        let (news_sentiment, news_magnitude, news_ticks_remaining) = news_event
            .map(|event| {
                let ticks_elapsed = tick.saturating_sub(event.start_tick);
                let ticks_remaining = event.duration_ticks.saturating_sub(ticks_elapsed);
                (event.sentiment, event.magnitude, ticks_remaining as f64)
            })
            .unwrap_or((0.0, 0.0, 0.0));

        features[idx::NEWS_SENTIMENT] = news_sentiment;
        features[idx::NEWS_MAGNITUDE] = news_magnitude;
        features[idx::NEWS_TICKS_REMAINING] = news_ticks_remaining;

        debug_assert_eq!(features.len(), Self::COUNT);

        Self {
            features,
            mid_price,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_feature_count() {
        assert_eq!(types::MARKET_FEATURE_NAMES.len(), MarketFeatures::COUNT);
        assert_eq!(MarketFeatures::COUNT, 42);
    }

    #[test]
    fn test_feature_name_prefix() {
        for name in types::MARKET_FEATURE_NAMES {
            assert!(
                name.starts_with("f_"),
                "Market feature name '{}' should start with 'f_'",
                name
            );
        }
    }
}
