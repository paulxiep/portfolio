//! Comprehensive feature extractor for ML training.
//!
//! V5.3: Extracts features covering price, technical indicators,
//! news, agent state, and risk metrics.
//!
//! ## Features Removed (incompatible with batch auction mode)
//!
//! | Feature | Reason |
//! |---------|--------|
//! | `f_spread`, `f_spread_bps` | Requires live bid/ask (books cleared between ticks) |
//! | `f_fair_value*` | Fair value engine not implemented in hooks |
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

use crate::features::{FeatureContext, FeatureExtractor};

/// Comprehensive feature extractor.
///
/// Extracts features from multiple categories:
/// - Price features (mid_price, returns at multiple horizons)
/// - Technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
/// - News/sentiment features
/// - Agent state features (position, cash, PnL)
/// - Risk features (equity, drawdown, sharpe, volatility)
#[derive(Debug, Clone)]
pub struct ComprehensiveFeatures {
    /// Lookback periods for price change calculations.
    #[allow(dead_code)]
    lookback_periods: Vec<usize>,
}

impl Default for ComprehensiveFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl ComprehensiveFeatures {
    /// Create a new comprehensive feature extractor.
    pub fn new() -> Self {
        Self {
            // Fibonacci-like geometric spread for batch auction market
            lookback_periods: PRICE_LOOKBACKS.to_vec(),
        }
    }

    /// Create with custom lookback periods.
    pub fn with_lookbacks(lookbacks: Vec<usize>) -> Self {
        Self {
            lookback_periods: lookbacks,
        }
    }
}

/// Lookback periods for price changes (geometric spread for batch auction).
const PRICE_LOOKBACKS: &[usize] = &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

/// Feature names in order (must match extract() output order).
const FEATURE_NAMES: &[&str] = &[
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
    // Agent state features (6)
    "f_position",
    "f_position_normalized",
    "f_cash",
    "f_cash_normalized",
    "f_total_pnl",
    "f_pnl_normalized",
    // Risk features (4)
    "f_equity",
    "f_max_drawdown",
    "f_sharpe",
    "f_volatility",
];

impl FeatureExtractor for ComprehensiveFeatures {
    fn feature_names(&self) -> &[&'static str] {
        FEATURE_NAMES
    }

    fn extract(&self, ctx: &FeatureContext) -> Vec<f64> {
        let mut features = Vec::with_capacity(FEATURE_NAMES.len());

        // ─────────────────────────────────────────────────────────────────────
        // Price features (17) - more granular horizons for batch auction
        // ─────────────────────────────────────────────────────────────────────

        let mid_price = ctx
            .market
            .mid_price(ctx.symbol)
            .map(|p| p.to_float())
            .unwrap_or(f64::NAN);
        features.push(mid_price);

        // Price changes (%) at Fibonacci lookback horizons
        for &period in PRICE_LOOKBACKS {
            features.push(
                ctx.price_history
                    .price_change(ctx.symbol, period)
                    .unwrap_or(f64::NAN),
            );
        }

        // Log returns at Fibonacci lookback horizons
        for &period in PRICE_LOOKBACKS {
            features.push(
                ctx.price_history
                    .log_return(ctx.symbol, period)
                    .unwrap_or(f64::NAN),
            );
        }

        // ─────────────────────────────────────────────────────────────────────
        // Technical indicators (13) - from quant crate via indicator engine
        // ─────────────────────────────────────────────────────────────────────

        let indicators = ctx.enriched.and_then(|e| e.indicators.get(ctx.symbol));

        let get_indicator = |name: &str| -> f64 {
            indicators
                .and_then(|m| m.get(name))
                .copied()
                .unwrap_or(f64::NAN)
        };

        // Moving averages (uppercase keys from market_data.rs)
        features.push(get_indicator("SMA_8"));
        features.push(get_indicator("SMA_16"));
        features.push(get_indicator("EMA_8"));
        features.push(get_indicator("EMA_16"));

        // RSI
        features.push(get_indicator("RSI_8"));

        // MACD (computed via Macd::standard().calculate_full())
        features.push(get_indicator("MACD_line"));
        features.push(get_indicator("MACD_signal"));
        features.push(get_indicator("MACD_histogram"));

        // Bollinger Bands (computed via BollingerBands::standard().calculate_full())
        let bb_upper = get_indicator("BB_upper");
        let bb_middle = get_indicator("BB_middle");
        let bb_lower = get_indicator("BB_lower");
        features.push(bb_upper);
        features.push(bb_middle);
        features.push(bb_lower);

        // Bollinger %B: (price - lower) / (upper - lower)
        let bb_percent_b = if !bb_upper.is_nan() && !bb_lower.is_nan() && !mid_price.is_nan() {
            let band_width = bb_upper - bb_lower;
            if band_width > 0.0 {
                (mid_price - bb_lower) / band_width
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
        features.push(bb_percent_b);

        // ATR
        features.push(get_indicator("ATR_8"));

        // ─────────────────────────────────────────────────────────────────────
        // News/sentiment features (4)
        // ─────────────────────────────────────────────────────────────────────

        let news_event = ctx.enriched.and_then(|e| {
            e.news_events
                .iter()
                .find(|n| n.event.symbol() == Some(ctx.symbol))
        });

        let has_active_news = if news_event.is_some() { 1.0 } else { 0.0 };
        features.push(has_active_news);

        let (news_sentiment, news_magnitude, news_ticks_remaining) = match news_event {
            Some(event) => {
                let ticks_elapsed = ctx.tick.saturating_sub(event.start_tick);
                let ticks_remaining = event.duration_ticks.saturating_sub(ticks_elapsed);
                (event.sentiment, event.magnitude, ticks_remaining as f64)
            }
            None => (0.0, 0.0, 0.0),
        };
        features.push(news_sentiment);
        features.push(news_magnitude);
        features.push(news_ticks_remaining);

        // ─────────────────────────────────────────────────────────────────────
        // Agent state features (6)
        // ─────────────────────────────────────────────────────────────────────

        let position = ctx
            .agent_summary
            .positions
            .get(ctx.symbol)
            .copied()
            .unwrap_or(0) as f64;
        features.push(position);

        let position_normalized = if ctx.position_limit > 0 {
            position / ctx.position_limit as f64
        } else {
            0.0
        };
        features.push(position_normalized);

        let cash = ctx.agent_summary.cash.0 as f64 / 100.0;
        features.push(cash);

        let cash_normalized = if ctx.initial_cash > 0.0 {
            cash / ctx.initial_cash
        } else {
            f64::NAN
        };
        features.push(cash_normalized);

        let total_pnl = ctx.agent_summary.total_pnl.0 as f64 / 100.0;
        features.push(total_pnl);

        let pnl_normalized = if ctx.initial_cash > 0.0 {
            total_pnl / ctx.initial_cash * 100.0
        } else {
            f64::NAN
        };
        features.push(pnl_normalized);

        // ─────────────────────────────────────────────────────────────────────
        // Risk features (4)
        // ─────────────────────────────────────────────────────────────────────

        let (equity, max_drawdown, sharpe, volatility) = match ctx.risk_snapshot {
            Some(risk) => (
                risk.equity,
                risk.max_drawdown,
                risk.sharpe.unwrap_or(f64::NAN),
                risk.volatility.unwrap_or(f64::NAN),
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        features.push(equity);
        features.push(max_drawdown);
        features.push(sharpe);
        features.push(volatility);

        debug_assert_eq!(
            features.len(),
            FEATURE_NAMES.len(),
            "Feature count mismatch: got {}, expected {}",
            features.len(),
            FEATURE_NAMES.len()
        );

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        let extractor = ComprehensiveFeatures::new();
        // 25 price (1+12*2) + 13 technical + 4 news + 6 agent + 4 risk = 52
        assert_eq!(extractor.feature_names().len(), 52);
    }

    #[test]
    fn test_feature_name_prefix() {
        let extractor = ComprehensiveFeatures::new();
        for name in extractor.feature_names() {
            assert!(
                name.starts_with("f_"),
                "Feature name '{}' should start with 'f_'",
                name
            );
        }
    }
}
