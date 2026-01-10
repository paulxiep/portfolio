//! Data Service REST API endpoints (V4.3).
//!
//! Provides analytics, portfolio, risk, and news data endpoints.
//! All endpoints query live simulation state and V3.9 storage.
//!
//! # Endpoints
//!
//! ## Analytics
//! - `GET /api/analytics/candles?symbol=X&timeframe=Y` - OHLCV candles
//! - `GET /api/analytics/indicators?symbol=X` - Technical indicators
//! - `GET /api/analytics/factors?symbol=X` - Factor scores
//!
//! ## Portfolio
//! - `GET /api/portfolio/agents` - List all agents with P&L summary
//! - `GET /api/portfolio/agents/:agent_id` - Detailed agent portfolio
//!
//! ## Risk
//! - `GET /api/risk/:agent_id` - Risk metrics for agent
//!
//! ## News
//! - `GET /api/news/active` - Current active events
//!
//! # Design Principles
//!
//! - **Declarative**: Pure handler functions returning typed responses
//! - **Modular**: Each domain (analytics, portfolio, risk, news) isolated
//! - **SoC**: Handlers extract from state, return JSON

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{AppError, AppResult};
use crate::state::ServerState;
use types::AgentId;

// =============================================================================
// Analytics Types
// =============================================================================

/// Query parameters for candles endpoint.
#[derive(Debug, Deserialize)]
pub struct CandlesQuery {
    /// Symbol to query (optional, returns all if not specified).
    pub symbol: Option<String>,
    /// Limit number of candles returned (default: 100).
    pub limit: Option<usize>,
}

/// OHLCV candle data for API response.
#[derive(Debug, Clone, Serialize)]
pub struct CandleData {
    pub symbol: String,
    pub tick: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}

/// Response for /api/analytics/candles.
#[derive(Debug, Serialize)]
pub struct CandlesResponse {
    /// Candles grouped by symbol.
    pub candles: HashMap<String, Vec<CandleData>>,
    /// Total candle count.
    pub total: usize,
}

/// Query parameters for indicators endpoint.
#[derive(Debug, Deserialize)]
pub struct IndicatorsQuery {
    /// Symbol to query.
    pub symbol: String,
}

/// Technical indicator values.
#[derive(Debug, Clone, Serialize)]
pub struct IndicatorData {
    /// Simple Moving Average values by period.
    pub sma: HashMap<u32, Option<f64>>,
    /// Exponential Moving Average values by period.
    pub ema: HashMap<u32, Option<f64>>,
    /// Relative Strength Index (14-period).
    pub rsi_14: Option<f64>,
    /// MACD values (line, signal, histogram).
    pub macd: Option<MacdData>,
    /// Bollinger Bands values.
    pub bollinger: Option<BollingerData>,
    /// Average True Range (14-period).
    pub atr_14: Option<f64>,
}

/// MACD indicator data.
#[derive(Debug, Clone, Serialize)]
pub struct MacdData {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

/// Bollinger Bands data.
#[derive(Debug, Clone, Serialize)]
pub struct BollingerData {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

/// Response for /api/analytics/indicators.
#[derive(Debug, Serialize)]
pub struct IndicatorsResponse {
    pub symbol: String,
    pub indicators: IndicatorData,
    pub tick: u64,
}

/// Query parameters for factors endpoint.
#[derive(Debug, Deserialize)]
pub struct FactorsQuery {
    /// Symbol to query.
    pub symbol: String,
}

/// Factor scores for a symbol.
#[derive(Debug, Clone, Serialize)]
pub struct FactorData {
    /// Momentum score (-1 to +1).
    pub momentum_score: f64,
    /// Value score (-1 to +1).
    pub value_score: f64,
    /// Volatility score (0 to 1).
    pub volatility_score: f64,
    /// Current price vs fair value ratio.
    pub price_to_fair_value: Option<f64>,
}

/// Response for /api/analytics/factors.
#[derive(Debug, Serialize)]
pub struct FactorsResponse {
    pub symbol: String,
    pub factors: FactorData,
    pub tick: u64,
}

// =============================================================================
// Portfolio Types
// =============================================================================

/// Summary of an agent for list view.
#[derive(Debug, Clone, Serialize)]
pub struct AgentSummary {
    pub agent_id: u64,
    pub name: String,
    /// Total P&L (realized + unrealized).
    pub total_pnl: f64,
    /// Cash balance.
    pub cash: f64,
    /// Total equity (cash + positions).
    pub equity: f64,
    /// Position count.
    pub position_count: usize,
    /// Whether this is a market maker.
    pub is_market_maker: bool,
    /// Agent tier (1, 2, or 3).
    pub tier: u8,
}

/// Response for /api/portfolio/agents.
#[derive(Debug, Serialize)]
pub struct AgentsResponse {
    pub agents: Vec<AgentSummary>,
    pub total_count: usize,
    pub tick: u64,
}

/// Detailed position information.
#[derive(Debug, Clone, Serialize)]
pub struct PositionDetail {
    pub symbol: String,
    pub quantity: i64,
    pub avg_cost: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub unrealized_pnl: f64,
}

/// Detailed agent portfolio response.
#[derive(Debug, Serialize)]
pub struct AgentPortfolioResponse {
    pub agent_id: u64,
    pub name: String,
    pub cash: f64,
    pub equity: f64,
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub positions: Vec<PositionDetail>,
    /// Equity curve (recent values).
    pub equity_curve: Vec<f64>,
    pub tick: u64,
}

// =============================================================================
// Risk Types
// =============================================================================

/// Risk metrics for an agent.
#[derive(Debug, Clone, Serialize)]
pub struct RiskMetricsResponse {
    pub agent_id: u64,
    pub name: String,
    /// Sharpe ratio (annualized).
    pub sharpe: Option<f64>,
    /// Sortino ratio (annualized).
    pub sortino: Option<f64>,
    /// Maximum drawdown (0.0 to 1.0).
    pub max_drawdown: f64,
    /// Value at Risk at 95% confidence.
    pub var_95: Option<f64>,
    /// Annualized volatility.
    pub volatility: Option<f64>,
    /// Total return as percentage.
    pub total_return: f64,
    /// Current equity.
    pub equity: f64,
    pub tick: u64,
}

// =============================================================================
// News Types
// =============================================================================

/// Active news event for API.
#[derive(Debug, Clone, Serialize)]
pub struct NewsEventData {
    pub id: u64,
    pub event_type: String,
    pub symbol: Option<String>,
    pub sector: Option<String>,
    pub sentiment: f64,
    pub magnitude: f64,
    pub start_tick: u64,
    pub duration_ticks: u64,
    pub effective_sentiment: f64,
    pub decay_factor: f64,
}

/// Response for /api/news/active.
#[derive(Debug, Serialize)]
pub struct ActiveNewsResponse {
    pub events: Vec<NewsEventData>,
    pub count: usize,
    pub tick: u64,
}

// =============================================================================
// Handlers - Analytics
// =============================================================================

/// Get OHLCV candles: `GET /api/analytics/candles`
pub async fn get_candles(
    State(state): State<ServerState>,
    Query(query): Query<CandlesQuery>,
) -> AppResult<Json<CandlesResponse>> {
    let sim_data = state.sim_data.read().await;
    let limit = query.limit.unwrap_or(100);

    let candles: HashMap<String, Vec<CandleData>> = match query.symbol {
        Some(ref symbol) => {
            let symbol_candles = sim_data
                .candles
                .get(symbol)
                .map(|candles| {
                    candles
                        .iter()
                        .rev()
                        .take(limit)
                        .rev()
                        .map(|c| CandleData {
                            symbol: c.symbol.clone(),
                            tick: c.tick,
                            open: c.open.to_float(),
                            high: c.high.to_float(),
                            low: c.low.to_float(),
                            close: c.close.to_float(),
                            volume: c.volume.raw(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let mut map = HashMap::new();
            map.insert(symbol.clone(), symbol_candles);
            map
        }
        None => sim_data
            .candles
            .iter()
            .map(|(sym, candles)| {
                let data = candles
                    .iter()
                    .rev()
                    .take(limit)
                    .rev()
                    .map(|c| CandleData {
                        symbol: c.symbol.clone(),
                        tick: c.tick,
                        open: c.open.to_float(),
                        high: c.high.to_float(),
                        low: c.low.to_float(),
                        close: c.close.to_float(),
                        volume: c.volume.raw(),
                    })
                    .collect();
                (sym.clone(), data)
            })
            .collect(),
    };

    let total = candles.values().map(|v| v.len()).sum();

    Ok(Json(CandlesResponse { candles, total }))
}

/// Get technical indicators: `GET /api/analytics/indicators`
pub async fn get_indicators(
    State(state): State<ServerState>,
    Query(query): Query<IndicatorsQuery>,
) -> AppResult<Json<IndicatorsResponse>> {
    let sim_data = state.sim_data.read().await;
    let symbol = &query.symbol;

    // Get indicator values from snapshot
    let indicators = sim_data.indicators.get(symbol).cloned().unwrap_or_default();

    // Build SMA map
    let mut sma = HashMap::new();
    sma.insert(10, indicators.get("SMA_10").copied());
    sma.insert(20, indicators.get("SMA_20").copied());
    sma.insert(50, indicators.get("SMA_50").copied());

    // Build EMA map
    let mut ema = HashMap::new();
    ema.insert(12, indicators.get("EMA_12").copied());
    ema.insert(26, indicators.get("EMA_26").copied());

    // RSI
    let rsi_14 = indicators.get("RSI_14").copied();

    // MACD
    let macd = match (
        indicators.get("MACD_line"),
        indicators.get("MACD_signal"),
        indicators.get("MACD_histogram"),
    ) {
        (Some(&line), Some(&signal), Some(&hist)) => Some(MacdData {
            macd_line: line,
            signal_line: signal,
            histogram: hist,
        }),
        _ => None,
    };

    // Bollinger Bands
    let bollinger = match (
        indicators.get("BB_upper"),
        indicators.get("BB_middle"),
        indicators.get("BB_lower"),
    ) {
        (Some(&upper), Some(&middle), Some(&lower)) => Some(BollingerData {
            upper,
            middle,
            lower,
        }),
        _ => None,
    };

    // ATR
    let atr_14 = indicators.get("ATR_14").copied();

    let indicator_data = IndicatorData {
        sma,
        ema,
        rsi_14,
        macd,
        bollinger,
        atr_14,
    };

    Ok(Json(IndicatorsResponse {
        symbol: symbol.clone(),
        indicators: indicator_data,
        tick: sim_data.tick,
    }))
}

/// Get factor scores: `GET /api/analytics/factors`
pub async fn get_factors(
    State(state): State<ServerState>,
    Query(query): Query<FactorsQuery>,
) -> AppResult<Json<FactorsResponse>> {
    let sim_data = state.sim_data.read().await;
    let symbol = &query.symbol;

    // Compute momentum score from RSI and SMA crossover
    let rsi = sim_data
        .indicators
        .get(symbol)
        .and_then(|ind| ind.get("RSI_14"))
        .copied()
        .unwrap_or(50.0);
    let momentum_score = (rsi - 50.0) / 50.0; // Normalize RSI to -1 to +1

    // Compute value score from price vs fair value
    let current_price = sim_data.prices.get(symbol).map(|p| p.to_float());
    let fair_value = sim_data.fair_values.get(symbol).map(|p| p.to_float());
    let (value_score, price_to_fair_value) = match (current_price, fair_value) {
        (Some(price), Some(fv)) if fv > 0.0 => {
            let ratio = price / fv;
            // If price < fair value, positive score (undervalued)
            let score = ((1.0 / ratio) - 1.0).clamp(-1.0, 1.0);
            (score, Some(ratio))
        }
        _ => (0.0, None),
    };

    // Compute volatility score from ATR
    let atr = sim_data
        .indicators
        .get(symbol)
        .and_then(|ind| ind.get("ATR_14"))
        .copied()
        .unwrap_or(0.0);
    let price = current_price.unwrap_or(100.0);
    let volatility_score = if price > 0.0 {
        (atr / price).min(1.0) // ATR as % of price, capped at 1.0
    } else {
        0.0
    };

    let factors = FactorData {
        momentum_score: momentum_score.clamp(-1.0, 1.0),
        value_score,
        volatility_score,
        price_to_fair_value,
    };

    Ok(Json(FactorsResponse {
        symbol: symbol.clone(),
        factors,
        tick: sim_data.tick,
    }))
}

// =============================================================================
// Handlers - Portfolio
// =============================================================================

/// List all agents with P&L summary: `GET /api/portfolio/agents`
pub async fn get_agents(State(state): State<ServerState>) -> AppResult<Json<AgentsResponse>> {
    let sim_data = state.sim_data.read().await;

    let agents: Vec<AgentSummary> = sim_data
        .agents
        .iter()
        .map(|a| AgentSummary {
            agent_id: a.id,
            name: a.name.clone(),
            total_pnl: a.total_pnl,
            cash: a.cash,
            equity: a.equity,
            position_count: a.position_count,
            is_market_maker: a.is_market_maker,
            tier: a.tier,
        })
        .collect();

    let total_count = agents.len();

    Ok(Json(AgentsResponse {
        agents,
        total_count,
        tick: sim_data.tick,
    }))
}

/// Get detailed agent portfolio: `GET /api/portfolio/agents/:agent_id`
pub async fn get_agent_portfolio(
    State(state): State<ServerState>,
    Path(agent_id): Path<u64>,
) -> AppResult<Json<AgentPortfolioResponse>> {
    let sim_data = state.sim_data.read().await;

    let agent = sim_data
        .agents
        .iter()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| AppError::NotFound(format!("Agent {} not found", agent_id)))?;

    let positions: Vec<PositionDetail> = agent
        .positions
        .iter()
        .map(|(symbol, pos)| {
            let current_price = sim_data
                .prices
                .get(symbol)
                .map(|p| p.to_float())
                .unwrap_or(0.0);
            let market_value = pos.quantity as f64 * current_price;
            let cost_basis = pos.quantity as f64 * pos.avg_cost;
            let unrealized_pnl = market_value - cost_basis;

            PositionDetail {
                symbol: symbol.clone(),
                quantity: pos.quantity,
                avg_cost: pos.avg_cost,
                current_price,
                market_value,
                unrealized_pnl,
            }
        })
        .collect();

    let unrealized_pnl: f64 = positions.iter().map(|p| p.unrealized_pnl).sum();

    // Get equity curve from risk tracker
    let equity_curve = sim_data
        .equity_curves
        .get(&AgentId(agent_id))
        .cloned()
        .unwrap_or_default();

    Ok(Json(AgentPortfolioResponse {
        agent_id,
        name: agent.name.clone(),
        cash: agent.cash,
        equity: agent.equity,
        total_pnl: agent.total_pnl,
        realized_pnl: agent.realized_pnl,
        unrealized_pnl,
        positions,
        equity_curve,
        tick: sim_data.tick,
    }))
}

// =============================================================================
// Handlers - Risk
// =============================================================================

/// Get risk metrics for agent: `GET /api/risk/:agent_id`
pub async fn get_risk_metrics(
    State(state): State<ServerState>,
    Path(agent_id): Path<u64>,
) -> AppResult<Json<RiskMetricsResponse>> {
    let sim_data = state.sim_data.read().await;

    let agent = sim_data
        .agents
        .iter()
        .find(|a| a.id == agent_id)
        .ok_or_else(|| AppError::NotFound(format!("Agent {} not found", agent_id)))?;

    let risk = sim_data
        .risk_metrics
        .get(&AgentId(agent_id))
        .cloned()
        .unwrap_or_default();

    Ok(Json(RiskMetricsResponse {
        agent_id,
        name: agent.name.clone(),
        sharpe: risk.sharpe,
        sortino: risk.sortino,
        max_drawdown: risk.max_drawdown,
        var_95: risk.var_95,
        volatility: risk.volatility,
        total_return: risk.total_return,
        equity: risk.equity,
        tick: sim_data.tick,
    }))
}

// =============================================================================
// Handlers - News
// =============================================================================

/// Get active news events: `GET /api/news/active`
pub async fn get_active_news(
    State(state): State<ServerState>,
) -> AppResult<Json<ActiveNewsResponse>> {
    let sim_data = state.sim_data.read().await;

    let events: Vec<NewsEventData> = sim_data
        .active_events
        .iter()
        .map(|e| {
            let event_type = match &e.event {
                news::FundamentalEvent::EarningsSurprise { .. } => "EarningsSurprise",
                news::FundamentalEvent::GuidanceChange { .. } => "GuidanceChange",
                news::FundamentalEvent::RateDecision { .. } => "RateDecision",
                news::FundamentalEvent::SectorNews { .. } => "SectorNews",
            };

            NewsEventData {
                id: e.id,
                event_type: event_type.to_string(),
                symbol: e.symbol().cloned(),
                sector: e.sector().map(|s| format!("{:?}", s)),
                sentiment: e.sentiment,
                magnitude: e.magnitude,
                start_tick: e.start_tick,
                duration_ticks: e.duration_ticks,
                effective_sentiment: e.effective_sentiment(sim_data.tick),
                decay_factor: e.decay_factor(sim_data.tick),
            }
        })
        .collect();

    let count = events.len();

    Ok(Json(ActiveNewsResponse {
        events,
        count,
        tick: sim_data.tick,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_data_serialization() {
        let candle = CandleData {
            symbol: "AAPL".to_string(),
            tick: 100,
            open: 150.0,
            high: 155.0,
            low: 148.0,
            close: 153.0,
            volume: 10000,
        };

        let json = serde_json::to_string(&candle).unwrap();
        assert!(json.contains("\"symbol\":\"AAPL\""));
        assert!(json.contains("\"tick\":100"));
    }

    #[test]
    fn test_factor_data_clamping() {
        let factors = FactorData {
            momentum_score: 1.5, // Should be clamped to 1.0 in handler
            value_score: -1.2,   // Should be clamped to -1.0 in handler
            volatility_score: 0.5,
            price_to_fair_value: Some(1.1),
        };

        assert!(factors.momentum_score <= 1.0 || factors.momentum_score >= -1.0);
    }

    #[test]
    fn test_agent_summary_serialization() {
        let summary = AgentSummary {
            agent_id: 1,
            name: "NoiseTrader-001".to_string(),
            total_pnl: 1234.56,
            cash: 95000.0,
            equity: 105000.0,
            position_count: 3,
            is_market_maker: false,
            tier: 1,
        };

        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("\"agent_id\":1"));
        assert!(json.contains("\"tier\":1"));
    }
}
