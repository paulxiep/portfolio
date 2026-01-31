//! Tree-based ML Agent for Tier 1 (V5.5).
//!
//! Generic agent that uses any [`MlModel`] to make trading decisions based on
//! 42 market features extracted from [`StrategyContext`].
//!
//! # Decision Logic
//!
//! For each tick:
//! 1. Extract 42 features for each watched symbol
//! 2. Call `model.predict(features)` to get `[p_sell, p_hold, p_buy]`
//! 3. Find best buy candidate (highest p_buy above threshold)
//! 4. Find best sell candidate (highest p_sell above threshold)
//! 5. Execute the stronger signal (if any)
//!
//! # Feature Extraction
//!
//! Features are extracted from `StrategyContext` indicators and market data.
//! During warmup (first 64 ticks), many features will be NaN - the model
//! handles this by going left at NaN splits (conservative behavior).
//!
//! # Usage
//!
//! ```ignore
//! let model = DecisionTree::from_json("models/shallow_decision_tree.json")?;
//! let config = TreeAgentConfig {
//!     symbols: vec!["ACME".into()],
//!     buy_threshold: 0.55,
//!     sell_threshold: 0.55,
//!     ..Default::default()
//! };
//! let agent = TreeAgent::new(AgentId(100), model, config);
//! ```

use std::marker::PhantomData;

use crate::state::AgentState;
use crate::{Agent, AgentAction, StrategyContext, floor_price};
use types::{AgentId, Cash, Order, OrderSide, Price, Quantity, Symbol, Trade};

use super::{ClassProbabilities, MlModel};

/// Configuration for a tree-based ML agent.
#[derive(Debug, Clone)]
pub struct TreeAgentConfig {
    /// Symbols to trade.
    pub symbols: Vec<Symbol>,
    /// Probability threshold to trigger a buy (e.g., 0.55 = 55% confidence).
    pub buy_threshold: f64,
    /// Probability threshold to trigger a sell.
    pub sell_threshold: f64,
    /// Order size in shares.
    pub order_size: u64,
    /// Maximum long position per symbol.
    pub max_long_position: i64,
    /// Maximum short position per symbol (as positive number).
    pub max_short_position: i64,
    /// Initial cash balance.
    pub initial_cash: Cash,
    /// Initial price reference when market data unavailable.
    pub initial_price: Price,
}

impl Default for TreeAgentConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["ACME".to_string()],
            buy_threshold: 0.55,
            sell_threshold: 0.55,
            order_size: 50,
            max_long_position: 1000,
            max_short_position: 200,
            initial_cash: Cash::from_float(100_000.0),
            initial_price: Price::from_float(100.0),
        }
    }
}

/// Generic tree-based ML agent.
///
/// Uses any model implementing [`MlModel`] to predict class probabilities
/// and generates orders based on configurable thresholds.
pub struct TreeAgent<M: MlModel> {
    /// Unique agent identifier.
    id: AgentId,
    /// Configuration.
    config: TreeAgentConfig,
    /// Common agent state (position, cash, metrics).
    state: AgentState,
    /// The ML model for predictions.
    model: M,
    /// Phantom marker for the model type.
    _marker: PhantomData<M>,
}

impl<M: MlModel> TreeAgent<M> {
    /// Create a new tree-based ML agent.
    pub fn new(id: AgentId, model: M, config: TreeAgentConfig) -> Self {
        let state = AgentState::with_symbols(config.initial_cash, config.symbols.clone());
        Self {
            id,
            config,
            state,
            model,
            _marker: PhantomData,
        }
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.model.name()
    }

    /// Get reference price for a symbol.
    fn get_reference_price(&self, symbol: &Symbol, ctx: &StrategyContext<'_>) -> Price {
        ctx.mid_price(symbol)
            .or(ctx.last_price(symbol))
            .unwrap_or(self.config.initial_price)
    }

    /// Check if we can buy more of a symbol.
    fn can_buy(&self, symbol: &Symbol) -> bool {
        self.state.position_for(symbol) < self.config.max_long_position
    }

    /// Check if we can sell/short more of a symbol.
    fn can_sell(&self, symbol: &Symbol) -> bool {
        self.state.position_for(symbol) > -self.config.max_short_position
    }

    /// Generate a buy order for a symbol.
    fn generate_buy_order(&self, symbol: &Symbol, mid_price: Price) -> Order {
        // Bid ABOVE mid to qualify in batch auction (bid >= ref_price)
        // Batch auction: bid qualifies if limit_price >= reference_price
        // Apply floor_price to prevent negative price spirals
        let order_price = Price::from_float(floor_price(mid_price.to_float() * 0.999));
        Order::limit(
            self.id,
            symbol,
            OrderSide::Buy,
            order_price,
            Quantity(self.config.order_size),
        )
    }

    /// Generate a sell order for a symbol.
    fn generate_sell_order(&self, symbol: &Symbol, mid_price: Price) -> Order {
        // Ask BELOW mid to qualify in batch auction (ask <= ref_price)
        // Batch auction: ask qualifies if limit_price <= reference_price
        // Apply floor_price to prevent negative price spirals
        let order_price = Price::from_float(floor_price(mid_price.to_float() * 1.001));
        Order::limit(
            self.id,
            symbol,
            OrderSide::Sell,
            order_price,
            Quantity(self.config.order_size),
        )
    }
}

impl<M: MlModel> Agent for TreeAgent<M> {
    fn id(&self) -> AgentId {
        self.id
    }

    fn on_tick(&mut self, ctx: &StrategyContext<'_>) -> AgentAction {
        let mut orders: Vec<Order> = Vec::new();
        let model_name = self.model.name();

        // Evaluate each symbol independently - can place up to 1 order per symbol
        for symbol in &self.config.symbols {
            // V5.6: Use centralized cached prediction (computed in Phase 3 of tick loop)
            // If cache unavailable, hold - no fallback to avoid redundant computation
            let probs: ClassProbabilities = match ctx.get_ml_prediction(model_name, symbol) {
                Some(p) => p,
                None => continue, // Skip this symbol if no prediction cached
            };

            let p_sell = probs[0] + rand::random::<f64>() * 0.005;
            let p_hold: f64 = probs[1]; // + rand::random::<f64>() * 0.01;
            let p_buy = probs[2] + rand::random::<f64>() * 0.005;

            let mid_price = self.get_reference_price(symbol, ctx);

            // Independent decision per symbol: stronger signal wins if above threshold
            let buy_signal =
                p_buy > self.config.buy_threshold && p_buy >= p_hold && self.can_buy(symbol);
            let sell_signal =
                p_sell > self.config.sell_threshold && p_sell >= p_hold && self.can_sell(symbol);

            match (sell_signal, buy_signal) {
                (true, true) if p_buy >= p_sell => {
                    orders.push(self.generate_buy_order(symbol, mid_price));
                    self.state.record_order();
                }
                (true, true) => {
                    orders.push(self.generate_sell_order(symbol, mid_price));
                    self.state.record_order();
                }
                (true, false) => {
                    // Only sell above threshold
                    orders.push(self.generate_sell_order(symbol, mid_price));
                    self.state.record_order();
                }
                (false, true) => {
                    // Only buy above threshold
                    orders.push(self.generate_buy_order(symbol, mid_price));
                    self.state.record_order();
                }
                (false, false) => {
                    // No signal above threshold for this symbol
                }
            }
        }

        if orders.is_empty() {
            AgentAction::none()
        } else {
            AgentAction::multiple(orders)
        }
    }

    fn on_fill(&mut self, trade: &Trade) {
        // Use separate if blocks (not else if) to handle self-trades correctly.
        if trade.buyer_id == self.id {
            self.state
                .on_buy(&trade.symbol, trade.quantity.raw(), trade.value());
        }
        if trade.seller_id == self.id {
            self.state
                .on_sell(&trade.symbol, trade.quantity.raw(), trade.value());
        }
    }

    fn name(&self) -> &str {
        self.model.name()
    }

    fn is_ml_agent(&self) -> bool {
        true
    }

    fn state(&self) -> &AgentState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tier1::ml::DecisionTree;

    fn sample_tree_json() -> &'static str {
        r#"{
            "model_type": "decision_tree",
            "model_name": "test",
            "feature_names": ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31","f32","f33","f34","f35","f36","f37","f38","f39","f40","f41"],
            "n_features": 42,
            "n_classes": 3,
            "classes": [-1, 0, 1],
            "tree": {
                "n_nodes": 3,
                "nodes": [
                    {"feature": 29, "threshold": 50.0, "left": 1, "right": 2, "value": null},
                    {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.1, 0.1, 0.8]},
                    {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.8, 0.1, 0.1]}
                ]
            }
        }"#
    }

    #[test]
    fn test_tree_agent_creation() {
        let model = DecisionTree::from_json_str(sample_tree_json()).unwrap();
        let config = TreeAgentConfig::default();
        let agent = TreeAgent::new(AgentId(1), model, config);

        assert_eq!(agent.id(), AgentId(1));
        assert_eq!(agent.model_name(), "DecisionTree_test");
    }

    #[test]
    fn test_price_change_calculation() {
        use types::price_change_from_candles;

        // Create candles: 100, 102, 105 (5% increase over 2 periods)
        let candles = vec![
            types::Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(100.0),
                high: Price::from_float(101.0),
                low: Price::from_float(99.0),
                close: Price::from_float(100.0),
                volume: types::Quantity(1000),
                tick: 1,
                timestamp: 1000,
            },
            types::Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(100.0),
                high: Price::from_float(103.0),
                low: Price::from_float(100.0),
                close: Price::from_float(102.0),
                volume: types::Quantity(1000),
                tick: 2,
                timestamp: 2000,
            },
            types::Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(102.0),
                high: Price::from_float(106.0),
                low: Price::from_float(102.0),
                close: Price::from_float(105.0),
                volume: types::Quantity(1000),
                tick: 3,
                timestamp: 3000,
            },
        ];

        // 1-period change: (105 - 102) / 102 * 100 ≈ 2.94%
        let change_1 = price_change_from_candles(&candles, 1);
        assert!((change_1 - 2.941).abs() < 0.01);

        // 2-period change: (105 - 100) / 100 * 100 = 5%
        let change_2 = price_change_from_candles(&candles, 2);
        assert!((change_2 - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_log_return_calculation() {
        use types::log_return_from_candles;

        let candles = vec![
            types::Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(100.0),
                high: Price::from_float(101.0),
                low: Price::from_float(99.0),
                close: Price::from_float(100.0),
                volume: types::Quantity(1000),
                tick: 1,
                timestamp: 1000,
            },
            types::Candle {
                symbol: "TEST".to_string(),
                open: Price::from_float(100.0),
                high: Price::from_float(111.0),
                low: Price::from_float(100.0),
                close: Price::from_float(110.0),
                volume: types::Quantity(1000),
                tick: 2,
                timestamp: 2000,
            },
        ];

        // ln(110/100) ≈ 0.0953
        let log_return = log_return_from_candles(&candles, 1);
        assert!((log_return - 0.0953).abs() < 0.001);
    }

    #[test]
    fn test_insufficient_candles_returns_nan() {
        use types::price_change_from_candles;

        let candles = vec![types::Candle {
            symbol: "TEST".to_string(),
            open: Price::from_float(100.0),
            high: Price::from_float(101.0),
            low: Price::from_float(99.0),
            close: Price::from_float(100.0),
            volume: types::Quantity(1000),
            tick: 1,
            timestamp: 1000,
        }];

        // Need at least 2 candles for lookback=1
        let change = price_change_from_candles(&candles, 1);
        assert!(change.is_nan());
    }
}
