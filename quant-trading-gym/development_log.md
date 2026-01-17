# Development Log

## 2026-01-17: V5.4 Tree-Based Training + Dual Parquet Split

### Summary
Implemented Python training script for tree-based ML models. Trains Decision Tree, Random Forest, and Histogram Gradient Boosted trees. Exports to JSON for V5.5 Rust inference. **Split Parquet output into market features (42, once per tick) vs agent features (10, per agent per tick)** with parallel record building.

### Dual Parquet Architecture
| File | Rows | Features | Description |
|------|------|----------|-------------|
| `*_market.parquet` | 1 per tick | 42 | Price, indicators, news (shared across agents) |
| `*_agents.parquet` | 1 per agent/tick | 10 | Position, cash, PnL, risk + action/reward labels |

**Rationale**: Market features are identical for all 15K agents at a given tick. Writing once per tick instead of 15K times reduces I/O by ~15,000x for market data.

### Feature Split
| Category | Count | File | Extraction |
|----------|-------|------|------------|
| Price | 25 | market | `MarketFeatures::extract()` |
| Technical | 13 | market | once per tick |
| News | 4 | market | |
| Agent State | 6 | agents | `AgentFeatures::extract()` |
| Risk | 4 | agents | per agent, parallel |

### Parallel Processing
- Agent records built with `parallel::filter_map_slice` (CPU-bound feature extraction)
- Parquet write remains sequential (Arrow writer not thread-safe)

### Models
| Model | Library | Output |
|-------|---------|--------|
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | `{name}_decision_tree.json` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `{name}_random_forest.json` |
| Gradient Boosted | `sklearn.ensemble.HistGradientBoostingClassifier` | `{name}_gradient_boosted.json` |

### JSON Schema (for Rust inference)
```json
{
  "model_type": "decision_tree",
  "model_name": "shallow",
  "feature_names": ["f_mid_price", ...],
  "n_features": 52,
  "n_classes": 3,
  "classes": [-1, 0, 1],
  "tree": {
    "n_nodes": 33,
    "nodes": [
      {"feature": 5, "threshold": 0.5, "left": 1, "right": 2, "value": null},
      {"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "value": [0.1, 0.7, 0.2]}
    ]
  },
  "metadata": {"accuracy": 0.98, "trained_at": "..."}
}
```
- `value`: Class probabilities `[p_sell, p_hold, p_buy]` for leaf nodes

### Files
- `scripts/train_trees.py` - Training script (loads dual parquet, joins on tick)
- `scripts/train_config.yaml` - Model hyperparameters
- `crates/storage/src/comprehensive_features.rs` - `MarketFeatures` (42) + `AgentFeatures` (10)
- `crates/storage/src/parquet_writer.rs` - `DualParquetWriter` (market + agents files)
- `crates/storage/src/recording_hook.rs` - Parallel agent record building
- `requirements.txt` - Python dependencies (polars, sklearn, pyyaml, shap)

### Usage
```bash
# Generate training data (outputs data/training_market.parquet + data/training_agents.parquet)
cargo run --release -- --headless-record --ticks 2000 --record-warmup 1000

# Train models (auto-joins market + agent files)
python scripts/train_trees.py --input data/training
```

### Config Format
```yaml
data:
  input: data/training  # loads {input}_market.parquet + {input}_agents.parquet
  output_dir: models
  test_size: 0.2

decision_trees:
  - name: shallow
    max_depth: 5
  - name: deep
    max_depth: 15

random_forests:
  - name: small
    n_estimators: 50
    max_depth: 8

gradient_boosted:
  - name: fast
    n_estimators: 50
    learning_rate: 0.2

shap:
  enabled: false
```

### Changes
- **Random seed default**: `SimulationConfig::default()` now uses `rand::random()` instead of fixed seed 42
- **NaN handling**: Training script imputes NaN with 0 (neutral/no history) for features like `f_sharpe`
- **.gitignore**: Added `quant-trading-gym/models/` to portfolio `.gitignore`

### Results (test run, 15M rows)
| Model | Accuracy | Notes |
|-------|----------|-------|
| Decision Tree (depth=5) | 97.7% | 33 nodes |
| Random Forest (10 trees) | 97.8% | Top features: f_equity, f_cash |
| HistGradientBoosting | 97.7% | 10 iterations |

### Deferred to V6
- Feature normalization
- Hyperparameter tuning (cross-validation)
- Neural networks
- Advanced feature engineering

---

## 2026-01-17: V5.3 Feature Recording Mode

### Summary
Implemented `--headless-record` mode for ML training data capture. Records 52 features per tick per agent to Parquet files for Python ML training. Updated indicator periods to geometric spread optimized for batch auction (1 tick = 1 candle).

### Storage Architecture
| Storage | Purpose |
|---------|---------|
| **Parquet (RecordingHook)** | ML training data (batch read) |
| **In-memory (StrategyContext)** | Agent inference (real-time) |
| **SQL (StorageHook)** | Optional debug/replay only |

### Indicator Periods (Geometric Spread)
| Indicator | Old | New | Rationale |
|-----------|-----|-----|-----------|
| SMA Fast | 10 | 8 | Clean doubling (8→16) |
| SMA Slow | 50 | 16 | 2x spread for batch auction |
| EMA Fast | 10 | 8 | Match SMA |
| EMA Slow | 50 | 16 | Match SMA |
| RSI | 14 | 8 | Faster response |
| MACD Fast | 12 | 8 | Clean doubling |
| MACD Slow | 26 | 16 | 2x spread |
| MACD Signal | 9 | 4 | Responsive smoothing |
| Bollinger | 20 | 12 | Between 8 and 16 |
| ATR | 14 | 8 | Match RSI |

### Features (52 total)
| Category | Count | Description |
|----------|-------|-------------|
| Price | 25 | mid_price + price_change/log_return at 12 lookback horizons |
| Technical | 13 | SMA 8/16, EMA 8/16, RSI 8, MACD 8/16/4, Bollinger 12, ATR 8 |
| News | 4 | Active news sentiment, magnitude, duration |
| Agent State | 6 | Position, cash, PnL (raw + normalized) |
| Risk | 4 | Equity, drawdown, Sharpe, volatility |

### Files
- `crates/storage/src/comprehensive_features.rs` - Feature extractor (52 features)
- `crates/storage/src/recording_hook.rs` - SimulationHook for Parquet capture
- `crates/storage/src/parquet_writer.rs` - Buffered Arrow/Parquet writer
- `crates/storage/src/price_history.rs` - Rolling price history
- `crates/storage/src/features.rs` - FeatureContext, FeatureExtractor trait
- `crates/types/src/indicators.rs` - MACD_STANDARD, BOLLINGER_STANDARD constants
- `crates/quant/src/engine.rs` - `with_common_indicators()` registration
- `crates/agents/src/tier1/strategies/` - Agent config defaults (8/16 periods)
- `src/main.rs` - CLI flags, `run_headless_record()` function
- `docs/indicator_period_impact.md` - Full impact analysis document

### CLI
```bash
cargo run --release -- --headless-record \
  --ticks 10000 \
  --record-output data/training.parquet \
  --record-warmup 100 \
  --record-interval 1
```

### Configuration Locations
| Setting | File | Location |
|---------|------|----------|
| Price lookback horizons | `crates/storage/src/comprehensive_features.rs` | `PRICE_LOOKBACKS` constant |
| Warmup default (100) | `crates/storage/src/recording_hook.rs` | `RecordingConfig::default()` |
| Candle interval (4) | `crates/simulation/src/config.rs` | `SimulationConfig::default()` |
| Max candles (200) | `crates/simulation/src/config.rs` | `SimulationConfig::default()` |

### Candle Interval Rationale
- **Batch auction model**: Each tick is a single clearing price → candle_interval=1 produces flat OHLC (open=high=low=close)
- **candle_interval=4**: Aggregates 4 batch auctions into one candle → meaningful OHLC variation
- **Warmup 100 ticks** = 25 candles → sufficient for all indicators (max period 16)
- **max_candles=200** = 800 ticks of history

### Server/Frontend Updates
- Updated indicator periods in `crates/server/src/routes/data.rs` (SMA 8/16, EMA 8/16, RSI 8, ATR 8)
- Updated frontend types in `frontend/src/types/api.ts` (rsi_8, atr_8)
- Updated `frontend/src/components/dashboard/IndicatorPanel.tsx` (display names)
- Updated `frontend/src/api.integration.test.ts` (field checks)

### Bug Fixes
- **Price scale fix**: `f_mid_price` in feature extraction was using `p.0 / 100.0` (wrong) instead of `p.to_float()` (correct). Fixed in `comprehensive_features.rs`. Now mid_price and SMA/EMA values are in same scale (~$50).
- **Parquet validation script**: Added `scripts/check_parquet.py` for data quality checks

### Verification
```
=== DATA QUALITY CHECK ===
Total rows: 1,500,000 (100 ticks × 15,000 agents)
Total columns: 66

=== SAMPLE VALUES (first row) ===
f_mid_price: 51.22  (correct scale)
f_sma_8: 49.22      (matches mid_price scale)
f_rsi_8: 64.38      (valid 0-100 range)
```

---

## 2026-01-16: V5.2 Simulation Decomposition & Parallel Crate Migration

### Summary
Refactored `runner.rs` (~1400→~1220 lines) following "Declarative, Modular, SoC" philosophy. Extracted auction logic to `AuctionEngine` subsystem with pure parallel order collection. Removed dead cancellation code. Eliminated wasteful `iter→collect→iter→collect` patterns by adding direct HashMap collection utilities. Optimized hook context building (4→2 builds per tick). Added `id` field to `AgentSummary` struct. **Migrated `parallel` crate** from simulation re-export to direct usage across crates. Server hooks and routes now use parallel crate directly for agent/position processing.

### Files

| File | Changes |
|------|---------|
| `crates/parallel/` | **Standalone crate** - declarative parallel/sequential utilities |
| `crates/simulation/src/parallel.rs` | **Removed** - consumers use `parallel` crate directly |
| `crates/simulation/Cargo.toml` | Depends on `parallel` crate, forwards feature flag |
| `crates/server/Cargo.toml` | Added `parallel` dependency |
| `crates/simulation/src/runner.rs` | Removed `use crate::parallel`; uses `parallel::` directly |
| `crates/simulation/src/subsystems/auction.rs` | Uses `parallel::` directly for order pipeline |
| `crates/simulation/src/subsystems/agents.rs` | Uses `parallel::` directly; builds `AgentSummary` with `id` |
| `crates/simulation/src/traits/agents.rs` | Added `id: AgentId` field to `AgentSummary` struct |
| `crates/server/src/hooks.rs` | Uses `parallel::map_slice` for agent summary conversion |
| `crates/server/src/routes/data.rs` | Uses `parallel::map_slice` for position computation |

### Design

**Parallel Crate Direct Usage:**
```rust
// Before: re-export indirection
// crates/simulation/src/parallel.rs: pub use parallel::*;
// use crate::parallel;

// After: direct crate usage (no re-export needed)
// Cargo.toml: parallel.workspace = true
// In code: parallel::map_slice(&items, |x| process(x), false);
```

**AgentSummary with ID:**
```rust
// Before: index-based ID assignment in hooks
let agents = parallel::map_indices(&indices, |i| {
    AgentData { id: (i + 1) as u64, ... }
});

// After: ID comes from AgentSummary
pub struct AgentSummary {
    pub id: AgentId,  // NEW
    pub name: String,
    pub positions: HashMap<Symbol, i64>,
    pub cash: Cash,
    pub total_pnl: Cash,
}

let agents = parallel::map_slice(&summaries, |s| {
    AgentData { id: s.id.0, ... }
});
```

**Position Computation (Parallel):**
```rust
// Before: sequential iter with fold
let (positions, unrealized_pnl) = agent.positions.iter()
    .map(|(s, p)| build_position_detail(s, p))
    .fold((Vec::new(), 0.0), ...);

// After: parallel map + sequential sum (sum is O(n) cheap)
let position_entries: Vec<_> = agent.positions.iter().collect();
let positions = parallel::map_slice(&position_entries, 
    |(symbol, pos)| build_position_detail(symbol, pos), false);
let unrealized_pnl: f64 = positions.iter().map(|p| p.unrealized_pnl).sum();
```

### Parallel Crate API

| Function | Purpose |
|----------|---------|
| `map_slice` | `&[T] → Vec<R>` |
| `filter_map_slice` | `&[T] → Vec<R>` (with filter) |
| `for_each_slice` | Side effects |
| `map_indices` / `filter_map_indices` | Index-based access |
| `map_vec` / `filter_map_vec` | Owned `Vec<T>` |
| `flat_filter_map_vec` | Expand + filter in one pass |
| `map_to_hashmap` / `filter_map_to_hashmap` | Direct to HashMap |
| `map_mutex_slice` / `map_mutex_slice_ref` | Mutex-wrapped items |
| `map_mutex_slice_ref_to_hashmap` | Mutex items → HashMap |

### Exit Criteria
```bash
cargo fmt --all                              # ✅ Formatted
cargo clippy --all-targets                   # ✅ No warnings
cargo test -p simulation -p server -p parallel # ✅ 38 tests pass
```

---

## 2026-01-11: V5.1 Fair Value Drift

### Summary
Added bounded random walk to fair value between news events, solving the "flat line with occasional jumps" price pattern problem. Previously, prices were unrealistically stable because all agents estimated similar fair values from deterministic Gordon Growth Model parameters. Now fair value drifts ±0.5% per tick (configurable), bounded by 10% floor and 10x ceiling of initial value.

### Files

| File | Changes |
|------|---------|
| `crates/news/src/config.rs` | Added `FairValueDriftConfig` struct with enabled, drift_pct, min_pct, max_multiple |
| `crates/news/src/lib.rs` | Added `FairValueDriftConfig` to re-exports |
| `crates/news/src/fundamentals.rs` | Added `DriftState` struct, `apply_drift()` method, drift multiplier tracking |
| `crates/simulation/src/config.rs` | Added `fair_value_drift` field to `SimulationConfig` with builder methods |
| `crates/simulation/src/runner.rs` | Added `drift_rng` field, `apply_fair_value_drift()` called in Phase 0b of tick |

### Design

**Approach:** Drift the Gordon Growth Model **output** (fair value), not the inputs (EPS, growth, rate). This keeps fundamentals internally consistent while adding realistic uncertainty.

**Drift Mechanics:**
```rust
// Each tick, apply small random drift to fair value multiplier
let drift = rng.gen_range(-drift_pct..drift_pct);
let new_multiplier = drift_state.multiplier * (1.0 + drift);

// Clamp to bounds based on initial fair value
let min_fv = initial_fair_value * config.min_pct;    // 10% floor
let max_fv = initial_fair_value * config.max_multiple; // 10x ceiling
```

**Configuration:**
```rust
pub struct FairValueDriftConfig {
    pub enabled: bool,        // Default: true (false for tests)
    pub drift_pct: f64,       // Default: 0.005 (0.5% per tick)
    pub min_pct: f64,         // Default: 0.1 (10% of initial)
    pub max_multiple: f64,    // Default: 10.0 (10x initial)
}
```

### Tick Loop Integration

```
Phase 0:  Process news events (apply permanent fundamental changes)
Phase 0b: Apply fair value drift ← NEW
Phase 1:  Hook: on_tick_start
Phase 2:  Determine agents to call
Phase 3:  Build strategy context (agents see drifted fair values)
...
```

**Determinism:** Drift uses separate seeded RNG (`config.seed + 1`) from news generator to maintain reproducibility while keeping drift independent.

### DriftState Tracking

Per-symbol drift state stored in `SymbolFundamentals`:
```rust
pub struct DriftState {
    pub multiplier: f64,         // Current drift multiplier (starts at 1.0)
    pub initial_fair_value: f64, // For bounds calculation
}
```

Fair value calculation now applies drift:
```rust
pub fn fair_value(&self, symbol: &Symbol) -> Option<Price> {
    let base_fv = fundamentals.fair_value(&self.macro_env);
    let multiplier = self.drift_state.get(symbol).map(|ds| ds.multiplier).unwrap_or(1.0);
    Some(Price::from_float(base_fv.to_float() * multiplier))
}
```

### Test Strategy

- **Disabled drift for existing tests**: Tests use `FairValueDriftConfig::disabled()` to maintain determinism
- **New drift-specific tests**: Verify drift changes values, respects bounds, and disabled mode works

```rust
#[test]
fn test_drift_stays_within_bounds() {
    // Apply 1000 drifts with extreme 10% drift per tick
    // Verify fair value never exceeds [50%, 200%] of initial
}
```

### Exit Criteria
```bash
cargo fmt                           # ✅ Formatted
cargo clippy --all-features         # ✅ No warnings
cargo test --all-features --all     # ✅ 335+ tests pass
cargo test -p news                  # ✅ 23 tests (3 new drift tests)
cargo test -p simulation            # ✅ 18 tests pass
```

### Notes
- Drift adds ~10 LOC to simulation tick hot path (negligible performance impact)
- Future: Per-symbol drift rates, Ornstein-Uhlenbeck mean reversion, correlated market-wide drift
- Agents now see gradually wandering fair values, creating realistic bid/ask spread dynamics

---

## 2026-01-11: V4.4 Simulation Dashboard

### Summary
Implemented full simulation dashboard with real-time visualization components. Created DataServiceHook to populate REST API caches, including pre-auction order distribution capture before batch auction clearing. Built complete React component library: price charts, indicator panels, order depth, factor gauges, risk metrics, news feed, agent explorer with sortable PnL table, and time controls.

### Files

| File | Changes |
|------|---------|
| `crates/server/src/hooks.rs` | Added `DataServiceHook` with `on_orders_collected` and `on_tick_end` |
| `crates/server/src/state.rs` | Added `OrderDistribution` struct for pre-auction bid/ask levels |
| `crates/server/src/lib.rs` | Exported `DataServiceHook`, `OrderDistribution` |
| `crates/server/Cargo.toml` | Added `parking_lot` dependency |
| `frontend/src/types/api.ts` | Updated types: `Candle`, `IndicatorsResponse`, `FactorSnapshot`, `AgentData`, `RiskMetricsResponse`, `NewsEventData`, `OrderDistributionResponse` |
| `frontend/src/hooks/useDataService.ts` | New: REST API hooks with auto-refresh |
| `frontend/src/hooks/index.ts` | Added useDataService exports |
| `frontend/src/components/dashboard/PriceChart.tsx` | New: SVG candlestick chart |
| `frontend/src/components/dashboard/IndicatorPanel.tsx` | New: Technical indicator display |
| `frontend/src/components/dashboard/OrderDepthChart.tsx` | New: Pre-auction bid/ask distribution |
| `frontend/src/components/dashboard/FactorGauges.tsx` | New: Macro factor gauges |
| `frontend/src/components/dashboard/RiskPanel.tsx` | New: VaR, drawdown, Sharpe display |
| `frontend/src/components/dashboard/NewsFeed.tsx` | New: Active news events feed |
| `frontend/src/components/dashboard/AgentTable.tsx` | New: Sortable agent explorer |
| `frontend/src/components/dashboard/TimeControls.tsx` | New: Play/pause/step controls |
| `frontend/src/components/dashboard/index.ts` | New: Dashboard components barrel |
| `frontend/src/components/index.ts` | Added dashboard exports |
| `frontend/src/pages/SimulationPage.tsx` | Replaced placeholder with full dashboard |

### DataServiceHook Architecture

**Pre-Auction Order Capture:**
The simulation uses batch auction where order book is cleared after each tick. To visualize order flow, `DataServiceHook` captures orders at `on_orders_collected` (Phase 6) before auction clearing.

```rust
impl SimulationHook for DataServiceHook {
    fn on_orders_collected(&mut self, orders: Vec<Order>) {
        // Capture pre-auction bid/ask distribution
        self.update_order_distribution(orders);
    }

    fn on_tick_end(&mut self, tick: u64, ...) {
        // Update candles, indicators, agents, etc.
        self.update_sim_data(tick, ...);
    }
}
```

**Order Distribution Structure:**
```rust
pub struct OrderDistribution {
    pub bids: Vec<(Price, u64)>,  // Descending by price
    pub asks: Vec<(Price, u64)>,  // Ascending by price
}
```

### Dashboard Components

**Charts:**
- `PriceChart`: SVG candlestick with price axis, green/red coloring
- `IndicatorPanel`: SMA, EMA, RSI, MACD, Bollinger, ATR values
- `OrderDepthChart`: Horizontal bar chart showing bid/ask imbalance

**Panels:**
- `FactorGauges`: Visual gauges for macro factors (rate, volatility, momentum)
- `RiskPanel`: VaR 95%, max drawdown, Sharpe ratio with risk level badge
- `NewsFeed`: Active news cards with impact badges and duration progress

**Tables:**
- `AgentTable`: Sortable columns (ID, type, cash, positions, PnL), pagination

**Controls:**
- `TimeControls`: WebSocket connection status, tick counter, play/pause/step buttons

### useDataService Hooks

```typescript
// Factory pattern for consistent API hooks
const useCandles = createDataHook<CandlesResponse>('/api/candles');
const useIndicators = createDataHook<IndicatorsResponse>('/api/indicators');
const useFactors = createDataHook<FactorsResponse>('/api/factors');
const useAgents = createDataHook<AgentsResponse>('/api/agents');
const useRiskMetrics = createDataHook<RiskMetricsResponse>('/api/risk');
const useActiveNews = createDataHook<ActiveNewsResponse>('/api/news');
const useOrderDistribution = createDataHook<OrderDistributionResponse>('/api/order-distribution');

// Composite hook for dashboard
const useDashboardData = (config, refresh) => ({
  candles: useCandles(config, refresh),
  indicators: useIndicators(config, refresh),
  // ...all data hooks
});
```

### SimulationPage Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Header: Logo ← | Configure button                          │
├─────────────────────────────────────────────────────────────┤
│ TimeControls: [●Connected] Tick: 1,234 | ▶ ⏸ ⏭ ■          │
├─────────────────────────────────────────────────────────────┤
│ Tabs: [Dashboard] [Agents] [Order Flow]                    │
├─────────────────────────────────────────────────────────────┤
│                                      │                     │
│  PriceChart (8 cols)                 │  IndicatorPanel     │
│  ┌─────────────────────────────┐    │  (4 cols)           │
│  │ AAPL  150.25 +0.50 (+0.33%) │    │  ┌───────────────┐  │
│  │ ╭───╮  ╭───╮                │    │  │ SMA-20: 149.8 │  │
│  │ │   │──│   │                │    │  │ RSI: 55.3     │  │
│  │ ╰───╯  ╰───╯                │    │  │ MACD: +0.4    │  │
│  └─────────────────────────────┘    │  └───────────────┘  │
│                                      │                     │
│  OrderDepthChart                     │  FactorGauges       │
│  ┌─────────────────────────────┐    │  ┌───────────────┐  │
│  │ BIDS ████░░░░ ASKS          │    │  │ Rate: ███░░   │  │
│  │ 150.10 ██████               │    │  │ Vol:  ████░   │  │
│  │ 150.05 ████                 │    │  └───────────────┘  │
│  └─────────────────────────────┘    │                     │
├──────────────────────────────────────┼─────────────────────┤
│  RiskPanel (6 cols)                  │  NewsFeed (6 cols)  │
│  ┌─────────────────────┐             │  ┌────────────────┐ │
│  │ [LOW RISK]          │             │  │ 📊 Earnings    │ │
│  │ VaR 95%: 1.5%       │             │  │    +2.5%       │ │
│  │ Drawdown: 3.2%      │             │  │ 🏛️ Fed Rate   │ │
│  │ Sharpe: 1.8         │             │  │    -0.1%       │ │
│  └─────────────────────┘             │  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles
- **Declarative**: Components driven by props from hooks, no internal state mutations
- **Modular**: Each component is self-contained with single responsibility
- **SoC**: Data fetching (hooks) separated from display (components) separated from layout (page)

### Exit Criteria
```bash
cargo build --package server     # ✅ Compiles
docker compose -f docker-compose.frontend.yaml run --rm typecheck  # ✅ No errors
docker compose -f docker-compose.frontend.yaml run --rm format     # ✅ Formatted
```

### Notes
- Pre-auction order capture enables "order book depth" visualization even in batch auction model
- All charts use pure SVG/CSS - no external charting library dependencies
- Auto-refresh interval configurable via `useDataService` hooks
- Agent table supports sorting by any numeric field, pagination for large agent counts

---

## 2026-01-10: V4.3 Data Service

### Summary
Implemented REST API endpoints for analytics, portfolio, risk, and news data. The Data Service provides comprehensive query APIs for the React frontend to fetch simulation state, enabling the V4.4 Simulation Dashboard. All endpoints read from a shared `SimData` cache updated by a hook on each tick.

### Files

| File | Changes |
|------|---------|
| `crates/server/src/routes/data.rs` | New: 7 data service endpoints with typed request/response |
| `crates/server/src/routes/mod.rs` | Added `data` module export |
| `crates/server/src/state.rs` | Added `SimData`, `AgentData`, `AgentPosition`, `NewsEventSnapshot` |
| `crates/server/src/app.rs` | Wired 7 new routes to router |
| `crates/server/src/lib.rs` | Updated exports for new types |
| `crates/server/Cargo.toml` | Added `quant` and `news` crate dependencies |

### Data Service Endpoints

**Analytics:**
- `GET /api/analytics/candles?symbol=X&limit=N` - OHLCV candle data
- `GET /api/analytics/indicators?symbol=X` - Technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR)
- `GET /api/analytics/factors?symbol=X` - Factor scores (momentum, value, volatility)

**Portfolio:**
- `GET /api/portfolio/agents` - List all agents with P&L summary
- `GET /api/portfolio/agents/{agent_id}` - Detailed agent portfolio (positions, equity curve)

**Risk:**
- `GET /api/risk/{agent_id}` - Risk metrics (Sharpe, Sortino, VaR, max drawdown, volatility)

**News:**
- `GET /api/news/active` - Active news events with sentiment and decay

### Architecture

**SimData Cache:**
```
SimData (Arc<RwLock<...>>)
├── tick: u64
├── candles: HashMap<Symbol, Vec<Candle>>
├── indicators: HashMap<Symbol, HashMap<String, f64>>
├── prices: HashMap<Symbol, Price>
├── fair_values: HashMap<Symbol, Price>
├── agents: Vec<AgentData>
├── risk_metrics: HashMap<AgentId, AgentRiskSnapshot>
├── equity_curves: HashMap<AgentId, Vec<f64>>
└── active_events: Vec<NewsEventSnapshot>
```

**Data Flow:**
```
Simulation Thread                    Axum Server
       │                                  │
       │── DataServiceHook ───────────────▶│ Update SimData
       │   (on_tick_end)                  │
       │                                  │
       │                                  │◀── GET /api/... ────
       │                                  │    Read SimData
```

### Response Types

**CandlesResponse:**
```json
{
  "candles": {
    "AAPL": [{ "tick": 100, "open": 150.0, "high": 155.0, ... }]
  },
  "total": 100
}
```

**IndicatorsResponse:**
```json
{
  "symbol": "AAPL",
  "indicators": {
    "sma": { "10": 150.5, "20": 149.8, "50": 148.2 },
    "rsi_14": 55.3,
    "macd": { "macd_line": 1.2, "signal_line": 0.8, "histogram": 0.4 },
    "bollinger": { "upper": 160.0, "middle": 150.0, "lower": 140.0 }
  }
}
```

**AgentsResponse:**
```json
{
  "agents": [
    { "agent_id": 1, "name": "MM-001", "total_pnl": 1234.56, "equity": 105000.0, "tier": 1 }
  ],
  "total_count": 25000
}
```

**RiskMetricsResponse:**
```json
{
  "agent_id": 1,
  "sharpe": 1.5,
  "sortino": 2.1,
  "max_drawdown": 0.05,
  "var_95": 0.02,
  "volatility": 0.15,
  "total_return": 0.05
}
```

### Design Principles
- **Declarative**: Typed request/response structs, pure handler functions
- **Modular**: Data service isolated from control endpoints (api.rs)
- **SoC**: Handlers extract from state, computation in simulation thread

### Exit Criteria
```
cargo build --package server  # ✅ Compiles without errors
cargo test --package server   # ✅ 20 tests pass
```

### Notes
- SimData hook implementation deferred to V4.4 (requires simulation thread integration)
- All endpoints return empty data until hook populates SimData
- Axum 0.8 uses `{param}` syntax instead of `:param` for path parameters
- V4.4 will implement real-time data population and WebSocket `/stream` endpoint

---

## 2026-01-10: V4.2 Services Foundation

### Summary
Implemented Axum async server as bridge between sync simulation and React frontend. New `server` crate provides HTTP/WebSocket endpoints for real-time tick streaming, health checks, and simulation control. Server runs simulation in background thread, broadcasts updates via tokio broadcast channels. Frontend hook `useWebSocket` connects to tick stream.

### Files

| File | Changes |
|------|---------|
| `crates/server/Cargo.toml` | New crate with axum 0.8, tokio, tower-http, serde |
| `crates/server/src/lib.rs` | Module exports, crate documentation |
| `crates/server/src/app.rs` | Axum router with routes, CORS, tracing middleware |
| `crates/server/src/state.rs` | `ServerState` with broadcast channels, metrics |
| `crates/server/src/error.rs` | `AppError` with HTTP status mapping |
| `crates/server/src/bridge.rs` | `TickData`, `SimUpdate`, `SimCommand` message types |
| `crates/server/src/hooks.rs` | `BroadcastHook` implementing `SimulationHook` |
| `crates/server/src/routes/mod.rs` | Route module organization |
| `crates/server/src/routes/health.rs` | `/health`, `/health/ready` endpoints |
| `crates/server/src/routes/ws.rs` | `/ws` WebSocket upgrade handler |
| `crates/server/src/routes/api.rs` | `/api/status`, `/api/command` REST endpoints |
| `src/main.rs` | Added `--server` mode with `run_with_server()` function |
| `Cargo.toml` (workspace) | Added server crate, tokio 1.42 |
| `docker-compose.yaml` | Added `server` and `frontend` services, profiles for tui/headless |
| `dockerfile/Dockerfile.server` | Multi-stage build for server mode |
| `frontend/src/hooks/useWebSocket.ts` | React hook for WebSocket connection |
| `frontend/src/hooks/index.ts` | Hooks barrel file |

### Server Architecture

**Endpoints:**
- `GET /health` - Liveness probe (tick, agents, uptime, ws_connections)
- `GET /health/ready` - Readiness probe (ready, reason, sim_running, sim_finished)
- `GET /ws` - WebSocket upgrade for real-time tick stream
- `GET /api/status` - Current simulation state
- `POST /api/command` - Send command (Start/Pause/Toggle/Step/Quit)

**Channel Bridge:**
```
Simulation (sync thread)      Server (async tokio)
       |                            |
       |-- BroadcastHook ---------->|-- broadcast::Sender<TickData>
       |      on_tick_end()         |       to WebSocket clients
       |                            |
       |<-- crossbeam::Receiver ----|-- SimCommand from clients
```

**Design Principles:**
- Declarative: Routes via Axum type-safe routing, message types define protocol
- Modular: Server crate independent of TUI, hooks decoupled from server internals
- SoC: Simulation runs sync loop, server observes via hooks, clients receive via WS

### Usage

```bash
# Run with server mode (replaces TUI)
cargo run -- --server --server-port 8001

# With docker
docker compose up server frontend
```

### Frontend Integration

New `useWebSocket` hook provides:
```typescript
const { tickData, connectionState, sendCommand } = useWebSocket();

// Start simulation
sendCommand('Start');

// Access real-time data
if (tickData) {
  console.log(`Tick: ${tickData.tick}, Trades: ${tickData.total_trades}`);
}
```

### Exit Criteria
- `cargo build --release` passes with server crate
- `cargo run -- --server` starts Axum server on port 8001
- `/health` returns JSON with tick/agents/uptime
- `/ws` accepts WebSocket connections
- Frontend hook connects and receives tick updates

### Notes
- V4.2 is read-only: server observes simulation state, no order submission
- WebSocket uses tokio broadcast channel (lagged clients skip messages)
- Simulation starts paused; send `{"command": "Start"}` to `/api/command`
- TUI and headless modes still available via `--headless` flag (no `--server`)

---

## 2026-01-10: V4.1 Web Frontend Landing & Config Pages

### Summary
Implemented React frontend with Landing and Config pages as first step toward V4 Web Frontend. Uses Vite + TypeScript + Tailwind CSS. Dockerized development workflow with hot reload, Prettier formatting, and production builds. Config page mirrors `src/config.rs` SimConfig structure with preset system (6 built-in presets + custom localStorage-backed presets).

### Files

| File | Changes |
|------|---------|
| `frontend/package.json` | React 19, React Router 7, Vite 6, TypeScript 5.7, Tailwind 3.4 |
| `frontend/vite.config.ts` | Dev server config with API proxy to port 8001 |
| `frontend/tailwind.config.ts` | Trading-themed colors (primary blue, accent green/red) |
| `frontend/src/types/config.ts` | `SimConfig`, `Sector`, `MarketRegime` types matching Rust |
| `frontend/src/config/defaults.ts` | `DEFAULT_CONFIG` + 5 built-in presets (Demo, Stress Test, etc.) |
| `frontend/src/components/ui/` | Accordion, Button, Input components (Tailwind-styled) |
| `frontend/src/components/config/` | 9 config sections, SymbolsEditor for multi-symbol management |
| `frontend/src/pages/LandingPage.tsx` | Hero section with feature bullets, Run Simulation + Configure CTAs |
| `frontend/src/pages/ConfigPage.tsx` | Full SimConfig form with collapsible sections, preset load/save |
| `frontend/src/pages/SimulationPage.tsx` | Placeholder for V4.2 |
| `dockerfile/Dockerfile.frontend` | Multi-stage: deps, dev, format, typecheck, builder, prod |
| `dockerfile/nginx.frontend.conf` | SPA fallback + API/WebSocket proxy to data-service |
| `docker-compose.frontend.yaml` | FE development: dev server, format, typecheck, build services |
| `docker-compose.tui.yaml` | Renamed from docker-compose.yaml (TUI-only simulation) |

### Frontend Architecture

**Stack:**
- React 19 + React Router 7 (SPA with client-side routing)
- Vite 6 (fast HMR, TypeScript, Tailwind integration)
- Tailwind CSS 3.4 (utility-first, custom trading theme)
- Prettier 3.4 (code formatting via Docker)

**Page Structure:**
- `/` - Landing: Hero, feature bullets, "Run Simulation" and "Configure" buttons
- `/config` - Config: Preset selector, 9 collapsible config sections, save preset
- `/sim` - Simulation: Placeholder (V4.2 will add WebSocket charts)

**Config Sections:**
1. Simulation Control (ticks, tick_ms, initial_cash, seed)
2. Symbols Editor (add/remove symbols with sector dropdown)
3. Tier 1 Agents (fundamental traders count/cash)
4. Tier 2 Agents (reactive traders count/cash, wake conditions)
5. Tier 3 Pool (noise traders count/cash)
6. Market Maker (spread, inventory limits, quote refresh)
7. Noise Trader (order size range, hold ticks)
8. Quant Strategy (momentum/mean-reversion/breakout weights)
9. Events (news frequency, earnings calendar)

### Docker Development Workflow

```powershell
# Start dev server with hot reload (port 5173)
docker compose -f docker-compose.frontend.yaml up frontend-dev

# Format all TypeScript files with Prettier
docker compose -f docker-compose.frontend.yaml run --rm format

# TypeScript type checking
docker compose -f docker-compose.frontend.yaml run --rm typecheck

# Production build (outputs to frontend/dist/)
docker compose -f docker-compose.frontend.yaml run --rm build
```

### Exit Criteria
```
docker compose -f docker-compose.frontend.yaml run --rm typecheck  # ✅ tsc passes
docker compose -f docker-compose.frontend.yaml run --rm build      # ✅ Vite bundles (252KB JS, 13KB CSS)
docker compose -f docker-compose.frontend.yaml run --rm format     # ✅ 21 files formatted
docker compose -f docker-compose.frontend.yaml up frontend-dev     # ✅ Dev server on :5173
```

### Notes
- Desktop-only design (min-width 1024px assumed)
- Preset storage: Built-in presets in code, custom presets in localStorage (SQLite backend in V4.2+)
- No form validation in V4.1 (deferred to later)
- SimConfig types match Rust `src/config.rs` structure for future API integration

---

## 2026-01-09: V3.9 Minimal Storage Infrastructure

### Summary
Implemented minimal storage layer as foundation for V4+ features. Storage crate provides trade history (append-only), candle aggregation (OHLCV), and portfolio snapshots via `SimulationHook` trait. Designed for headless simulation mode to generate data for future versions: V4 (Web Frontend queries storage), V5 (Feature Engineering ML), V6 (RL training on historical data), and V7 (Portfolio Manager Game replay/leaderboards).

### Files

| File | Changes |
|------|---------|
| `crates/storage/src/lib.rs` | Storage crate with declarative, modular, SoC philosophy |
| `crates/storage/src/schema.rs` | SQLite schema: trades, candles, portfolio_snapshots tables |
| `crates/storage/src/candles.rs` | `CandleAggregator` with in-memory buffering, flush on period end |
| `crates/storage/src/hook.rs` | `StorageHook` implementing `SimulationHook` trait |
| `crates/storage/src/tests.rs` | Integration test placeholder |
| `crates/storage/Cargo.toml` | Dependencies: rusqlite 0.38, serde_json, parking_lot |
| `Cargo.toml` (workspace) | Added storage crate, rusqlite with bundled feature |
| `docker-compose.yaml` | Added `./data:/data` volume mounts, `STORAGE_PATH` env var |
| `.gitignore` | Ignore `/data/`, `*.db` files |
| `src/main.rs` | Added `--storage-path` CLI flag, StorageHook integration (headless mode only) |

### Storage Architecture

**Philosophy:** V3.9 provides shared infrastructure for all future versions (V4-V7).

**Three Data Streams:**

1. **Trade History (Append-Only Event Log)**
   - Schema: `(tick, symbol, price, quantity, buyer_id, seller_id)`
   - Purpose: V4 (charting), V5 (features), V6 (RL rewards), V7 (game replay)
   - No updates or deletes

2. **Candle Aggregation (Time-Series OLAP)**
   - Schema: `(symbol, timeframe, tick_start, open, high, low, close, volume)`
   - Timeframes: 1m, 5m, 1h (configurable via `StorageConfig`)
   - In-memory buffer with periodic flush (on candle period completion)
   - Used by: V4 (frontend charts), V5/V6 (RL observations)

3. **Portfolio Snapshots (Analysis Checkpoints)**
   - Schema: `(tick, agent_id, cash, positions_json, realized_pnl, equity)`
   - Frequency: Every 1000 ticks (configurable)
   - Purpose: V4 (analytics), V6 (episode evaluation), V7 (leaderboards)

### Implementation Details

**StorageHook Pattern:**
- Implements `SimulationHook` trait (V3.6 hook system)
- Interior mutability via `Mutex<Connection>` (trait requires `&self`)
- Hooks: `on_trades()` for trade persistence, `on_tick_end()` for candle flush

**CandleAggregator:**
- Tracks current candles per symbol: `HashMap<Symbol, (tick_start, Candle)>`
- Buffered completed candles: `Vec<(Symbol, tick_start, Candle)>`
- Flush pattern: `std::mem::take(&mut self.completed)`

**Type Conversions:**
- `Price` (i64) → SQLite i64 (direct)
- `Quantity` (u64) → SQLite i64 (via `i64::try_from()`)
- `AgentId` (u64) → SQLite i64 (via `i64::try_from()`)
- `positions` (HashMap) → JSON string (via `serde_json::json!()`)

### Docker Integration

**Volume Mounts:**
```yaml
volumes:
  - ./data:/data  # V3.9: Persistent storage
environment:
  - STORAGE_PATH=/data/sim.db
```

**Use Cases:**
- TUI mode: Ephemeral (no storage needed)
- Headless mode: Persistent storage for V4+ data pipelines

### Deferred Features

**Deferred to Later Versions:**
- ❌ REST APIs for real-time queries → V4 (Web Frontend)
- ❌ Fill-level events (finer granularity) → V5/V6 (if RL training demands it)
- ❌ Game save/resume functionality → V7 (Portfolio Manager Game)
- ❌ Agent-level trade attribution → V7 (leaderboards)

**Why Minimal Scope:**
- V3.9 provides data persistence only
- V4 will add query APIs (Axum endpoints reading from storage)
- V5/V6 will consume stored data for training
- V7 will add game-specific persistence features

### Exit Criteria
```
cargo fmt              # ✅ No formatting issues
cargo clippy --all-targets -- -D warnings  # ✅ No warnings
cargo test --package storage  # ✅ 8 tests pass
cargo run --headless --ticks 100 --storage-path ./test.db  # ✅ Storage integration works
```

### Notes
- ✅ Storage integration with main.rs completed (headless mode only)
- ✅ CLI flag `--storage-path` added with env var support `STORAGE_PATH`
- ✅ Graceful error handling if storage initialization fails
- `persist_snapshots()` method ready but needs agent summary access (future enhancement)
- Clippy satisfied with `i64::try_from()` for u64→i64 conversions
- `is_multiple_of()` used for snapshot interval checks (clippy suggestion)

---

## 2026-01-08: V3.8 Performance Profiling

### Summary
Added fine-grained parallelization control for profiling. All parallel functions accept runtime `force_sequential` override via `ParallelizationConfig`. CLI/environment variables control 9 phases independently. PowerShell script automates benchmarking.

### Files

| File | Changes |
|------|---------|
| `crates/simulation/src/parallel.rs` | Added `force_sequential: bool` parameter to all 10 functions |
| `crates/simulation/src/config.rs` | `ParallelizationConfig` struct with 9 boolean fields |
| `crates/simulation/src/runner.rs` | Updated 7 parallel call sites to use config flags |
| `crates/sim-core/src/batch_auction.rs` | Added `force_sequential` to `run_parallel_auctions` |
| `src/main.rs` | 9 CLI args (env var support): `PAR_AGENT_COLLECTION`, `PAR_INDICATORS`, etc. |
| `run_profiling.ps1` | Automated profiling script (11 configs × 3 trials) |

### Parallelization Control

9 independently controllable phases:

| Phase | Config Field | Description |
|-------|-------------|-------------|
| 3 | `parallel_indicators` | Build indicator snapshot per-symbol |
| 4 | `parallel_agent_collection` | Collect agent actions |
| 5 | `parallel_order_validation` | Validate orders |
| 6 | `parallel_auctions` | Batch auctions across symbols |
| 9 | `parallel_candle_updates` | Update candles per-symbol |
| 9 | `parallel_trade_updates` | Update recent trades per-symbol |
| 10 | `parallel_fill_notifications` | Process fill notifications |
| 10 | `parallel_wake_conditions` | Restore T2 wake conditions |
| 11 | `parallel_risk_tracking` | Update risk tracking |

### Usage

```bash
# Disable specific phase
PAR_AUCTIONS=false cargo run --release --all-features -- --headless --ticks 1000

# Automated profiling (Windows)
.\run_profiling.ps1
# Outputs: profiling_results.csv (config_name, trial, elapsed_ms, ticks_per_sec, total_trades)
```

### Notes
- 2^9 = 512 total permutations; script tests 11 meaningful configs
- Uses exact same agent configuration as default run (not minimal agents)
- Runtime control avoids recompilation

---

## 2026-01-07: V3.7 Containerization & CLI

### Summary
Added Docker support with ttyd for browser-accessible TUI. `--headless` flag enables benchmarks/CI without terminal. Environment variables (`SIM_*`) override config.

### Files

| File | Changes |
|------|---------|
| `src/main.rs` | `Args` struct with clap, `run_headless()`, env var support |
| `Cargo.toml` | Added `clap` dependency |
| `dockerfile/Dockerfile.simulation` | Distroless image for headless benchmarks |
| `dockerfile/Dockerfile.tui` | Debian + ttyd for browser TUI |
| `docker-compose.yaml` | Local dev setup with both services |

### Usage

```bash
# Local headless benchmark
cargo run --release -- --headless --ticks 10000

# Docker TUI (browser at http://localhost:7681)
docker compose up tui

# Docker headless benchmark
docker compose up simulation
```

### CLI Args

| Flag | Env Var | Description |
|------|---------|-------------|
| `--headless` | `SIM_HEADLESS` | Disable TUI |
| `--ticks N` | `SIM_TICKS` | Total simulation ticks |
| `--tier1 N` | `SIM_TIER1` | Tier 1 agent count |
| `--tier2 N` | `SIM_TIER2` | Tier 2 agent count |
| `--pool-size N` | `SIM_POOL_SIZE` | Background pool size |
| `--tick-delay N` | `SIM_TICK_DELAY` | Delay between ticks (ms) |

### Notes
- ttyd installed from GitHub releases (not in Debian repos)
- Distroless runtime for headless (~20MB image)
- Debian slim for TUI (ttyd requires libc)

---

## 2026-01-07: V3.6 Hooks System

### Summary
Added extensible hook infrastructure for simulation events. Hooks receive **owned data** (snapshots, cloned orders/trades) to avoid borrow-checker issues with async/network use. TUI BookDepth widget removed (meaningless in batch auction mode).

### Files

| File | Changes |
|------|---------|
| `crates/simulation/src/hooks.rs` | `SimulationHook` trait, `HookContext`, `HookRunner` |
| `crates/simulation/src/metrics.rs` | `MetricsHook` implementation with atomic counters |
| `crates/simulation/src/runner.rs` | Hook integration in `step()` phases |
| `crates/tui/src/widgets/update.rs` | Added `Serialize` derives for network hooks |
| `crates/tui/src/app.rs` | Removed BookDepth, expanded risk panel |

### Design Decisions

1. **Owned data over references**: Hooks receive `Vec<Order>`, `Vec<Trade>`, `MarketSnapshot` — enables serialization, async, and avoids lifetime complexity
2. **`Arc<dyn SimulationHook>`**: Shared ownership for registering hooks across contexts
3. **Built-in `MetricsHook`**: Aggregates tick/trade/volume stats with atomics for thread-safety

---

## 2026-01-07: V3.5 Parallel Execution & Batch Auction

### Summary
Implemented parallel agent execution with rayon and switched order matching from continuous to batch auction. The `parallel` module provides declarative helpers that abstract over `par_iter`/`iter` with cfg logic in ONE place. Batch auction enables full parallelism across symbols by computing a single clearing price per symbol.

### Architecture

**Two-Phase Tick:**
1. **Collection Phase**: All agents run `on_tick()` in parallel, collecting orders
2. **Auction Phase**: Per-symbol batch auction computes clearing price, matches all crossing orders

**Key Insight**: Since we clear the book every tick anyway, batch auction semantics are natural. All agents see the same market state and compete in a single auction per tick.

### `parallel` Module (`crates/simulation/src/parallel.rs`)

Declarative helpers that keep `#[cfg(feature = "parallel")]` in ONE place:
- `map_slice()`, `filter_map_slice()` — parallel iteration over slices
- `map_indices()`, `filter_map_indices()` — index-based iteration
- `map_mutex_slice()`, `map_mutex_slice_ref()` — safe parallel access to `Mutex<T>` slices

### Batch Auction (`crates/sim-core/src/batch_auction.rs`)

**Clearing Price Algorithm:**
1. Collect all unique limit prices + reference price (fair_value or last_price)
2. For each candidate, compute executable volume (min of supply/demand)
3. Return price with maximum volume (prefer reference price on ties for stability)

**Price Stability Fix:** Reference price from `fundamentals.fair_value()` anchors clearing price, preventing wild oscillations when market orders dominate.

### Files Modified

| File | Changes |
|------|---------|
| `crates/simulation/src/parallel.rs` | New module: declarative parallel helpers |
| `crates/simulation/src/runner.rs` | Batch auction integration, reference price from fair_value |
| `crates/sim-core/src/batch_auction.rs` | New module: clearing price & matching with reference anchor |
| `crates/sim-core/Cargo.toml` | Added `rayon` optional dependency |

### Key Design Decisions

1. **Batch auction over continuous**: Enables full parallelism, natural fit for "clear book each tick"
2. **`parallel::` helpers**: Single source of truth for cfg logic, clean call sites
3. **Reference price anchor**: Uses `fair_value` (from fundamentals) to stabilize clearing price after events
4. **T3 retains continuous matching**: Background pool orders still use `process_order()` for different semantics

### Exit Criteria
```
cargo clippy           # ✅ No warnings
cargo test --workspace # ✅ All tests pass (parallel and sequential)
```

---

## 2026-01-07: V3.4 Background Agent Pool (Tier 3)

### Summary
Implemented Tier 3 Background Agent Pool for statistical order generation. A single pool instance trades all symbols, simulating 45,000+ background agents with ~2KB memory overhead. Orders go through the real matching engine and trade against T1/T2 agents and market makers.

### Architecture

**Single Pool, All Symbols:**
- One `BackgroundAgentPool` instance trades ALL configured symbols
- Randomly selects symbol per order based on activity
- Per-symbol sentiment tracking (sector news affects correct symbols)
- Aggregate P&L accounting via `BackgroundPoolAccounting`

**Memory Budget:**
- Config: ~200 bytes
- Sentiments: ~100 bytes per symbol
- Distributions: ~50 bytes
- Accounting: ~500 bytes
- RNG: ~200 bytes
- **Total: ~2 KB** (vs 90K individual agents)

### Module Structure (`crates/agents/src/tier3/`)

| File | Purpose |
|------|---------|
| `config.rs` | `MarketRegime`, `RegimePreset`, `BackgroundPoolConfig` |
| `distributions.rs` | `PriceDistribution`, `SizeDistribution` traits; `ExponentialPriceSpread`, `LogNormalSize` |
| `accounting.rs` | `BackgroundPoolAccounting`, `SanityCheckResult`, per-symbol P&L tracking |
| `pool.rs` | `BackgroundAgentPool`, `PoolContext`, `BACKGROUND_POOL_ID` sentinel |

### Order Generation

```rust
// Each tick, pool generates orders based on:
// 1. pool_size × base_activity (regime-dependent)
// 2. ±20% random variance
// 3. Sentiment-biased side selection
// 4. Log-normal size distribution
// 5. Exponential price spread from mid
```

**Market Regimes:**
| Regime | base_activity | Description |
|--------|---------------|-------------|
| Calm | 0.1 | 10% of pool trades/tick |
| Normal | 0.3 | 30% of pool trades/tick |
| Volatile | 0.6 | 60% of pool trades/tick |
| Crisis | 0.9 | 90% of pool trades/tick |

### Config Integration (`src/config.rs`)

```rust
// New Tier 3 fields
enable_background_pool: true,
background_pool_size: 45_000,
background_regime: MarketRegime::Normal,
t3_mean_order_size: 15.0,
t3_max_order_size: 100,
t3_order_size_stddev: 10.0,
t3_base_activity: Option<f64>,  // Override regime default
t3_price_spread_lambda: 20.0,
t3_max_price_deviation: 0.02,
```

### Simulation Integration (`runner.rs`)

**Phase 4 in `step()`:**
1. Build `PoolContext` with mid prices and active events
2. Call `pool.generate(&ctx)` → `Vec<Order>`
3. Process orders through existing `process_order()` (real matching)
4. Update `BackgroundPoolAccounting` with fills
5. Notify counterparty agents via `on_fill()`

```rust
// Declarative trade processing
let t3_trades: Vec<_> = t3_orders
    .into_iter()
    .flat_map(|order| {
        let (trades, _) = self.process_order(order);
        trades
    })
    .collect();

t3_trades.into_iter().for_each(|trade| {
    // Update accounting and notify agents
});
```

### TUI Updates

**StatsPanel shows:**
- `Agents: 12T1 + 3900T2 + 45000T3`
- `T3 Orders: X` (orders generated this tick)
- `Pool P&L: $Y` (realized P&L, green/red colored)

**SimUpdate new fields:**
- `tier3_count: usize`
- `t3_orders: usize`
- `background_pnl: f64`

### Files Modified

| File | Changes |
|------|---------|
| `crates/agents/src/tier3/*.rs` | New module: config, distributions, accounting, pool |
| `crates/agents/src/lib.rs` | Export tier3 module |
| `crates/agents/Cargo.toml` | Added `rand_distr = "0.4"` |
| `crates/simulation/src/runner.rs` | `background_pool` field, Phase 4 integration, `t3_orders_this_tick` stat |
| `crates/tui/src/widgets/update.rs` | `tier3_count`, `t3_orders`, `background_pnl` fields |
| `crates/tui/src/widgets/stats_panel.rs` | T3 count display, T3 orders/P&L line |
| `crates/tui/src/app.rs` | Wire T3 stats to StatsPanel |
| `src/config.rs` | T3 config fields with defaults |
| `src/main.rs` | Phase 5: BackgroundAgentPool creation, SimUpdate wiring |
| `Cargo.toml` (workspace) | Downgraded rand to 0.8 (rand_distr compatibility) |

### Key Design Decisions

1. **Single pool trades all symbols** (not per-symbol pools) - simpler accounting
2. **Real order matching** - T3 orders go through same engine as T1/T2
3. **BACKGROUND_POOL_ID = AgentId(0)** sentinel for all pool orders
4. **Append-only accounting** - fills recorded but never read during generation
5. **rand 0.8 + rand_distr 0.4** - required `r#gen` escape for Rust 2024

### Exit Criteria
```
cargo fmt              # ✅ No formatting issues
cargo clippy           # ✅ No warnings
cargo test --workspace # ✅ All tests pass (116 in agents crate)
```

---

## 2026-01-07: V3.3 Multi-Symbol Strategies

### Summary
Implemented two flagship multi-symbol strategies: PairsTrading (Tier 1) and SectorRotator (Tier 2). Added quant extensions for cointegration tracking and sector sentiment aggregation. Updated TUI to show Total P&L (realized + unrealized).

### Quant Extensions (`crates/quant/src/stats.rs`)

**CointegrationTracker:**
- Rolling OLS hedge ratio computation
- Spread z-score calculation for mean-reversion signals
- Configurable lookback window

**SectorSentimentAggregator:**
- `NewsEventLike` trait for decoupling from `news` crate
- Decay-weighted sentiment aggregation per sector
- Event expiration filtering by magnitude threshold

### PairsTrading Strategy (Tier 1)

```rust
pub struct PairsTradingConfig {
    pub symbol_a: Symbol,
    pub symbol_b: Symbol,
    pub entry_z_threshold: f64,   // Default: 2.0
    pub exit_z_threshold: f64,    // Default: 0.5
    pub max_position_per_leg: i64,
}
```

- Runs every tick (continuous spread monitoring)
- Uses `CointegrationTracker` for z-score signals
- Returns `AgentAction::multiple()` for simultaneous leg execution
- Declarative exit logic via `filter_map` patterns

### SectorRotator Strategy (Tier 2)

```rust
pub struct SectorRotatorConfig {
    pub symbols_per_sector: HashMap<Sector, Vec<Symbol>>,
    pub sentiment_scale: f64,      // ±30% allocation shift
    pub rebalance_threshold: f64,  // 5% drift threshold
}
```

- Wakes on `NewsEvent` for watched symbols
- Implements `initial_wake_conditions()` trait method (critical fix)
- Sentiment-driven allocation with clamping and normalization
- Multi-symbol rebalance orders via `flat_map` patterns

### TUI Updates

Changed P&L display from "Realized P&L" to "Total P&L":
- `AgentState::total_pnl(&prices)` computes realized + unrealized
- Unrealized = Σ (current_price - avg_cost) × quantity
- Agents sorted by total P&L descending

### Config & Simulation Integration

```rust
// New config fields
num_pairs_traders: 50,     // Tier 1 (included in specified_tier1_agents)
num_sector_rotators: 300,  // Special Tier 2 (added to tier2_count)
```

- PairsTrading cycles through symbol pairs
- SectorRotator watches all sectors with all symbols
- TUI widget counts updated: `tier2_count = num_tier2_agents + num_sector_rotators`

### Files Modified

| File | Changes |
|------|---------|
| `crates/quant/src/stats.rs` | `CointegrationTracker`, `SectorSentimentAggregator`, `NewsEventLike` |
| `crates/agents/src/tier1/strategies/pairs_trading.rs` | New: Tier 1 multi-symbol pairs strategy |
| `crates/agents/src/tier2/sector_rotator.rs` | New: Tier 2 sentiment-driven rotation |
| `crates/agents/src/state.rs` | Added `total_pnl()` method |
| `crates/simulation/src/runner.rs` | `agent_summaries()` returns total P&L |
| `crates/tui/src/widgets/update.rs` | `AgentInfo.total_pnl` field |
| `crates/tui/src/widgets/agent_table.rs` | "Total P&L" column, sorting by total_pnl |
| `src/config.rs` | `num_pairs_traders`, `num_sector_rotators`, `specified_tier1_agents()` |
| `src/main.rs` | `spawn_sector_rotators()`, pairs trading spawn, tier count fix |

### Key Fixes

1. **SectorRotator wake conditions**: Must implement `initial_wake_conditions(tick)` trait method, not just a helper method
2. **Tier 2 count**: `tier2_count = num_tier2_agents + num_sector_rotators` (was missing sector rotators)
3. **Declarative refactoring**: Converted for loops to `filter_map`, `flat_map`, `fold` patterns

### Exit Criteria
```
cargo fmt              # ✅ No formatting issues
cargo clippy           # ✅ No warnings  
cargo test --workspace # ✅ All tests pass
```

---

## 2026-01-06: V3.2 Tier 2 Reactive Agents

### Summary
Complete Tier 2 reactive agent system with proper wake condition lifecycle. 4000 T2 agents spawn with randomized strategies (1 entry + 1 exit + optional NewsReactor). Entry conditions removed at max capacity, exit conditions added/removed based on position state.

### Wake Condition Lifecycle

**Problem:** T2 agents triggered repeatedly on every price cross because conditions were never removed after acting.

**Solution:** `post_fill_condition_update()` returns `ConditionUpdate` (add/remove lists):
- Entry conditions (ThresholdBuyer): Remove at max capacity, re-add when flat
- Exit conditions (StopLoss/TakeProfit): Add when opening position (computed from cost_basis), remove when closing

```
At startup: ThresholdBuyer/Seller → PriceCross conditions; StopLoss/TakeProfit → NOT registered
After BUY: If at_capacity → REMOVE entry; If just opened → ADD exits
After SELL: If closed to flat → REMOVE exits, ADD entry back
```

### ReactiveAgent Implementation
- `AgentState` field for position/cash tracking (SoC: AgentState owns state, ReactiveAgent owns strategy)
- `Agent` trait implementation with `on_tick()` and `on_fill()`
- `compute_condition_update()` detects state transitions and returns add/remove lists
- Optimized T1/T2 iteration using index arrays (avoids iterating all 4000 agents)

### Config & TUI
- `SimConfig`: Added `num_tier2_agents` (default 4000), `t2_initial_cash`, `t2_max_position`
- TUI: T1/T2 agent counts displayed separately (T1 cyan, T2 magenta)

### Files Modified
| File | Changes |
|------|---------|
| `crates/agents/src/traits.rs` | Added `post_fill_condition_update()` to Agent trait |
| `crates/agents/src/tier2/agent.rs` | AgentState integration, Agent trait impl, `compute_condition_update()` |
| `crates/agents/src/lib.rs` | Export ReactiveAgent, ReactivePortfolio, ReactiveStrategyType |
| `crates/simulation/src/runner.rs` | Collect and apply condition updates, T1/T2 index optimization |
| `src/config.rs` | `num_tier2_agents`, `t2_initial_cash`, `t2_max_position` |
| `src/main.rs` | `spawn_tier2_agents()`, random strategy generators |
| `crates/tui/src/widgets/*.rs` | T1/T2 count display |

### Exit Criteria
```
cargo fmt              # ✅ No formatting issues
cargo clippy           # ✅ No warnings
cargo test --workspace # ✅ All tests pass
```

---

## 2026-01-05: V3.1 Multi-Symbol & IOC Orders

### Summary
Refactored `AgentState` to multi-symbol HashMap-based architecture and implemented IOC (Immediate-Or-Cancel) order behavior. All strategies updated to per-symbol position tracking.

### Multi-Symbol AgentState

#### Core Changes (`agents/state.rs`)
- `PositionEntry { quantity: i64, avg_cost: f64 }` for per-symbol tracking
- `positions: HashMap<Symbol, PositionEntry>` replaces `position: i64`
- New API: `on_buy(symbol, qty, value)`, `on_sell(symbol, qty, value)`, `position_for(symbol)`
- Weighted average cost tracking for realized P&L calculation
- `equity(&HashMap<Symbol, Price>)` for multi-symbol mark-to-market

#### Agent Trait Extensions (`agents/traits.rs`)
- `positions()`, `position_for(symbol)`, `watched_symbols()`, `equity()`, `equity_for()`

#### Strategy Updates
All 7 strategies (NoiseTrader, MarketMaker, Momentum, TrendFollower, MacdCrossover, BollingerReversion, VwapExecutor) updated to use `AgentState::new(cash, &[symbols])` and per-symbol `on_buy`/`on_sell`.

### IOC Order Expiration
- `OrderBook::clear()` removes all resting orders at end of each tick
- Each tick starts fresh - cleaner mental model, no stale order accumulation
- Market makers now quote every tick (`refresh_interval: 1`)

### Files Modified
| File | Changes |
|------|---------|
| `crates/agents/src/state.rs` | Complete refactor to multi-symbol |
| `crates/agents/src/traits.rs` | New trait methods |
| `crates/agents/src/strategies/*.rs` | All 7 strategies updated |
| `crates/sim-core/src/order_book.rs` | Added `clear()` method |
| `crates/simulation/src/runner.rs` | Multi-symbol equity, IOC clearing |
| `crates/tui/src/widgets/agent_table.rs` | Per-symbol position display |
| `src/config.rs` | Default `mm_refresh_interval: 1` |

### Exit Criteria
```
cargo fmt --check      # ✅ No formatting issues
cargo clippy           # ✅ No warnings  
cargo test --workspace # ✅ All 224 tests pass
```

---

## 2026-01-04: V2.4 - Fundamentals, Events & TUI Controls

### Summary
Complete news/events system with Gordon Growth Model fair value, sector-based event generation, TUI start/stop control, and event distribution fixes.

### Completed

#### News & Events System (`news/`)
- ✅ `FundamentalEvent` enum: `EarningsSurprise`, `GuidanceChange`, `RateDecision`, `SectorNews`
- ✅ `NewsEvent`: Time-bounded event with sentiment, magnitude, decay
- ✅ `NewsGenerator`: Configurable event generation with min intervals
- ✅ `NewsGeneratorConfig`: Per-event-type frequency and magnitude settings

#### Fundamentals Model (`news/fundamentals.rs`)
- ✅ `Fundamentals`: Per-symbol EPS, growth estimate, payout ratio
- ✅ `MacroEnvironment`: Risk-free rate (4%), equity risk premium (5%)
- ✅ `SymbolFundamentals`: Container with `fair_value(&symbol)` method
- ✅ Gordon Growth Model: `fair_value = D1 / (r - g)` with P/E fallback

#### Sector Support (`types/`)
- ✅ `Sector` enum: 10 sectors (Tech, Energy, Finance, Healthcare, Consumer, Industrials, Materials, Utilities, RealEstate, Communications)
- ✅ `SymbolConfig.sector` field for sector assignment
- ✅ `SectorModel`: Maps symbols to sectors for sector-wide events

#### TUI Start/Stop Control
- ✅ `SimCommand` enum: `Start`, `Pause`, `Toggle`, `Quit`
- ✅ Bidirectional channels: `SimUpdate` (Sim→TUI), `SimCommand` (TUI→Sim)
- ✅ TUI starts paused, Space key toggles running state
- ✅ Header shows PAUSED (red) / RUNNING (yellow) / FINISHED (green)
- ✅ Footer shows Space key hint

#### Event Distribution Fix
- ✅ **Bug**: Fixed seed caused same symbols to always get same event outcomes
- ✅ **Fix**: Generate event value (surprise/growth) BEFORE selecting symbol
- ✅ **Result**: Events now distributed fairly across all symbols

#### Guidance Range Fix
- ✅ **Bug**: Growth range (-2% to +12%) could exceed required return (9%)
- ✅ **Fix**: Capped growth_range max at 7% to prevent Gordon Growth Model breakdown
- ✅ **Result**: Fair values stay bounded, no runaway $400+ prices

### Technical Notes

**Market Behavior (Mean-Reverting):**
- Tick-level simulation is realistic for liquid equity markets
- Prices anchor to fair value (~$52.50 from Gordon Growth Model)
- Momentum strategies have low activity (RSI rarely hits 30/70 extremes)
- This matches real HFT environments where momentum has negative alpha

**Fair Value Calculation:**
```
EPS = initial_price / 20  (P/E of 20)
D1 = EPS × payout_ratio × (1 + growth) = $5 × 0.40 × 1.05 = $2.10
r = risk_free + equity_premium = 4% + 5% = 9%
g = growth_estimate = 5%
fair_value = $2.10 / (0.09 - 0.05) = $52.50
```

**Event Frequencies (Default):**
| Event | Probability | Min Interval | Duration |
|-------|-------------|--------------|----------|
| Earnings | 0.2% | 100 ticks | 50 ticks |
| Guidance | 0.1% | 200 ticks | 30 ticks |
| Rate Decision | 0.05% | 500 ticks | 100 ticks |
| Sector News | 0.3% | 50 ticks | 40 ticks |

### Files Modified
| File | Changes |
|------|---------|
| `crates/news/src/events.rs` | `FundamentalEvent`, `NewsEvent` |
| `crates/news/src/fundamentals.rs` | Gordon Growth Model, `SymbolFundamentals` |
| `crates/news/src/generator.rs` | Event generation, correlation fix |
| `crates/news/src/config.rs` | Event frequencies, guidance range cap |
| `crates/news/src/sectors.rs` | `SectorModel` for sector lookup |
| `crates/types/src/config.rs` | `Sector` enum, `SymbolConfig.sector` |
| `crates/tui/src/lib.rs` | `SimCommand` enum |
| `crates/tui/src/app.rs` | Start/stop control, status display |
| `crates/simulation/src/runner.rs` | Event processing, fundamentals integration |
| `src/main.rs` | Command channel, paused start |
| `docs/executive_summary.md` | Business overview |
| `docs/technical_summary.md` | Technical architecture |

### Exit Criteria
```
cargo fmt --check      # ✅ No formatting issues
cargo clippy           # ✅ No warnings  
cargo test --workspace # ✅ All 213 tests pass
```

---

## 2026-01-04: V2.3 - Multi-Symbol Infrastructure

### Summary
Complete multi-symbol foundation: `StrategyContext` for agents, `MarketView` trait, multi-symbol TUI with symbol tabs/overlay, configurable symbol list, multi-symbol simulation runner, and agent distribution across symbols.

### Completed

#### Part 1: Core Infrastructure (`sim-core/`, `agents/`)

##### MarketView Trait (`sim-core/src/market.rs`)
- ✅ `MarketView` trait: Read-only interface for market state
  - `symbols()`: List of available symbols
  - `has_symbol()`: Check symbol existence
  - `mid_price()`, `best_bid()`, `best_ask()`, `last_price()`: Price queries
  - `snapshot()`: Full order book snapshot
  - `total_bid_volume()`, `total_ask_volume()`: Liquidity queries
- ✅ `SingleSymbolMarket<'a>`: Adapter wrapping `&OrderBook` to implement `MarketView`
- ✅ `Market` struct: Multi-symbol container with `HashMap<Symbol, OrderBook>`

##### StrategyContext (`agents/src/context.rs`)
- ✅ `StrategyContext<'a>`: Unified context replacing `MarketData`
  - References instead of clones: `&'a dyn MarketView`, `&'a HashMap<Symbol, Vec<Candle>>`, etc.
  - Single-symbol convenience: `primary_symbol()`, `mid_price()`, `last_price()`
  - Multi-symbol access: `market()` for full `MarketView` interface

##### Agent Trait Migration
- ✅ `Agent::on_tick()` signature changed: `&MarketData` → `&StrategyContext<'_>`
- ✅ All 7 strategies migrated to `StrategyContext` API
- ✅ Removed deprecated `MarketData` struct (no external consumers)

#### Part 2: Multi-Symbol TUI (`tui/`)

##### SimUpdate (`tui/src/widgets/update.rs`)
- ✅ Per-symbol data: `price_history`, `bids`, `asks`, `last_price` as `HashMap<Symbol, _>`
- ✅ `selected_symbol: usize` - Current tab index
- ✅ Helper methods: `current_symbol()`, `current_price_history()`, `current_bids()`, etc.

##### TuiApp (`tui/src/app.rs`)
- ✅ Symbol navigation: `Tab`/`→`/`←`/`1-9` keys
- ✅ `O` key: Toggle price overlay mode (all symbols on one chart)
- ✅ `draw_symbol_tabs()` for tab bar rendering
- ✅ Fixed: Sync `self.state.selected_symbol` with `TuiApp.selected_symbol` on updates

##### PriceChart (`tui/src/widgets/price_chart.rs`)
- ✅ `PriceChart::multi()` constructor for overlay mode
- ✅ Different colors per symbol in overlay view

#### Part 3: Multi-Symbol Config (`src/config.rs`)

##### SymbolSpec Struct
- ✅ `SymbolSpec { symbol: String, initial_price: Price }`
- ✅ `SimConfig.symbols: Vec<SymbolSpec>` for multi-symbol configuration
- ✅ Accessor methods: `get_symbols()`, `primary_symbol()`, `symbol_count()`

#### Part 4: Multi-Symbol Simulation (`simulation/`)

##### SimulationConfig (`simulation/src/config.rs`)
- ✅ `symbol_configs: Vec<SymbolConfig>` replaces single `symbol_config`

##### Simulation Runner (`simulation/src/runner.rs`)
- ✅ `market: Market` replaces single `book: OrderBook`
- ✅ Per-symbol HashMaps: `candles`, `current_candles`, `recent_trades`, `total_shares_held`
- ✅ `process_order()` routes to correct book via `market.get_book_mut(&order.symbol)`

#### Part 5: Agent Distribution (`src/main.rs`)

##### Distribution Strategy
- ✅ **Market Makers**: Distributed across symbols (N / num_symbols each, remainder random)
- ✅ **Noise Traders**: Distributed across symbols (same logic)
- ✅ **Quant Strategies**: Randomly assigned to symbols (equal probability)
- ✅ **Random Fill Agents**: Randomly assigned to symbols

##### Noise Trader Balance Fix
- ✅ `nt_initial_position`: 50 → 0 (start flat, no long imbalance)
- ✅ `nt_initial_cash`: $95,000 → $100,000 (equals quant_initial_cash)
- ✅ Cash adjustment formula for different symbol prices:
  ```
  adjusted_cash = quant_initial_cash - (nt_initial_position × symbol_price)
  ```

##### Agent Counts (3 symbols example)
With `num_market_makers: 100`, `num_noise_traders: 400`:
- Each symbol gets ~33 MMs, ~133 NTs (total unchanged)
- Quant strategies randomly distributed across symbols

### Technical Notes

**TUI Layout:**
```
┌──[1:Food] [2:Energy] [3:Hotels]──────────────────────────┐
│ Price Chart (selected symbol, or overlay with O key)     │
├────────────────────────┬─────────────────────────────────┤
│ Book Depth             │ Stats (selected symbol)         │
├────────────────────────┴─────────────────────────────────┤
│ Agent P&L (portfolio)          │ Risk (portfolio-level)  │
└──────────────────────────────────────────────────────────┘
```

**Price Drop Issue (Resolved):**
- **Cause**: Initial long imbalance from MMs (500 shares each) + old NTs (50 shares each)
- **Fix**: NTs start flat (0 position), agents distributed across symbols equally

### Files Modified
| File | Changes |
|------|---------|
| `crates/sim-core/src/market.rs` | `MarketView` trait, `SingleSymbolMarket`, `Market` |
| `crates/agents/src/context.rs` | `StrategyContext<'a>` struct |
| `crates/agents/src/traits.rs` | `Agent::on_tick(&StrategyContext)`, removed MarketData |
| `crates/simulation/src/config.rs` | `symbol_configs: Vec<SymbolConfig>` |
| `crates/simulation/src/runner.rs` | Multi-symbol: Market, per-symbol HashMaps |
| `crates/tui/src/app.rs` | Symbol tabs, navigation, selected_symbol sync fix |
| `crates/tui/src/widgets/update.rs` | Multi-symbol SimUpdate and AgentInfo |
| `crates/tui/src/widgets/price_chart.rs` | Multi-symbol overlay support |
| `src/config.rs` | SymbolSpec, multi-symbol config, nt_initial_position=0 |
| `src/main.rs` | Agent distribution across symbols, spawn_agent with symbol param |

### Exit Criteria
```
cargo fmt --check      # ✅ No formatting issues
cargo clippy           # ✅ No warnings
cargo test --workspace # ✅ All 193 tests pass
```

---

## 2026-01-04: V2.2 - Slippage & Partial Fills

### Completed

#### Fill Type (V2.2)
- ✅ `FillId` newtype for unique fill identifiers
- ✅ `Fill` struct: Represents atomic execution at a single price level
  - Distinct from `Trade` (aggregated view)
  - Tracks `aggressor_id`, `resting_id`, `aggressor_side`
  - Includes `reference_price` for slippage calculation
  - `slippage()` and `slippage_bps()` methods

#### Slippage & Market Impact (V2.2)
- ✅ `SlippageConfig`: Configures impact model
  - `enabled`: Master toggle for slippage tracking
  - `impact_threshold_bps`: Minimum order size before impact applies
  - `linear_impact_bps`: Impact coefficient
  - `use_sqrt_model`: Use square-root impact (more realistic)
- ✅ `SlippageMetrics`: Aggregates fill metrics
  - VWAP calculation across multiple fills
  - `levels_crossed`, `best_fill_price`, `worst_fill_price`
  - `slippage_buy()`, `slippage_sell()`, `slippage_bps(side)`
- ✅ `SlippageCalculator` (`slippage.rs`): Pre-trade impact estimation
  - `available_liquidity()`: Total volume on opposite side
  - `liquidity_ratio_bps()`: Order size as fraction of liquidity
  - `is_large_order()`: Above impact threshold check
  - `estimate_impact_bps()`: Linear or sqrt impact model
  - `estimate_execution_price()`: Expected VWAP
  - `analyze_impact()`: Full pre-trade analysis
- ✅ `ImpactEstimate`: Pre-trade analysis result struct

#### Matching Engine Updates
- ✅ `MatchResult` now includes:
  - `fills: Vec<Fill>`: Per-level execution details
  - `slippage_metrics: SlippageMetrics`: Aggregated metrics
  - `vwap()`: Volume-weighted average price
  - `levels_crossed()`: Number of price levels hit
  - `has_fills()`: Check for fill existence
- ✅ `match_order_with_reference()`: Match with explicit reference price
- ✅ Fills generated alongside trades for each level crossed
- ✅ Reference price defaults to mid price at order submission

#### OrderBook Enhancements
- ✅ `total_bid_volume()`: Sum of all bid quantities
- ✅ `total_ask_volume()`: Sum of all ask quantities
- ✅ `bid_depth_to_price(min_price)`: Liquidity above threshold
- ✅ `ask_depth_to_price(max_price)`: Liquidity below threshold

### Technical Notes

**Impact Model Formula:**
- Linear: `impact_bps = coefficient * (order_size / liquidity) * 100`
- Sqrt: `impact_bps = coefficient * sqrt(order_size / liquidity) * 100`

**Why separate Fill from Trade:**
- Fills are atomic executions at exactly one price
- Trades aggregate multiple fills for reporting
- Fills enable precise slippage measurement per level
- Supports future features like transaction cost analysis (TCA)

**Slippage Sign Convention:**
- Positive slippage = worse execution (paid more / received less)
- Buy: slippage = fill_price - reference_price
- Sell: slippage = reference_price - fill_price

---

## 2026-01-04: V2.1 - Position Limits & Short-Selling Infrastructure

### Completed

#### Position Limits & Short-Selling (V2.1)
- ✅ `SymbolConfig`: Tracks `shares_outstanding` per symbol
- ✅ `ShortSellingConfig`: Controls short selling rules
  - `enabled`: Master toggle for short selling
  - `max_short_per_agent`: Per-agent short position limit (default 500)
  - `locate_required`: Whether borrow locate is required before shorting
- ✅ `BorrowLedger` (`borrow_ledger.rs`): Tracks share borrowing
  - Manages available borrow pool (default 10% of shares outstanding)
  - Tracks individual agent borrows with tick timestamps
  - `borrow()`, `return_shares()`, `can_borrow()` API
- ✅ `PositionValidator` (`position_limits.rs`): Pre-trade risk checks
  - Cash sufficiency validation for buys
  - Shares outstanding limit (aggregate long positions)
  - Short limit enforcement with exemption flag for market makers
  - Borrow availability checks for short sales
- ✅ `RiskViolation` enum: `InsufficientCash`, `InsufficientShares`, `ShortSellingDisabled`, `ShortLimitExceeded`, `NoBorrowAvailable`

#### Agent Improvements
- ✅ **Market Maker Exemption**: `is_market_maker()` trait method
  - Market makers exempt from `max_short_per_agent` limit
  - Allows them to provide liquidity without position constraints
- ✅ **Market Maker Initial Position**: `initial_position` config (default 500)
  - MMs start with inventory to provide two-sided quotes
- ✅ **Noise Trader Position Constraints**:
  - Can only sell shares they own (no short selling)
  - `initial_position` config (default 50) for balanced buy/sell
  - `initial_cash` reduced to $95,000 (total value = $100,000 with shares)

#### Configuration Updates
- ✅ Scaled to 1000 Tier 1 agents (100 MM, 400 noise, 50 each quant type)
- ✅ Added `nt_initial_position` to `SimulationConfig` (default 50)
- ✅ Short limit set to 500 per agent (down from 10,000)
- ✅ `tick_delay_ms` set to 2ms for larger agent counts

#### Project Plan Restructure
- ✅ V2 renamed to "Events & Market Realism" (V2.1-V2.4)
- ✅ V3 renamed to "Scaling & Persistence" (V3.1-V3.3)
- ✅ Updated phase reference map for each version

### Technical Notes

**Why noise traders can't short sell:**
- Noise traders represent retail-like random participants
- Short selling requires sophistication (margin, borrow locate, etc.)
- Prevents unrealistic one-sided order flow at simulation start

**Why market makers are exempt from short limits:**
- MMs must provide liquidity on both sides of the book
- Position limits would prevent them from fulfilling their role
- Real MMs have special exemptions and higher limits

---

## 2026-01-03: V1.3 - Phase 7-8 Strategies, Tier Configuration & TUI Scrolling

### Completed

#### Phase 7 Strategies (4 new indicator-based agents)
- ✅ `MomentumTrader` (`momentum.rs`): RSI-based strategy
  - Buys when RSI < 30 (oversold), sells when RSI > 70 (overbought)
  - Configurable thresholds and position limits
- ✅ `TrendFollower` (`trend_follower.rs`): SMA crossover strategy
  - Golden cross (fast > slow): bullish signal → buy
  - Death cross (fast < slow): bearish signal → sell
  - Uses SMA(10) and SMA(50) by default
- ✅ `MacdCrossover` (`macd_crossover.rs`): MACD/Signal line crossover
  - Buys when MACD crosses above signal line
  - Sells when MACD crosses below signal line
  - Tracks previous state to detect crossovers
- ✅ `BollingerReversion` (`bollinger_reversion.rs`): Mean reversion at bands
  - Buys when price touches lower band (oversold)
  - Sells when price touches upper band (overbought)
  - Uses Bollinger(20, 2.0) by default

#### Phase 8 Strategy (partial)
- ✅ `VwapExecutor` (`vwap_executor.rs`): VWAP-targeting execution algorithm
  - Executes target quantity over configurable time horizon
  - Slices orders into intervals to minimize market impact
  - Uses limit orders near mid price to avoid crossing spread

#### Tier-Based Configuration System
- ✅ `Tier1AgentType` enum: `NoiseTrader`, `MarketMaker`, `Momentum`, `TrendFollower`, `Macd`, `Bollinger`
- ✅ Per-type minimums: `num_noise_traders`, `num_market_makers`, `num_momentum_traders`, etc.
- ✅ Tier minimum: `min_tier1_agents` ensures minimum total agent count
- ✅ Random fill: If specified agents < tier minimum, randomly selects from `SPAWNABLE` types
- ✅ Shared quant config: `quant_initial_cash`, `quant_order_size`, `quant_max_position`
- ✅ Agent numbering fix: Dynamic width based on agent count (`digit_width()` function)

#### TUI Scrollable Panels with Mouse Support
- ✅ **Visual Scrollbars**: `ratatui::Scrollbar` widget on both Risk and Agent P&L panels
  - Scrollbar renders on right edge (▲, █, ▼ track/thumb)
  - Scrollbar position reflects current scroll offset vs total items
- ✅ **Mouse Wheel Scrolling**: Hover over a panel and scroll with mouse wheel
  - Enabled via `crossterm::EnableMouseCapture`
  - Area tracking (`risk_area`, `agent_area`) to detect which panel mouse is over
- ✅ **Scrollbar Click/Drag**: Click on track to jump, drag thumb for continuous scrolling
- ✅ **Footer**: `q Quit │ 🖱 Scroll Mouse wheel or drag scrollbar`

### Analysis: Momentum vs NoiseTrader Metrics

**Observation**: Momentum traders show better returns but lower equity than NoiseTraders.

**Explanation**: This is **intentional by design** - the two panels measure different things:

| Panel | Metric | What It Measures |
|-------|--------|------------------|
| Risk | `total_return` | Percentage gain from initial equity |
| P&L | `equity` | Absolute wealth (cash + position × price) |

NoiseTraders accumulate large positions through frequent random trading. Even with poor percentage returns, their absolute equity can be higher because:
1. They hold more shares
2. Price appreciation on large positions compounds
3. `equity = cash + position × mark_price`

**Not a bug** - sorting by return (performance) vs equity (wealth) are both valid views.

### Un-Implemented Phase 8 Strategies

| Strategy | Blocker | When to Add |
|----------|---------|-------------|
| **Pairs Trading** | Requires two correlated symbols | V2+ (multi-symbol support) |
| **Factor Long-Short** | Requires factor infrastructure | V2+ (factor engine) |
| **News Reactive** | Requires news event system | V3+ (per project plan) |

### Files Created
| File | Purpose |
|------|---------|
| `crates/agents/src/strategies/momentum.rs` | RSI-based momentum trader |
| `crates/agents/src/strategies/trend_follower.rs` | SMA crossover trend following |
| `crates/agents/src/strategies/macd_crossover.rs` | MACD/signal line crossover |
| `crates/agents/src/strategies/bollinger_reversion.rs` | Bollinger Bands mean reversion |
| `crates/agents/src/strategies/vwap_executor.rs` | VWAP execution algorithm |

### Files Modified
| File | Changes |
|------|---------|
| `crates/agents/src/strategies/mod.rs` | Exported 5 new strategy modules |
| `src/config.rs` | Tier1AgentType enum, per-type minimums, random fill logic |
| `src/main.rs` | spawn_agent(), two-phase spawn, dynamic agent numbering |
| `crates/tui/src/app.rs` | Mouse capture, `handle_mouse_event()`, scrollbar rendering |
| `crates/tui/src/widgets/risk_panel.rs` | Added scroll_offset, dynamic visible rows |
| `crates/tui/src/widgets/agent_table.rs` | Converted to Widget with scroll_offset, scrollbar |
| `Cargo.toml` | Added `rand.workspace = true` |

### Rust Concepts Demonstrated
- **Enum dispatch**: `Tier1AgentType` with `random()` and `SPAWNABLE` const array
- **Mouse event handling**: `crossterm::event::MouseEvent` with button/position tracking
- **Area-based hit testing**: Storing `Rect` for each panel to detect mouse position
- **ratatui Scrollbar widget**: `Scrollbar::new(ScrollbarOrientation::VerticalRight)` with `ScrollbarState`
- **Saturating arithmetic**: `saturating_sub()` for safe bounds checking

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 132 tests pass
```

---

## 2026-01-03: V1.2 - Risk Metrics & Per-Agent Tracking

### Completed
- ✅ **AgentRiskTracker** (`quant/tracker.rs`): Per-agent risk monitoring
  - `AgentRiskTracker` struct tracks equity history using rolling windows
  - `RiskTrackerConfig` for configurable window size and parameters
  - `AgentRiskSnapshot` captures point-in-time risk metrics per agent
  - Computes: Sharpe ratio, Sortino ratio, max drawdown, VaR (95%), total return, volatility
  - Efficient O(1) equity recording with bounded memory usage

- ✅ **Agent equity computation**: Added `equity()` method to Agent trait
  - Computes mark-to-market equity: `cash + (position × price)`
  - Handles both long and short positions correctly
  - Default implementation uses existing `cash()` and `position()` methods

- ✅ **Simulation risk integration**:
  - `Simulation` struct now holds `AgentRiskTracker`
  - Records equity for all agents after each tick
  - New methods: `agent_risk_metrics()`, `agent_risk()`
  - Uses last trade price for mark-to-market valuation

- ✅ **TUI RiskPanel widget** (`tui/widgets/risk_panel.rs`):
  - Color-coded display of per-agent risk metrics
  - Shows: Return, Max DD, Sharpe for up to 10 agents
  - Aggregate metrics: Average Sharpe, worst max drawdown (excludes market makers)
  - Color coding: green (good), yellow (caution), red (risk)

- ✅ **TUI layout improvements**:
  - Rebalanced left column: Stats (top), Order book (fixed 14 lines), Risk panel (expanded)
  - Risk panel shows up to 10 agents sorted by total return
  - Market makers sorted to bottom as "infrastructure" agents

- ✅ **Agent identification & sorting**:
  - Agent names prefixed with ID: `"{:02}-{name}"` (e.g., "04-NoiseTrader")
  - `is_market_maker` flag distinguishes infrastructure vs trading agents
  - **P&L table**: Sorted by equity (descending), market makers at bottom
  - **Risk panel**: Sorted by total return (descending), market makers at bottom
  - Aggregate metrics computed excluding market makers (focus on trader performance)

- ✅ **SimUpdate extended**: Added `risk_metrics: Vec<RiskInfo>`, `equity`, `is_market_maker`
  - `RiskInfo` struct: name, sharpe, max_drawdown, total_return, var_95, equity, is_market_maker
  - `AgentInfo` struct: Added `equity: f64`, `is_market_maker: bool` fields
  - Main binary populates from simulation's `agent_risk_metrics()`

### Rust Concepts Demonstrated
- **Composition**: `AgentRiskTracker` composes `RollingWindow` for memory-bounded history
- **Trait default methods**: `equity()` provides default implementation using other trait methods
- **HashMap with newtype keys**: `HashMap<AgentId, RollingWindow>` for per-agent tracking
- **Builder pattern**: `RiskPanel::new().agents(vec![...]).aggregate_sharpe(Some(1.2))`
- **Tiered sorting**: Multi-key sorting with `sort_by(|a, b| a.is_mm.cmp(&b.is_mm).then(equity_cmp))`

### SOLID Compliance
- **S**: `AgentRiskTracker` only tracks equity and computes metrics (single responsibility)
- **O**: New risk widget added without modifying existing widgets
- **L**: `RiskInfo` and `AgentRiskSnapshot` are interchangeable data carriers
- **I**: `AgentRiskTracker` exposes minimal interface (record, compute)
- **D**: Simulation depends on `AgentRiskTracker` abstraction, not concrete risk calculations

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 113 tests pass
                       # 16 agents + 38 quant + 4 binary + 24 sim-core
                       # + 6 simulation + 4 integration + 11 tui + 10 types
```

### Files Created
| File | Purpose |
|------|---------|
| `crates/quant/src/tracker.rs` | Per-agent risk tracking and metrics |
| `crates/tui/src/widgets/risk_panel.rs` | Risk metrics TUI widget |

### Files Modified
| File | Changes |
|------|---------|
| `crates/quant/src/lib.rs` | Added tracker module and exports |
| `crates/agents/src/traits.rs` | Added `equity()` method to Agent trait |
| `crates/simulation/src/runner.rs` | Added AgentRiskTracker, equity tracking in step() |
| `crates/tui/src/widgets/mod.rs` | Added risk_panel module export |
| `crates/tui/src/widgets/update.rs` | Added RiskInfo, AgentInfo with is_market_maker & equity |
| `crates/tui/src/widgets/agent_table.rs` | Tiered sorting: MMs at bottom, sort by equity |
| `crates/tui/src/app.rs` | Rebalanced layout, tiered sorting for risk panel |
| `crates/tui/src/lib.rs` | Re-export RiskInfo |
| `src/main.rs` | ID-prefixed names, is_market_maker flag, equity calculation |

### Design Notes
- **Rolling window vs full history**: Using `RollingWindow` (default 500 observations) prevents unbounded memory growth while keeping enough data for meaningful statistics
- **Mark-to-market**: Risk metrics use last trade price for position valuation; if no trades, uses initial price
- **Market maker treatment**: MMs are infrastructure agents that start with high capital; sorting them to bottom keeps focus on actual trading agents
- **Agent naming**: ID prefix (`04-NoiseTrader`) makes agents distinguishable across panels
- **Aggregate metrics**: Exclude market makers—computed only for trading agents to reflect true performance
- **Minimum observations**: Risk metrics require 20+ data points before computing Sharpe/Sortino/VaR to avoid meaningless early values
- **Sorting rationale**: Risk panel sorts by total return (performance measure); P&L table sorts by equity (wealth measure); both put MMs at bottom

---

## 2026-01-03: V1.1 - Quant Layer (Indicators)

### Completed
- ✅ **quant crate**: Technical analysis and quantitative tools
  - New crate at `crates/quant/` with modular architecture
  - Pure math calculations for indicators, risk, and statistics
  - Thread-safe design (`Send + Sync`) for future parallelization

- ✅ **Indicators module** (`indicators.rs`): 6 technical indicators
  - `Sma` — Simple Moving Average with configurable period
  - `Ema` — Exponential Moving Average with smoothing factor
  - `Rsi` — Relative Strength Index (bounded 0-100)
  - `Macd` — Moving Average Convergence Divergence (signal line + histogram)
  - `BollingerBands` — Upper/middle/lower bands with configurable std devs
  - `Atr` — Average True Range for volatility measurement
  - Factory function `create_indicator()` for runtime construction

- ✅ **Engine module** (`engine.rs`): Indicator management
  - `IndicatorEngine` — registers and computes indicators
  - `IndicatorCache` — per-tick caching to avoid redundant computation
  - `IndicatorSnapshot` — frozen indicator values for MarketData
  - `with_common_indicators()` preset for standard setup

- ✅ **Rolling window** (`rolling.rs`): Efficient data structure
  - O(1) push with automatic oldest-value eviction
  - Running sum/mean for incremental computation
  - Variance, std_dev, min, max accessors

- ✅ **Statistics** (`stats.rs`): Statistical utilities
  - `mean`, `variance`, `std_dev`, `sample_variance`, `sample_std_dev`
  - `returns`, `log_returns` for price series
  - `covariance`, `correlation` for pair analysis
  - `percentile` for distribution analysis

- ✅ **Risk metrics** (`risk.rs`): Foundation for V1.2
  - `max_drawdown` — peak-to-trough analysis
  - `sharpe_ratio`, `sortino_ratio` — risk-adjusted returns
  - `historical_var` — Value at Risk at configurable confidence
  - `annualized_volatility` — for portfolio analysis
  - `RiskMetrics` struct aggregates common metrics

- ✅ **Type additions** (`types` crate): New market data types
  - `Candle` — OHLCV candle with timestamp
  - `IndicatorType` — enum for all indicator variants
  - `MacdOutput`, `BollingerOutput` — multi-value indicator results

- ✅ **Simulation integration**: Indicators available to agents
  - `CandleBuilder` builds candles from trade stream
  - `MarketData` now includes `candles` and `indicators` fields
  - `get_indicator()` helper for easy access in strategies

### Rust Concepts Demonstrated
- **Trait objects** — `Box<dyn Indicator>` for polymorphic indicator storage
- **Associated types** — `Indicator::Output` for flexible return types
- **Factory pattern** — `create_indicator()` constructs from enum variant
- **Builder pattern** — `CandleBuilder` accumulates trades into candles
- **Caching** — `IndicatorCache` prevents redundant expensive calculations
- **Module organization** — `pub mod` + re-exports at crate root

### SOLID Compliance
- **S**: Each indicator has single responsibility (compute one metric)
- **O**: `Indicator` trait open for extension, closed for modification
- **L**: All indicators safely return `None` for insufficient data
- **I**: `Indicator` trait is minimal (3 methods)
- **D**: Engine depends on `Indicator` abstraction, not concrete types

### Module Structure Decision
Kept `rolling`, `risk`, `stats` in `quant` crate because:
1. Dependency graph shows `quant` is shared by gym (rewards), data service (risk APIs), and agents
2. All modules are coherent — part of quantitative analysis domain
3. RL gym rewards (Phase 15) will call `quant::sharpe_ratio()`, `quant::max_drawdown()`
4. Data service `/risk/*` endpoints will use same functions

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 102 tests pass
                       # 16 agents + 30 quant + 4 binary + 24 sim-core
                       # + 6 simulation + 4 integration + 8 tui + 10 types
```

### Files Created
| File | Purpose |
|------|---------|
| `crates/quant/Cargo.toml` | Quant crate manifest |
| `crates/quant/src/lib.rs` | Crate root with re-exports |
| `crates/quant/src/indicators.rs` | 6 technical indicator implementations |
| `crates/quant/src/engine.rs` | IndicatorEngine and caching |
| `crates/quant/src/rolling.rs` | Rolling window data structure |
| `crates/quant/src/stats.rs` | Statistical utility functions |
| `crates/quant/src/risk.rs` | Risk metrics (Sharpe, VaR, drawdown) |

### Files Modified
| File | Changes |
|------|---------|
| `Cargo.toml` | Added quant crate to workspace |
| `crates/types/src/lib.rs` | Added Candle, IndicatorType, MacdOutput, BollingerOutput |
| `crates/agents/Cargo.toml` | Added quant dependency |
| `crates/agents/src/traits.rs` | Added candles/indicators to MarketData |
| `crates/simulation/Cargo.toml` | Added quant dependency |
| `crates/simulation/src/runner.rs` | Added candle building, indicator computation |
| `src/config.rs` | Fixed hard-coded tests to be resilient to default changes |

### Design Notes
- **Fixed-Point Precision**: Indicators compute with `f64` internally, converting from/to `Price`/`Cash` newtypes as needed. This maintains monetary precision while allowing statistical operations.
- **Lazy Computation**: `IndicatorCache` ensures expensive calculations (like MACD which needs 26+ candles) only run once per tick, regardless of how many agents query them.
- **Candle Building**: `CandleBuilder` aggregates trades into OHLCV candles. Default interval is 100 ticks; configurable for different timeframes.

---

## 2026-01-03: V0.4 - TUI Visualization

### Completed
- ✅ **tui crate**: Real-time terminal visualization
  - `ratatui` v0.29 for terminal UI rendering
  - `crossterm` v0.28 for cross-platform terminal control
  - `crossbeam-channel` v0.5 for thread-safe communication

- ✅ **PriceChart widget**: Line graph of price history
  - Braille markers for smooth rendering
  - Auto-scaling Y axis with min/max labels
  - Handles empty state gracefully

- ✅ **BookDepth widget**: Order book visualization
  - Side-by-side bid/ask columns
  - Horizontal quantity bars with relative scaling
  - Color-coded (green bids, red asks)

- ✅ **AgentTable widget**: Agent P&L summary
  - Position, cash, realized P&L columns
  - Color-coded positions (green long, red short)
  - Auto-sizing columns

- ✅ **StatsPanel widget**: Simulation statistics
  - Tick counter, last price, spread
  - Total trades and orders
  - Agent count

- ✅ **Multi-threaded architecture**: Channel-based design
  - Simulation runs in dedicated thread
  - TUI renders at 30fps without blocking simulation
  - Bounded channel (100 capacity) for backpressure

- ✅ **Agent trait extensions**: Added optional methods
  - `position()` — returns agent's share position (default: 0)
  - `cash()` — returns agent's cash balance (default: ZERO)
  - `realized_pnl()` — returns realized P&L (default: ZERO, stub for V1)
  - Defaults are intentional for agents that don't track state (LSP-compliant)

- ✅ **AgentState struct** (DRY refactoring): Common state tracking
  - Extracted from duplicated `NoiseTraderState` and `MarketMakerState`
  - Encapsulates position, cash, orders_placed, fills_received
  - Methods: `on_buy()`, `on_sell()`, `record_order()`, `record_orders()`
  - Private fields with public getters for encapsulation
  - NoiseTrader and MarketMaker now use shared AgentState

### Rust Concepts Demonstrated
- **External crate integration** — ratatui, crossterm for TUI
- **Channels (Actor Model)** — `crossbeam_channel::bounded` for thread comm
- **Trait default methods** — `position()`, `cash()` with safe fallback defaults
- **Generic iterators** — `TuiApp<R: Iterator<Item = SimUpdate>>`
- **DRY principle** — `AgentState` eliminates ~30 lines of duplication
- **Encapsulation** — Private fields with getter methods
- **Test-only methods** — `#[cfg(test)]` for `set_position()`

### SOLID Compliance
- **S**: Each struct has single responsibility (AgentState = state, Agent = behavior)
- **O**: Agent trait open for new implementations, closed for modification
- **L**: Default trait methods return safe fallbacks (0/ZERO) for any Agent impl
- **I**: Optional methods via defaults — agents only override what they need
- **D**: Simulation depends on Agent trait abstraction, not concrete types

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 68 tests pass (10 types + 24 sim-core + 16 agents + 6 simulation + 4 integration + 8 tui)
cargo run             # ✅ Watch 22 agents trade in real-time TUI
```

### Files Created
| File | Purpose |
|------|---------|
| `crates/tui/Cargo.toml` | TUI crate manifest |
| `crates/tui/src/lib.rs` | Crate root exports |
| `crates/tui/src/app.rs` | Main TUI application loop |
| `crates/tui/src/widgets/mod.rs` | Widget module exports |
| `crates/tui/src/widgets/update.rs` | SimUpdate message type |
| `crates/tui/src/widgets/price_chart.rs` | Price chart widget |
| `crates/tui/src/widgets/book_depth.rs` | Order book depth widget |
| `crates/tui/src/widgets/agent_table.rs` | Agent P&L table widget |
| `crates/tui/src/widgets/stats_panel.rs` | Statistics panel widget |
| `crates/agents/src/state.rs` | AgentState shared struct |

### Files Modified
| File | Changes |
|------|---------|
| `Cargo.toml` | Added tui crate, ratatui, crossterm, crossbeam-channel deps |
| `crates/agents/src/lib.rs` | Added `mod state` and export AgentState |
| `crates/agents/src/traits.rs` | Added `position()`, `cash()`, `realized_pnl()` to Agent trait |
| `crates/agents/src/strategies/noise_trader.rs` | Uses AgentState, implements trait methods |
| `crates/agents/src/strategies/market_maker.rs` | Uses AgentState + MM-specific fields, implements trait methods |
| `crates/simulation/src/runner.rs` | Added `agent_summaries()` method |
| `src/main.rs` | Complete rewrite with TUI + channel integration |

### Design Notes
- **Channel Design**: Bounded channel (100 slots) prevents memory growth if TUI lags. Simulation thread sends updates every tick; TUI drains all updates each frame (keeps latest).
- **Separation of Concerns**: `SimUpdate` is a pure data struct — TUI doesn't know about `Simulation`, just renders what it receives.
- **Agent Count**: Increased to 22 agents (2 MM + 20 NoiseTraders) for more visual activity while keeping V0 practical.
- **Frame Rate**: TUI renders at 30fps; simulation runs at ~100 ticks/sec with 10ms delay between ticks.

### V0 MVP Simulation Complete! 🎉
The simulation now produces a watchable TUI showing:
- Price movements in real-time chart
- Order book depth with bid/ask bars
- All 22 agents with positions and cash
- Running statistics (trades, orders, tick count)

---

## 2026-01-03: V0.3 - Agent Strategies (NoiseTrader & MarketMaker)

### Completed
- ✅ **NoiseTrader agent**: Random market activity generator
  - Configurable via `NoiseTraderConfig` (probability, price deviation, quantity range)
  - Uses mid price → last price → initial price fallback chain (zombie prevention)
  - Tracks position and cash internally via `on_fill()`
  - Uses `StdRng` (Send-compatible) for thread safety
  - 4 unit tests

- ✅ **MarketMaker agent**: Two-sided liquidity provider
  - Configurable via `MarketMakerConfig` (spread, quote size, refresh interval)
  - Inventory skew: adjusts quotes based on position to reduce risk
  - Seeds initial liquidity at startup (prevents zombie simulation)
  - Periodic quote refresh (configurable interval)
  - 5 unit tests

- ✅ **strategies module**: Organized strategy implementations
  - `agents/src/strategies/mod.rs` exports concrete agents
  - Re-exported at crate root for convenient access

- ✅ **Integration tests**: Agent strategy validation
  - `test_10_noise_traders_2_market_makers_produce_trades` — main exit criterion
  - `test_market_makers_alone_no_trades` — MMs don't self-trade
  - `test_noise_traders_can_trade_among_themselves` — crossing orders match
  - `test_simulation_with_fills_runs_without_panic` — smoke test for fill path

### Rust Concepts Demonstrated
- **rand crate** — `StdRng::from_os_rng()` for reproducible, Send-safe randomness
- **Mutable borrows** — agents read market, mutate internal state
- **Interior state** — position/cash tracking without interior mutability

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 56 tests pass (10 types + 24 sim-core + 12 agents + 6 simulation + 4 integration)
```

### Files Created
| File | Purpose |
|------|---------|
| `crates/agents/src/strategies/mod.rs` | Strategy module exports |
| `crates/agents/src/strategies/noise_trader.rs` | NoiseTrader implementation |
| `crates/agents/src/strategies/market_maker.rs` | MarketMaker implementation |
| `crates/simulation/tests/agent_strategies.rs` | Agent strategy integration tests |

### Design Notes
- **Zombie Prevention**: MarketMaker seeds tight quotes around initial price ($100); NoiseTraders use mid price as reference, ensuring orders land near MM quotes and produce matches.
- **Inventory Skew**: MarketMaker adjusts bid/ask prices based on inventory position (long → lower prices to sell, short → higher prices to buy).
- **Send Requirement**: `Agent` trait requires `Send`, so NoiseTrader uses `StdRng` instead of `ThreadRng`.

---

## 2026-01-03: V0.2 - Simulation Loop (Agent Trait & Runner)

### Completed
- ✅ **agents crate**: Agent trait and market data
  - `Agent` trait with `on_tick()` and `on_fill()` methods
  - `AgentAction` struct supporting multiple orders per tick
  - `MarketData` snapshot with book, recent trades, last price
  - Clean trait interface following "Declarative, Modular, SoC" mantra
  - 3 unit tests

- ✅ **simulation crate**: Tick-based event loop
  - `Simulation` struct holding `OrderBook` and `Vec<Box<dyn Agent>>`
  - `SimulationConfig` for declarative configuration
  - `step()` method advancing simulation by one tick
  - Order processing through `MatchingEngine`
  - Fill notifications to agents via `on_fill()`
  - `SimulationStats` for tracking metrics
  - 6 unit tests

- ✅ **main.rs binary**: Runs 1000-tick simulation with passive agents

- ✅ **Documentation**: Added "Declarative, Modular, SoC" mantra to:
  - `project_plan_vertical.md`
  - `project_plan.md`
  - `README.md`

### Rust Concepts Demonstrated
- **Traits** — `Agent` trait for polymorphic behavior
- **Trait objects** — `Box<dyn Agent>` for heterogeneous agent storage
- **Send bound** — Agents are `Send` for future multi-threading

### Exit Criteria
```
cargo fmt --check     # ✅ No formatting issues
cargo clippy          # ✅ No warnings
cargo test --workspace # ✅ 43 tests pass (10 types + 24 sim-core + 3 agents + 6 simulation)
cargo run             # ✅ 1000 ticks without panic
```

### Files Created
| File | Purpose |
|------|---------|
| `crates/agents/Cargo.toml` | Agents crate manifest |
| `crates/agents/src/lib.rs` | Exports Agent, AgentAction, MarketData |
| `crates/agents/src/traits.rs` | Agent trait definition |
| `crates/simulation/Cargo.toml` | Simulation crate manifest |
| `crates/simulation/src/lib.rs` | Exports Simulation, SimulationConfig |
| `crates/simulation/src/config.rs` | SimulationConfig struct |
| `crates/simulation/src/runner.rs` | Simulation runner with step() |
| `src/main.rs` | Binary demonstrating simulation |

---

## 2026-01-03: V0.1 - Core Engine (Types & Order Book)

### Completed
- ✅ **types crate**: Core data types with fixed-point arithmetic
  - `Price` and `Cash` newtypes with 4 decimal places (PRICE_SCALE = 10,000)
  - `Quantity` newtype (u64) with `From<u64>` and `PartialEq<u64>` for ergonomic usage
  - Uses `derive_more` crate for clean trait derivation (Add, Sub, Neg, AddAssign, SubAssign, Sum, From, Into)
  - `Order`, `Trade`, `OrderSide`, `OrderType`, `OrderStatus`
  - `BookSnapshot`, `BookLevel` for order book visualization
  - Comprehensive test coverage (10 tests)

- ✅ **sim-core crate**: Market mechanics
  - `OrderBook` using `BTreeMap<Price, VecDeque<Order>>` for price-time priority
  - `MatchingEngine` with price-time priority matching
  - Support for limit and market orders
  - Partial fills, order cancellation
  - 24 unit tests covering matching edge cases

### Exit Criteria
```
cargo fmt --check  # No formatting issues
cargo clippy -- -D warnings  # No clippy warnings
cargo test  # 34 tests pass
```
