# Development Log

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
