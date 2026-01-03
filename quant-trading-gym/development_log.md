# Development Log

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
