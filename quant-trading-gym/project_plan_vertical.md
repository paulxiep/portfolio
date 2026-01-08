# Quant Trading Gym: Vertical plan

Ideated with LLM assistance, summarised by Opus 4.5 Thinking.

Refer to [full plan here](project_plan.md)

---

## Philosophy

**Build vertically, not horizontally.**

Instead of completing all types → all matching → all agents → all viz, we build a thin slice through the entire stack each iteration. Every iteration produces something *runnable*.

---

## Guiding Mantra

> **"Declarative, Modular, SoC"**

Every implementation decision should be evaluated against these three principles:

| Principle | Meaning | Example |
|-----------|---------|----------|
| **Declarative** | Describe *what*, not *how*. Config over code. Data-driven behavior. | Strategies declare indicators they need; the system provides them. Agent behavior defined by config/trait impl, not hardcoded logic. |
| **Modular** | Components are self-contained, swappable, and independently testable. | Each crate compiles alone. Strategies are plugins. Swap `NoiseTrader` for `RLAgent` without touching simulation. |
| **SoC** (Separation of Concerns) | Each module has ONE job. No god objects. Clear boundaries. | `types/` = data. `sim-core/` = matching. `agents/` = behavior. `simulation/` = orchestration. No crate does two things. |

**Before writing code, ask:**
1. Am I describing behavior or implementing mechanics? (Declarative)
2. Can this be swapped out without ripple effects? (Modular)
3. Does this component have exactly one responsibility? (SoC)

---

## V0: The Steel Thread (4 Weeks)

**Goal:** A single-threaded simulation with TUI visualization showing agents trading.

### Week 1: The Engine (The "Truth")

**Focus:** Types, `BTreeMap`, Matching Logic

**Deliverables:**
- `OrderBook` struct using `BTreeMap<Price, VecDeque<Order>>`
- `LimitOrder` and `Trade` types
- Price-time priority matching
- Basic unit tests

**Rust Concepts:**
- Ownership (who owns the order after submission?)
- Structs, Enums (`Buy`/`Sell`)
- `Option<Trade>` for match results
- Newtype pattern: `struct Price(i64);` with fixed-point scale (e.g., 10,000 = 4 decimal places)

**Files:**
```
crates/
└── types/
    └── lib.rs          # Order, Trade, Side, Price types
└── sim-core/
    ├── lib.rs
    ├── order_book.rs   # BTreeMap-based book
    └── matching.rs     # Price-time priority
```

**Exit Criteria:** `cargo test` passes for order insertion and matching.

---

### Week 2: The Loop (The "Time")

**Focus:** Event-driven simulation architecture

**Deliverables:**
- `Simulation` struct that holds `OrderBook` and `Vec<Box<dyn Agent>>`
- Tick-based loop: `simulation.step()` advances one tick
- `MarketData` snapshot passed to agents each tick
- `Agent` trait definition

**Rust Concepts:**
- **Traits** — the polymorphic heart (`trait Agent { fn on_tick(&mut self, market: &MarketData) -> Option<Order>; }`)
- Trait objects (`Box<dyn Agent>`)
- Lifetimes (MarketData borrowing from OrderBook)
- The borrow checker fight begins

**Technical Note:** Orders need `agent_id: AgentId` (newtype around `u64`) so the simulation can update the correct agent's P&L when a trade occurs. Add `AgentId` to `types/lib.rs`.

**Files:**
```
crates/
└── agents/
    ├── lib.rs
    └── traits.rs       # Agent trait
└── simulation/
    ├── lib.rs
    ├── runner.rs       # Simulation struct, step()
    └── context.rs      # MarketData/StrategyContext for agents (matches full plan naming)
```

**Exit Criteria:** Empty simulation runs 1000 ticks without panic.

---

### Week 3: The Dumb Agents

**Focus:** Implement concrete agents, state management

**Deliverables:**
- `NoiseTrader`: Random buy/sell at random prices near mid
- `MarketMaker`: Places bid/ask spread, manages inventory
- Agents maintain internal state (position, cash)

**Rust Concepts:**
- `rand` crate usage
- Mutable borrows (agents reading market, mutating themselves)
- Interior mutability patterns (if needed)
- Implementing traits for structs

**Files:**
```
crates/
└── agents/
    ├── strategies/
    │   ├── mod.rs
    │   ├── noise_trader.rs   # Tier 1 agent: runs every tick, random orders
    │   └── market_maker.rs   # Tier 1 agent: runs every tick, provides liquidity
```

**Migration Note:** In V2, when adding tiers, move these files into the tier1 folder:
- `strategies/noise_trader.rs` → `tier1/strategies/noise_trader.rs`
- `strategies/market_maker.rs` → `tier1/strategies/market_maker.rs`

**Why Tier 1?** These agents run every tick with full decision logic. Tier 2 (Reactive) agents only wake on conditions like price crosses or intervals.

**Exit Criteria:** 10 NoiseTraders + 2 MarketMakers produce trades over 1000 ticks.

**⚠️ Zombie Risk:** If NoiseTraders place orders at random prices without a reference point, nothing matches — the simulation runs but produces zero trades. **Fix:** MarketMaker must seed a tight spread around an initial price (e.g., bid $99, ask $101 around $100 start). NoiseTraders then generate orders near the current mid price, not truly random.

---

### Week 4: The Viz (The "Reward")

**Focus:** Terminal UI with live updating

**Deliverables:**
- `ratatui` integration
- Live price chart (candlesticks or line)
- Order book depth visualization
- Agent P&L summary table

**Rust Concepts:**
- External crate integration
- Event loops (simulation tick vs render tick)
- **Channels (Actor Model):** Use `std::sync::mpsc` or `crossbeam` channels
  - Thread A (Sim): Runs fast, pushes `SimUpdate` events (trades, price changes)
  - Thread B (TUI): Reads channel at ~60fps, renders UI
  - Prevents slow terminal rendering from blocking matching engine

**Files:**
```
crates/
└── tui/                # NOTE: Not in full plan. Becomes SimulationHook in V3,
    ├── lib.rs          # then replaced by services/frontend in V4-Game
    ├── app.rs          # Main TUI app
    ├── price_chart.rs
    └── book_depth.rs
src/
└── main.rs             # Binary that runs sim + TUI
```

**Migration Note:** The TUI crate is V0-specific for fast feedback. In V3, refactor it to implement `SimulationHook` trait. In V4-Game, the web frontend supersedes it (TUI becomes optional dev tool).

**Exit Criteria:** Watch agents trade in real-time in terminal. Screenshot-worthy.

---

## V0 Summary

| Week | Deliverable | Key Rust Learning |
|------|-------------|-------------------|
| 1 | OrderBook + Matching | Ownership, Enums, BTreeMap |
| 2 | Simulation Loop | Traits, Trait Objects, Lifetimes |
| 3 | NoiseTrader, MarketMaker | Rand, Mutable Borrows |
| 4 | TUI Visualization | Crate Integration, Event Loops |

**Total:** ~4 weeks
**Lines of Code:** ~1,500-2,500
**Dependencies:** `rand`, `ratatui`, `crossterm` (no decimal crate — uses `i64` fixed-point)

---

# Iterative Expansion: V0 → Full Plan

After V0, expand **iteratively** — each iteration adds a vertical slice of functionality.

```
V0 (Steel Thread)
 │
 ├──► V1: Quant Layer (indicators, risk, strategies)
 │
 ├──► V2: Events & Market Realism (fundamentals, multi-symbol, position limits)
 │
 ├──► V3: Scaling & Persistence (tiers, storage, hooks)
 │
 ├──► V4: RL Track OR Game Track
 │
 └──► V5: Full Integration
```

---

## V1: Quant Layer (+2 weeks)

**Add:** Indicators, risk metrics, better strategies

### V1.1: Indicators (~4 days)
- SMA, EMA, RSI, MACD calculations
- Rolling window data structures
- Wire indicators into `MarketData`

### V1.2: Risk Metrics (~3 days)
- Sharpe ratio, max drawdown, VaR
- Per-agent risk tracking
- Add to TUI dashboard

### V1.3: One Real Strategy (~3 days)
- Implement `MomentumTrader` using RSI
- Or `TrendFollower` using SMA crossover
- Prove the indicator pipeline works

**Maps to Original:** Phase 3 (Quant Foundation) + Phase 7 (partial: 1 strategy)

---

## V2: Events & Market Realism (+3 weeks)

**Add:** Realistic market constraints, fundamental value system, multi-symbol trading

### V2.1: Position Limits & Short-Selling (~2 days)
- `SymbolConfig` with `shares_outstanding` (natural long limit)
- `ShortSellingConfig` with borrow pool derived from float
- `BorrowLedger` for tracking borrowed shares
- Order validation: cash + shares_outstanding for longs, borrow availability for shorts
- MarketMakers start with `initial_position` (inventory from float)
- **Addresses V1 issue:** agents accumulating unrealistic positions

### V2.2: Slippage & Partial Fills (~2 days)
- Market orders experience slippage based on book depth
- Large orders partially fill across price levels
- Add `Fill` events distinct from full `Trade`
- Impact model: `slippage = f(order_size / available_liquidity)`

### V2.3: Multi-Symbol Infrastructure (~3 days)
- `Market` with `HashMap<Symbol, OrderBook>`
- `SimulationConfig` with `symbols: Vec<SymbolConfig>`
- Per-symbol position tracking in `AgentState`
- Portfolio-level risk metrics (correlation, beta exposure)
- Enable pairs trading strategies

### V2.4: Fundamentals & Events (~5 days)
- `Fundamentals` struct: EPS, growth estimate, payout ratio
- `MacroEnvironment`: risk-free rate, equity risk premium
- `fair_value()` derived via Gordon Growth Model
- `FundamentalEvent` enum: earnings surprise, guidance change, rate decision
- **MarketMaker anchors quotes to `fair_value()`** instead of last price
- **NoiseTrader trades around `fair_value()`** with configurable deviation
- Event generator with configurable frequency and magnitude
- **Result:** Smart strategies now have alpha — prices mean-revert to fundamentals

**Why Events in V2?**
Without a fundamental anchor, momentum/mean-reversion strategies trade noise.
V2.4 gives price a "reason" to move, making strategy performance meaningful.
Tier 1 agents poll events each tick (fine for <1000 agents).
V3 adds efficient event subscriptions for scale.

**Maps to Original:** Part 16 (Position Limits/Short-Selling) + Phase 4 (News/Events) + Phase 9 (Multi-Symbol in Simulation)

---

## V3: Scaling & Persistence (+4 weeks)

**Add:** Multi-symbol agent state, tiered agent architecture for 100k+ scale, storage layer

### V3.1: Multi-Symbol AgentState (~3 days)
- Refactor `AgentState` from single `position: i64` to `positions: HashMap<Symbol, PositionEntry>`
- Add `PositionEntry { quantity: i64, avg_cost: f64 }` for per-symbol tracking with weighted average cost
- Clean API: `on_buy(symbol, qty, value)`, `on_sell(symbol, qty, value)`, `position_for(symbol)`
- Extend `Agent` trait: `positions()`, `watched_symbols()`, `equity(&prices_map)`, `equity_for(symbol, price)`
- `position()` returns aggregate sum across all symbols (convenience, not backward-compat)
- Update simulation runner to build `HashMap<Symbol, Price>` for mark-to-market valuation
- Update all Tier 1 strategies to use symbol-aware state methods
- **Foundation for:** Tier 2 wake conditions, PairsTrading, SectorRotator

### V3.2: Tier 2 Reactive Agents (~4 days)
- `ReactiveAgent` struct (lightweight, event-driven)
- Wake conditions: price threshold, interval, event subscription
- `WakeConditionIndex` for O(log n) lookups using `watched_symbols()`
- **Event subscription:** Tier 2 agents wake only on relevant `FundamentalEvent`
- **Parametric condition updates** — modify wake conditions at runtime via `ConditionUpdate` buffer
- `LightweightContext` with triggered symbol, price, and wake reason
- `ReactivePortfolio` enum: `SingleSymbol(i64, Price)` (~150 bytes) or `Full(HashMap)` (~1KB)
- Enum dispatch for strategies: `ThresholdBuyer`, `ThresholdSeller`, `NewsReactor`, `MomentumFollower`

### V3.3: Multi-Symbol Strategies

**Goal:** Two flagship multi-symbol strategies demonstrating the V3.1/V3.2 infrastructure.

#### Quant Extensions

Added statistical tools to `quant/stats.rs`:

- **CointegrationTracker**: Rolling cointegration test for pairs trading
  - `update(price_a, price_b) -> Option<CointegrationResult>` with spread, z_score, hedge_ratio
  - OLS-based hedge ratio computation
  - Rolling window for mean/std calculation

- **SectorSentimentAggregator**: Sentiment aggregation from news events
  - `aggregate_all(events, current_tick) -> HashMap<Sector, SectorSentiment>`
  - Decay-weighted sentiment with magnitude scaling
  - Event expiration filtering

#### PairsTrading Strategy (Tier 1)

**What it does:** Exploits mean-reversion between two cointegrated symbols.

```rust
pub struct PairsTradingConfig {
    pub symbol_a: Symbol,
    pub symbol_b: Symbol,
    pub lookback_window: usize,      // Default: 100 ticks
    pub entry_z_threshold: f64,      // Default: 2.0
    pub exit_z_threshold: f64,       // Default: 0.5
    pub max_position_per_leg: i64,   // Default: 100 shares
}
```

**Decision logic:**
1. Update `CointegrationTracker` with latest prices
2. If no position and `|z_score| > entry_threshold` → enter spread
3. If in position and `|z_score| < exit_threshold` → exit both legs

**Returns:** `AgentAction::multiple(orders)` for simultaneous leg execution.

#### SectorRotator Strategy (Tier 2)

**What it does:** Shifts portfolio allocation toward sectors with positive sentiment.

```rust
pub struct SectorRotatorConfig {
    pub symbols_per_sector: HashMap<Sector, Vec<Symbol>>,
    pub sentiment_scale: f64,         // How much sentiment shifts allocation
    pub min_allocation: f64,          // Floor (e.g., 0.05)
    pub max_allocation: f64,          // Ceiling (e.g., 0.40)
    pub rebalance_threshold: f64,     // Only trade if drift > threshold
}
```

**Wake condition:** `WakeCondition::NewsEvent { symbols }` for all watched symbols.

**Decision logic:**
1. On wake, aggregate sentiment per sector
2. Compute target allocations: `base + (sentiment * scale)`, clamped and normalized
3. Generate rebalance orders if drift exceeds threshold

#### Simulation Integration

- Config fields: `num_pairs_traders` (50), `num_sector_rotators` (300)
- PairsTrading: Tier 1 (runs every tick via `specified_tier1_agents()`)
- SectorRotator: Special Tier 2 (counted in `tier2_count`, wakes on news)
- TUI: Shows "PairsTrading" and "SectorRotator" names, Total P&L column

#### Files Modified

| File | Changes |
|------|---------|
| `crates/quant/src/stats.rs` | `CointegrationTracker`, `SectorSentimentAggregator`, `NewsEventLike` trait |
| `crates/agents/src/tier1/strategies/pairs_trading.rs` | New: Tier 1 multi-symbol pairs strategy |
| `crates/agents/src/tier2/sector_rotator.rs` | New: Tier 2 sentiment-driven rotation |
| `crates/agents/src/state.rs` | Added `total_pnl()` method |
| `crates/simulation/src/runner.rs` | `agent_summaries()` returns total P&L |
| `crates/tui/src/widgets/update.rs` | `AgentInfo.total_pnl` field |
| `crates/tui/src/widgets/agent_table.rs` | "Total P&L" column header |
| `src/config.rs` | `num_pairs_traders`, `num_sector_rotators`, updated `total_agents()` |
| `src/main.rs` | `spawn_sector_rotators()`, pairs trading spawn, tier counts |

#### V3.3 Borrow-Checker Pitfalls

| Pitfall | Scenario | Solution |
|---------|----------|----------|
| **Multi-symbol price reads** | Reading prices for 2+ symbols from `&Market` | Safe: `Market::mid_price()` returns owned `Price` |
| **Sentiment aggregation during tick** | Aggregating from `ActiveNewsState` while tick runs | Safe: Read-only access via `&StrategyContext` |
| **Position updates for multiple symbols** | PairsTrading adjusting both legs | Return both orders; simulation applies sequentially |
| **Target allocation mutation** | SectorRotator updating `target_allocations` | `&mut self` — agent owns its targets |
| **WakeCondition registration** | SectorRotator registering many conditions | Implement `initial_wake_conditions()` trait method |

**Maps to Original:** Part of Phase 7 (Advanced Strategies)

### V3.4: Tier 3 Background Pool (~4 days)
- Statistical order generation (no individual agents)
- `BackgroundAgentPool` struct
- Configurable distributions (size, price, direction)
- **Sentiment-driven:** Pool bias shifts with active `FundamentalEvent`s
- Per-sector sentiment tracking

### V3.5: Performance Tuning (~3 days)
- Benchmark 100k agents
- Profile and optimize hot paths
- Memory budget validation
- Two-phase tick architecture (read phase parallel, write phase sequential)
- **Rayon integration:**
  - Cargo feature flag: `parallel` (default on, disabled in debug builds for faster compilation)
  - Runtime toggle: `SimConfig.parallel_execution: bool` — parallel by default, sequential for deterministic runs
  - Sequential mode required for reproducible simulations (e.g., RL training, CI assertions)

### V3.6: Hooks System (~2 days)
- `SimulationHook` trait
- Metrics hook, persistence hook
- TUI becomes a hook (optional observer)

### V3.7: Simulation Containerization (~2 days)

**Goal:** Containerized simulation for reproducible benchmarks, CI/CD, and V4 foundation.

**Distroless deployment:**
```dockerfile
# dockerfile/Dockerfile.simulation
FROM rust:1.75-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin quant-trading-gym

FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=builder /app/target/release/quant-trading-gym /
COPY --from=builder /app/config /config
ENTRYPOINT ["/quant-trading-gym"]
CMD ["--headless", "--config", "/config/default.toml"]
```

**Why distroless?**
- No shell, no package manager → minimal attack surface
- ~20MB image vs ~80MB slim, ~1GB full
- Forces proper configuration (no SSH-in-and-fix)
- `nonroot` user by default

**Deliverables:**
- Multi-stage Dockerfile with distroless runtime
- `docker-compose.yaml` for local dev:
  ```yaml
  services:
    simulation:
      build: .
      volumes:
        - ./config:/config      # Runtime config
      environment:
        - SIM_TICKS=100000
        - SIM_AGENTS=1000
  ```
- `--headless` flag (disables TUI, requires V3.6)
- Environment-based config override (`SIM_*` env vars)
- Volume mounts for config files (SQLite persistence added in V3.9)
- GitHub Actions workflow: build → test → push to GHCR
- Health check endpoint (`/health` via minimal HTTP, or exit code)

**File structure:**
```
dockerfile/
  Dockerfile.simulation   # Distroless production image
  Dockerfile.dev          # Full image with debug tools (optional)
docker-compose.yaml       # Local development
.github/
  workflows/
    docker.yaml           # CI/CD pipeline
```

**Declarative:** Config via TOML + env vars, not code changes.
**Modular:** Base image reused by V4-Game service containers.
**SoC:** Container = runtime concern, separate from simulation logic.

**Maps to Original:** Part 16 (Containerization & Deployment)

### V3.8: Performance Profiling (~2 days)

**Goal:** Identify which parallelization strategies provide optimal performance.

**Runtime Parallelization Control:**
- All parallel functions accept `force_sequential: bool` parameter
- `ParallelizationConfig` with 9 independently controllable phases:
  - Agent collection, indicators, order validation, auctions
  - Candle updates, trade updates, fill notifications
  - Wake conditions, risk tracking
- CLI/environment variables for runtime control (no recompilation)

**Profiling Infrastructure:**
- PowerShell script (`run_profiling.ps1`) for automated benchmarking
- Tests 11 configurations (all-parallel, 9 individual sequential, all-sequential)
- 3 trials per config, outputs CSV with timing/throughput data
- Uses full production agent configuration for realistic results

**Deliverables:**
```rust
// crates/simulation/src/config.rs
pub struct ParallelizationConfig {
    pub parallel_agent_collection: bool,
    pub parallel_indicators: bool,
    pub parallel_order_validation: bool,
    pub parallel_auctions: bool,
    pub parallel_candle_updates: bool,
    pub parallel_trade_updates: bool,
    pub parallel_fill_notifications: bool,
    pub parallel_wake_conditions: bool,
    pub parallel_risk_tracking: bool,
}
```

**Usage:**
```bash
# Single phase sequential
PAR_AUCTIONS=false cargo run --release --all-features -- --headless --ticks 1000

# Automated profiling
.\run_profiling.ps1  # Outputs profiling_results.csv
```

**Analysis:** 2^9 = 512 total permutations; script tests 11 key configurations to identify high-impact vs low-impact parallelization phases.

**Maps to Original:** Part 12 (Performance Tuning) extension

### V3.9: SQLite Storage (~4 days)
- Trade history persistence
- Candle aggregation (1m, 5m, 1h)
- Portfolio snapshots
- **Game snapshots for save/resume** (`GameSnapshot`, `AgentSnapshot`)
- **Trade log** (append-only, for post-game analysis)
- **API consideration:** Change `on_fill(Trade)` → `on_fill(Fill)` to expose per-order slippage metrics to agents (V2.2 infrastructure ready, deferred here to avoid early API churn)
- **Docker integration:** Add `./data:/data` volume mount to `docker-compose.yaml`

**Maps to Original:** Phase 10 (Storage)

**Borrow-Checking Pitfalls to Address:**
1. **Multi-symbol state updates (V3.1):** Use interior mutability or collect updates, apply sequentially
2. **Parallel agent execution (V3.5):** Two-phase tick (immutable read → sequential write)
3. **WakeConditionIndex updates (V3.2):** Collect `ConditionUpdate` during tick, apply after
4. **Background pool accounting (V3.4):** Append-only fill recording
5. **Multi-symbol strategy reads (V3.3):** Return owned values from `Market` queries; no overlapping borrows

**Maps to Original:** Phase 6 (Agent Scaling) + Phase 10 (Storage) + Phase 12 (Scale Testing)

**Optional additions at V3:** 
- `NewsReactiveTrader` (Tier 2 strategy that wakes on `FundamentalEvent`s, requires V3.2)

**Strategy Refinements to Consider at V3:**
- **VWAP Executor**: Currently configured as a buyer accumulating 1000 shares. This is an *execution algorithm*, not a *strategy*. In real markets, VWAP execution is used to fill large orders while minimizing impact. Options:
  1. Remove from default agents (it's infrastructure, not a standalone strategy)
  2. Convert to seller with initial position (liquidation scenario)
  3. Make it a child execution layer that other strategies delegate to
  4. Accept underperformance in repricing markets (current behavior)
- **Momentum/TrendFollower**: Low activity in mean-reverting tick-level simulation (realistic for HFT). Consider wider thresholds or daily timeframe aggregation if more activity desired.

**Multi-Symbol Agent Design Notes:**
Multi-symbol agents (e.g., `PairsTrading`, `SectorRotator`) differ from single-symbol agents:
- They observe multiple symbols simultaneously via `StrategyContext.market()`
- They may generate orders for multiple symbols in a single `on_tick()`
- Position limits apply per-symbol; portfolio-level risk is the agent's responsibility
- For V3 Tier 2: wake on ANY watched symbol's condition, receive single trigger symbol

**Agent Trait Extensions (V3.1):**
```rust
trait Agent {
    fn position(&self) -> i64;  // Aggregate sum across all symbols
    fn position_for(&self, symbol: &str) -> i64;  // Per-symbol position
    fn positions(&self) -> &HashMap<Symbol, PositionEntry>;
    fn watched_symbols(&self) -> &[Symbol] { &[] }  // For Tier 2 wake conditions (V3.2)
    fn equity(&self, prices: &HashMap<Symbol, Price>) -> Cash;  // Mark-to-market
    fn equity_for(&self, symbol: &str, price: Price) -> Cash;  // Single-symbol equity
}

struct PositionEntry {
    quantity: i64,
    avg_cost: f64,  // Weighted average cost basis
}
```

---

## V4: Choose Your Track

At this point, you have a solid simulation. Pick ONE track based on interest:

### V4-RL: Reinforcement Learning Track (+5 weeks)

```
V4-RL.1: Gym Environment (1 wk)
    └─► TradingEnv with step/reset
    
V4-RL.2: Observations (1 wk)
    └─► Price, book depth, portfolio features
    └─► Observation parity contract
    
V4-RL.3: Rewards (3 days)
    └─► PnL, Sharpe, Drawdown penalties
    
V4-RL.4: PyO3 Bindings (1 wk)
    └─► Python can call Rust env
    
V4-RL.5: Training (1 wk)
    └─► DQN/PPO with stable-baselines3
    └─► Export to ONNX
```

**Maps to Original:** Phases 13-18 (RL Track)

### V4-Game: Multiplayer Game Track (+8 weeks)

**Critical Insight:** Humans cannot compete at AI tick speeds (<10ms). The game track MUST include time controls and a quant dashboard, otherwise humans lose before they can comprehend the market state.

**Service Architecture:** 4 services (consolidated from original 8)

```
┌─────────────────────────────────────────────────────────────┐
│                      SIMULATION                             │
│  (sync, computes everything for agents)                     │
│  - Matching engine, agent tick loop                         │
│  - Lightweight indicators (for agents)                      │
│  - News generation                                          │
│  - Emits: TickEvent { prices, trades, portfolios }          │
└──────────────────────────┬──────────────────────────────────┘
                           │ broadcast
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │    DATA     │   │    GAME     │   │   STORAGE   │
  │   SERVICE   │   │   SERVICE   │   │   SERVICE   │
  │   :8001     │   │   :8002     │   │   :8003     │
  │             │   │             │   │             │
  │ /analytics/*│   │ /game/*     │   │ /storage/*  │
  │ /portfolio/*│   │ WebSocket   │   │ Snapshots   │
  │ /risk/*     │   │ Sessions    │   │ Trade log   │
  │ /news/*     │   │ Time ctrl   │   │ Queries     │
  │             │   │ Orders      │   │             │
  └─────────────┘   └─────────────┘   └─────────────┘
         │                 │
         └────────┬────────┘
                  ▼
           ┌─────────────┐
           │   CHATBOT   │
           │   :8004     │
           │  NLP → API  │
           └─────────────┘
```

```
V4-G.1: Services Foundation (1 wk)
    └─► Axum, async bridge, error handling
    
V4-G.2: Data Service (1.5 wks)
    └─► /analytics/* (candles, indicator history, charts)
    └─► /portfolio/* (positions, P&L, equity curves)
    └─► /risk/* (VaR, drawdown, exposure, alerts)
    └─► /news/* (event history, sentiment)
    
V4-G.3: Game Service + Dashboard BFF (2 wks)
    └─► WebSocket stream (real-time tick updates)
    └─► Sessions, matchmaking, lobby
    └─► Time control system (pause/step/speed)
    └─► Order submission
    └─► BFF aggregation (combines Data + Storage for frontend)
    └─► Save/Resume system:
        - `Simulation::to_snapshot()` / `from_snapshot()`
        - Auto-save every N ticks (configurable, default 10k)
        - Manual save (game continues)
        - Save & Exit (save + pause + lobby)
        - Resume from lobby (load snapshot, continue)
    
V4-G.4: Chatbot Service (1 wk)
    └─► NLP intent parsing (LLM function calling)
    └─► Routes to Game/Data/Storage
    └─► "Buy 100 ACME" → Game service
    └─► "What's my P&L?" → Data service
    └─► "Show my trades" → Storage service
    
V4-G.5: Frontend + Human Dashboard (2.5 wks)
    └─► React/TypeScript base UI
    └─► Order book, chart, portfolio views
    └─► Time controls UI (pause, step, speed slider)
    └─► Indicator panel (SMA, RSI, MACD, Bollinger)
    └─► Factor gauges (momentum, value, volatility)
    └─► Risk dashboard (VaR, drawdown, exposure)
    └─► Signal summary (aggregate buy/sell/hold)
    └─► Quick trade buttons, alert banners
    └─► Save/load UI, leaderboard, game lobby
```

#### Dashboard Integration Architecture

The frontend calls Game service, which acts as **BFF (Backend-For-Frontend)**:

```
Frontend (React)
      │
      │ Single WebSocket + REST
      ▼
Game Service :8002  (/game/dashboard, /game/stream)
      │
      ├──► Data :8001    →  indicators, portfolio, risk, news
      └──► Storage :8003 →  snapshots, trade history
```

**Benefits:** Single connection, consistent tick data, no CORS complexity, simplified frontend.

#### Human Player Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Time Controls** | Pause/Step/Slow(1 tick/s)/Normal(10 tick/s)/Fast(100 tick/s) |
| **Quant Dashboard** | Same indicators AI sees, visualized |
| **Decision Support** | Quick trade buttons, risk preview, bracket orders |
| **Information Parity** | Humans see everything AI observes |

#### Time Control Modes

| Mode | Speed | Use Case |
|------|-------|----------|
| Paused | Frozen | Analysis, planning |
| Step | Manual | Learning, debugging |
| Slow | 1 tick/sec | Comfortable play |
| Normal | 10 tick/sec | Engaged play |
| Fast | 100 tick/sec | Skip boring periods |

#### Dashboard Panels

| Panel | Shows | Source (via BFF) |
|-------|-------|------------------|
| Indicator Panel | SMA, EMA, RSI, MACD, Bollinger, ATR | Data :8001 → /analytics/* |
| Factor Gauges | Momentum, Value, Volatility scores | Data :8001 → /analytics/* |
| Risk Dashboard | VaR, Sharpe, max drawdown | Data :8001 → /risk/* |
| Portfolio | Holdings, P&L, equity curve | Data :8001 → /portfolio/* |
| Signal Summary | Strong Buy → Strong Sell | Game :8002 → aggregated |
| News Feed | Active events with sentiment | Data :8001 → /news/* |

**Maps to Original:** Phases 13-22 (Game Track) + Part 11 (Human Player Interface)

#### Containerization (V4-Game)

Extends V3.7 base image for multi-service deployment:

| Environment | Tooling | Use Case |
|-------------|---------|----------|
| Development | Docker Compose | Local multi-service testing |
| Staging | Docker Compose | Demo, integration testing |
| Production | Kubernetes | Scalable cloud deployment |

Key elements:
- **Reuses V3.7 distroless base** for simulation service
- Per-service Dockerfiles (Data, Game, Storage, Chatbot)
- Health checks on all services (`/health` endpoint)
- Environment-based configuration (`.env` files)
- CI/CD builds on push to main

**See:** V3.7 for base simulation container, Part 16 (Containerization & Deployment) in full plan

---

## V5: Full Integration (+1 week)

If you did BOTH V4-RL and V4-Game:
- RL agents as game opponents
- Leaderboards with RL baselines
- "Beat the Bot" game mode


**Maps to Original:** Phase 23 (RL Game Integration)

---

## Architectural Considerations (V2+)

### Multi-Symbol Support (V2)

Multi-symbol infrastructure is added in V2 because:
1. `TieredOrchestrator` needs agent-symbol relationships
2. `WakeConditionIndex` benefits from symbol-scoped indexing
3. Background pool sentiment should be per-sector
4. Pairs trading strategy requires correlated symbols

```rust
// crates/sim-core/market.rs
pub struct Market {
    books: HashMap<Symbol, OrderBook>,  // Multiple symbols
    pending: PendingOrderQueue,
}
```

### Position Limits & Short-Selling (V2)

**Problem:** V1 allows unrestricted positions (agents with -1500 shares = unrealistic).

**Solution:** Natural constraints for long positions, explicit infrastructure for shorts:

| Constraint | Type | Implementation |
|------------|------|----------------|
| **Long positions** | Natural | Cash available + `shares_outstanding` per symbol |
| **Short positions** | Explicit | `max_short` per agent + borrow availability |
| **Borrow pool** | Derived | % of `shares_outstanding` available to borrow |
| **Locate** | Optional | Require locate before shorting |

No artificial `max_long` — you can buy as many shares as exist and you can afford.

### Reactive Agent Parametric Conditions (V3.2)

Agents can update their wake conditions at runtime without being recreated:

```rust
pub struct ConditionUpdate {
    pub agent_id: AgentId,
    pub remove: Vec<WakeCondition>,
    pub add: Vec<WakeCondition>,
}
```

**Use cases:**
- Price thresholds become stale as market moves → update thresholds
- Sector rotation → change news filters
- Volatility regimes → adjust time intervals

### Borrow-Checking Pitfalls by Version

| Version | Pitfall | Solution |
|---------|---------|----------|
| V3.1 | Multi-symbol state mutation | Return owned `PositionEntry`, update sequentially after tick |
| V3.2 | WakeConditionIndex updates during tick | Deferred `ConditionUpdate` buffer |
| V3.3 | Multi-symbol strategy reads | Return owned values from `Market` queries; no overlapping borrows |
| V3.4 | Background pool accounting | Append-only fill recording |
| V3.5 | Parallel agent execution | Two-phase tick: read (parallel) → write (sequential) |
| V3.6 | SimulationHook borrows | Sequential hook invocation |
| V3.8 | Snapshot during active tick | Snapshots only at tick boundaries |
| V4-RL | PyO3 GIL blocking | `py.allow_threads()` for Rust computation |
| V4-Game | Async/sync boundary | Channel-based `SimulationBridge` |

### Two-Phase Tick Architecture (V3.5)

```rust
impl TieredOrchestrator {
    pub fn tick(&mut self, market: &Market) -> Vec<Order> {
        // Phase 1: Read (parallel-safe, borrows &Market immutably)
        let tier1_orders = self.run_tier1_parallel(market);  // rayon
        let tier2_orders = self.run_tier2_triggered(market);
        let tier3_orders = self.pool.generate(market);
        
        // Phase 2: Collect (orders returned, applied by Simulation)
        [tier1_orders, tier2_orders, tier3_orders].concat()
    }
}
```

---

## Iteration Timeline Summary

| Version | Focus | Status |
|---------|-------|--------|
| V0 | MVP Simulation | ✅ Complete |
| V1 | Quant Strategy Agents | ✅ Complete |
| V2 | Multi-Symbol & Events | ✅ Complete |
| V3 | Scaling & Persistence | 🔲 Planned |
| V4 | RL Track OR Game Track | 🔲 Planned |

---

## Crate Evolution Map

How crates grow across versions:

```
V0                  V1                  V2                  V3
─────────────────────────────────────────────────────────────────────────
types/              types/              types/              types/
  lib.rs              lib.rs              lib.rs              lib.rs
                                          order.rs            order.rs
                                          config.rs           config.rs

sim-core/           sim-core/           sim-core/           sim-core/
  lib.rs              lib.rs              lib.rs              lib.rs
  order_book.rs       order_book.rs       order_book.rs       order_book.rs
  matching.rs         matching.rs         matching.rs         matching.rs
                                          market.rs           market.rs
                                          slippage.rs         slippage.rs

                    quant/              quant/              quant/
                      lib.rs              lib.rs              lib.rs
                      indicators/         indicators/         indicators/
                        mod.rs              mod.rs              mod.rs
                        sma.rs              sma.rs              sma.rs
                        ema.rs              ema.rs              ema.rs
                        rsi.rs              rsi.rs              rsi.rs
                        macd.rs             macd.rs             macd.rs
                        bollinger.rs        bollinger.rs        bollinger.rs
                      engine.rs           engine.rs           engine.rs
                      rolling.rs          rolling.rs          rolling.rs
                      risk.rs             risk.rs             risk.rs
                      tracker.rs          tracker.rs          tracker.rs
                      stats.rs            stats.rs            stats.rs

                                        news/               news/
                                          lib.rs              lib.rs
                                          events.rs           events.rs
                                          fundamentals.rs     fundamentals.rs
                                          generator.rs        generator.rs
                                          config.rs           config.rs
                                          sectors.rs          sectors.rs

agents/             agents/             agents/             agents/
  lib.rs              lib.rs              lib.rs              lib.rs
  traits.rs           traits.rs           traits.rs           traits.rs
                                          context.rs          context.rs
                                          state.rs            state.rs (V3.1: multi-symbol)
                                          position_limits.rs  position_limits.rs
                                          borrow_ledger.rs    borrow_ledger.rs
                                                              tiers.rs (V3.2: WakeCondition, etc.)
  strategies/         strategies/         strategies/         strategies/
    mod.rs              mod.rs              mod.rs              mod.rs
    noise_trader.rs     noise_trader.rs     noise_trader.rs     noise_trader.rs
    market_maker.rs     market_maker.rs     market_maker.rs     market_maker.rs
                        momentum.rs         momentum.rs         momentum.rs
                        trend_follower.rs   trend_follower.rs   trend_follower.rs
                        macd_crossover.rs   macd_crossover.rs   macd_crossover.rs
                        bollinger.rs        bollinger.rs        bollinger.rs
                        vwap_executor.rs    vwap_executor.rs    vwap_executor.rs
                                                              tier2/ (V3.2)
                                                                mod.rs
                                                                agent.rs
                                                                wake_index.rs
                                                                strategies.rs
                                                              tier3/ (V3.4)
                                                                mod.rs
                                                                pool.rs

simulation/         simulation/         simulation/         simulation/
  lib.rs              lib.rs              lib.rs              lib.rs
  runner.rs           runner.rs           runner.rs           runner.rs
                                          config.rs           config.rs
                                                              orchestrator.rs (V3.2: TieredOrchestrator)
                                                              hooks.rs (V3.6)

tui/                tui/                tui/                tui/ (becomes hook)
  lib.rs              lib.rs              lib.rs              lib.rs
  app.rs              app.rs              app.rs              app.rs
                                          widgets/            widgets/
                                            mod.rs              mod.rs
                                            update.rs           update.rs
                                            price_chart.rs      price_chart.rs
                                            book_depth.rs       book_depth.rs
                                            stats_panel.rs      stats_panel.rs
                                            agent_table.rs      agent_table.rs
                                            risk_panel.rs       risk_panel.rs

                                                              storage/
                                                                lib.rs
                                                                schema.rs
                                                                trades.rs
                                                                candles.rs
                                                                snapshots.rs

src/                src/                src/                src/
  main.rs             main.rs             main.rs             main.rs
                                          config.rs           config.rs

                                        docs/               docs/
                                          executive_summary   executive_summary
                                          technical_summary   technical_summary
```

**Key Migration Points:**
- **V0→V1:** Added `quant/` crate with indicators, risk metrics
- **V1→V2:** Added `news/` crate, `context.rs` moved to agents, multi-symbol `market.rs`, slippage, position limits
- **V2→V3.1:** Refactor `AgentState` to multi-symbol `positions: HashMap<Symbol, PositionEntry>`
- **V3.1→V3.2:** Add `tier2/`, `tiers.rs`, `orchestrator.rs`, `WakeConditionIndex`
- **V3.2→V3.3:** Add `tier1/strategies/pairs_trading.rs`, `tier2/strategies/sector_rotator.rs`, extend `quant/stats.rs`
- **V3.3→V3.4:** Add `tier3/` with `BackgroundAgentPool`
- **V3.4→V3.5:** Performance tuning, two-phase tick (no new files, optimization pass)
- **V3.5→V3.6:** Implement `SimulationHook` trait, TUI becomes hook
- **V3.6→V3.7:** Add `dockerfile/`, `docker-compose.yaml`, `--headless` flag, CI workflow
- **V3.7→V3.8:** Add `ParallelizationConfig`, runtime parallelization control, profiling script
- **V3.8→V3.9:** Add `storage/` crate

---

## What We're NOT Doing (Yet)

Explicitly deferred to keep V0-V2 lean:

| Feature | Deferred To | Reason |
|---------|-------------|--------|
| 100k agent scale | V3 | Optimization, not learning |
| Database persistence | V3 | Tedious plumbing |
| Tier 2/3 agents | V3 | Scale optimization |
| Multi-threading | V3+ | Single-threaded is simpler to debug |
| ONNX inference | V4-RL | Requires full gym first |
| HTTP services | V4-Game | Requires stable core first |
| React frontend | V4-Game | TUI is enough for learning |

---

## Strategy Roadmap

| Strategy | Version | Status | Notes |
|----------|---------|--------|-------|
| **NoiseTrader** | V0 | ✅ | Random trades around fair value |
| **MarketMaker** | V0 | ✅ | Two-sided quotes, inventory management |
| **Momentum (RSI)** | V1 | ✅ | Buy oversold, sell overbought; low activity in mean-reverting market |
| **TrendFollower (SMA)** | V1 | ✅ | Golden/death cross signals |
| **MACD Crossover** | V1 | ✅ | MACD/signal line crossover |
| **Bollinger Reversion** | V1 | ✅ | Mean reversion at bands |
| **VWAP Executor** | V1 | ✅ | Execution algo (accumulates shares); see V3 notes |
| **Pairs Trading** | V3.3 | 🔲 | Tier 1 multi-symbol, cointegration-based spread trading |
| **Sector Rotator** | V3.3 | 🔲 | Tier 2 multi-symbol, sentiment-driven allocation |
| **Factor Long-Short** | V3.3+ | 🔲 | Requires `quant/factors.rs` (value, momentum, quality) |
| **ThresholdBuyer/Seller** | V3.2 | 🔲 | Tier 2 reactive strategy |
| **News Reactive** | V3.2 | 🔲 | Tier 2 wake on `FundamentalEvent` |
| **RL Agent** | V4 | 🔲 | Requires gym + ONNX |

**Notes:**
- Momentum/TrendFollower have low activity — realistic for tick-level mean-reverting markets
- VWAP is an execution algorithm, not a strategy; consider restructuring in V3.5

---

## Success Metrics

| Version | Metric | Status |
|---------|--------|--------|
| V0 | "I can watch agents trade in my terminal" | ✅ Achieved |
| V1 | "My agents use real indicators and I see risk metrics" | ✅ Achieved |
| V2 | "Prices anchor to fundamentals; events move markets" | ✅ Achieved |
| V3 | "100k agents without OOM; trades persist across runs" | 🔲 Planned |
| V4-RL | "I trained an RL agent that beats noise traders" | 🔲 Planned |
| V4-Game | "I can play, pause, analyze, and make informed trades" | 🔲 Planned |

---

## Getting Started

```bash
# Create workspace
cargo new quant-trading-gym
cd quant-trading-gym

# Set up workspace Cargo.toml
cat > Cargo.toml << 'EOF'
[workspace]
members = [
    "crates/types",
    "crates/sim-core",
    "crates/agents",
    "crates/simulation",
    "crates/tui",
]
resolver = "2"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
ratatui = "0.28"
crossterm = "0.28"
EOF

# Create crates (Week 1-2)
cargo new crates/types --lib
cargo new crates/sim-core --lib

# Create crates (Week 2-3)
cargo new crates/agents --lib
cargo new crates/simulation --lib

# Create crate (Week 4)
cargo new crates/tui --lib

# Start coding Week 1!
```

---

## Current Crate Structure (V2.4)

| Crate | Files | Purpose |
|-------|-------|-------|
| `types` | `lib.rs`, `order.rs`, `config.rs` | Order, Trade, Price, Symbol, Sector, SymbolConfig |
| `sim-core` | `lib.rs`, `order_book.rs`, `matching.rs`, `market.rs`, `slippage.rs` | OrderBook, MatchingEngine, Market (multi-symbol) |
| `agents` | `lib.rs`, `traits.rs`, `context.rs`, `state.rs`, `position_limits.rs`, `borrow_ledger.rs` | Agent trait, StrategyContext, position validation |
| `agents/strategies` | `mod.rs`, `noise_trader.rs`, `market_maker.rs`, `momentum.rs`, `trend_follower.rs`, `macd_crossover.rs`, `bollinger_reversion.rs`, `vwap_executor.rs` | 7 Tier 1 strategies |
| `news` | `lib.rs`, `events.rs`, `fundamentals.rs`, `generator.rs`, `config.rs`, `sectors.rs` | Events, Gordon Growth Model, SectorModel |
| `quant` | `lib.rs`, `engine.rs`, `indicators/`, `risk.rs`, `tracker.rs`, `rolling.rs`, `stats.rs` | SMA, EMA, RSI, MACD, Bollinger, Sharpe, MaxDD, VaR |
| `simulation` | `lib.rs`, `runner.rs`, `config.rs` | Tick loop, event processing, agent orchestration |
| `tui` | `lib.rs`, `app.rs`, `widgets/` | Terminal UI with price chart, book depth, agent table, risk panel |

**V3.x Migration Notes:**
- **V3.1:** Refactor `state.rs` for multi-symbol positions; update trait in `traits.rs`
- **V3.2:** Add `tiers.rs`, `tier2/` module with `agent.rs`, `wake_index.rs`, `strategies.rs`; add `orchestrator.rs` to simulation
- **V3.3:** Add `tier1/strategies/pairs_trading.rs`, `tier2/strategies/sector_rotator.rs`, extend `quant/stats.rs`
- **V3.4:** Add `tier3/` module with `pool.rs`
- **V3.5:** Performance tuning pass (no new files)
- **V3.6:** Add `hooks.rs` to simulation; refactor TUI to implement `SimulationHook`
- **V3.7:** Add `dockerfile/`, `docker-compose.yaml`, CI workflow
- **V3.8:** Add `storage/` crate
