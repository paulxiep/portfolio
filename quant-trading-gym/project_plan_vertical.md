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

## V3: Scaling & Persistence (+3 weeks)

**Add:** Tiered agent architecture for 100k+ scale, storage layer

### V3.1: Tier 2 Reactive Agents (~4 days)
- `ReactiveAgent` struct (lightweight, event-driven)
- Wake conditions: price threshold, interval, event subscription
- `WakeConditionIndex` for O(log n) lookups
- **Event subscription:** Tier 2 agents wake only on relevant `FundamentalEvent`
- **Parametric condition updates** — modify wake conditions at runtime

### V3.2: Tier 3 Background Pool (~4 days)
- Statistical order generation (no individual agents)
- `BackgroundAgentPool` struct
- Configurable distributions (size, price, direction)
- **Sentiment-driven:** Pool bias shifts with active `FundamentalEvent`s
- Per-sector sentiment tracking

### V3.3: Performance Tuning (~3 days)
- Benchmark 100k agents
- Profile and optimize hot paths
- Memory budget validation
- Two-phase tick architecture (read phase parallel, write phase sequential)

### V3.4: SQLite Storage (~4 days)
- Trade history persistence
- Candle aggregation (1m, 5m, 1h)
- Portfolio snapshots
- **Game snapshots for save/resume** (`GameSnapshot`, `AgentSnapshot`)
- **Trade log** (append-only, for post-game analysis)
- **API consideration:** Change `on_fill(Trade)` → `on_fill(Fill)` to expose per-order slippage metrics to agents (V2.2 infrastructure ready, deferred here to avoid early API churn)

### V3.5: Hooks System (~2 days)
- `SimulationHook` trait
- Metrics hook, persistence hook
- TUI becomes a hook (optional observer)

**Borrow-Checking Pitfalls to Address:**
1. **Parallel agent execution:** Two-phase tick (immutable read → sequential write)
2. **WakeConditionIndex updates:** Collect `ConditionUpdate` during tick, apply after
3. **Background pool accounting:** Append-only fill recording

**Maps to Original:** Phase 6 (Agent Scaling) + Phase 10 (Storage) + Phase 12 (Scale Testing)

**Optional additions at V3:** 
- `NewsReactiveTrader` (Tier 2 strategy that wakes on `FundamentalEvent`s from V2.4)
- `PairsTrading` (multi-symbol agent exploiting cointegration, requires V2.3 multi-symbol + `quant/stats.rs`)
- `SectorRotator` (shifts allocation based on sector sentiment from V2.4 events)

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

**Agent Trait Extensions for V3:**
```rust
trait Agent {
    // Existing (V0)
    fn position(&self) -> i64;  // Keep for backward compat (sum of positions)
    
    // New for multi-symbol (V3)
    fn positions(&self) -> HashMap<Symbol, i64> { HashMap::new() }
    fn watched_symbols(&self) -> Vec<Symbol> { vec![] }  // For Tier 2 wake conditions
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

All services containerized for environment-agnostic deployment:

| Environment | Tooling | Use Case |
|-------------|---------|----------|
| Development | Docker Compose | Local multi-service testing |
| Staging | Docker Compose | Demo, integration testing |
| Production | Kubernetes | Scalable cloud deployment |

Key elements:
- Multi-stage Dockerfiles (Rust builder → slim runtime)
- Health checks on all services (`/health` endpoint)
- Environment-based configuration (`.env` files)
- CI/CD builds on push to main

**See:** Part 16 (Containerization & Deployment) in full plan

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

### Reactive Agent Parametric Conditions (V2)

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
| V2 | Parallel agent execution | Two-phase tick: read (parallel) → write (sequential) |
| V2 | WakeConditionIndex updates during tick | Deferred `ConditionUpdate` buffer |
| V2 | Background pool accounting | Append-only fill recording |
| V3 | SimulationHook borrows | Sequential hook invocation |
| V3 | Snapshot during active tick | Snapshots only at tick boundaries |
| V4-RL | PyO3 GIL blocking | `py.allow_threads()` for Rust computation |
| V4-Game | Async/sync boundary | Channel-based `SimulationBridge` |

### Two-Phase Tick Architecture (V2)

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

| Version | Focus | Status | Tests |
|---------|-------|--------|-------|
| V0 | MVP Simulation | ✅ Complete | — |
| V1 | Quant Strategy Agents | ✅ Complete | — |
| V2.1 | Position Limits & Short-Selling | ✅ Complete | — |
| V2.2 | Slippage & Partial Fills | ✅ Complete | — |
| V2.3 | Multi-Symbol Infrastructure | ✅ Complete | — |
| V2.4 | Fundamentals & Events | ✅ Complete | 213 |
| V3 | Scaling & Persistence | 🔲 Planned | — |
| V4 | RL Track OR Game Track | 🔲 Planned | — |

**Current State (V2.4):**
- 7 agent strategies (MarketMaker, NoiseTrader, Momentum, TrendFollower, MACD, Bollinger, VWAP)
- 4 symbols across 4 sectors
- News events with Gordon Growth Model fair value
- TUI with start/pause control, symbol tabs, price overlay
- Mean-reverting tick-level market (realistic for HFT)

**Next Decision:** V3 (Scaling) or jump to V4 (RL/Game)?

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
                                          state.rs            state.rs
                                          position_limits.rs  position_limits.rs
                                          borrow_ledger.rs    borrow_ledger.rs
  strategies/         strategies/         strategies/         strategies/
    mod.rs              mod.rs              mod.rs              mod.rs
    noise_trader.rs     noise_trader.rs     noise_trader.rs     noise_trader.rs
    market_maker.rs     market_maker.rs     market_maker.rs     market_maker.rs
                        momentum.rs         momentum.rs         momentum.rs
                        trend_follower.rs   trend_follower.rs   trend_follower.rs
                        macd_crossover.rs   macd_crossover.rs   macd_crossover.rs
                        bollinger.rs        bollinger.rs        bollinger.rs
                        vwap_executor.rs    vwap_executor.rs    vwap_executor.rs
                                                              tier2/
                                                                mod.rs
                                                                agent.rs
                                                                wake_index.rs
                                                              tier3/
                                                                mod.rs
                                                                pool.rs

simulation/         simulation/         simulation/         simulation/
  lib.rs              lib.rs              lib.rs              lib.rs
  runner.rs           runner.rs           runner.rs           runner.rs
                                          config.rs           config.rs
                                                              hooks.rs

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
- **V2→V3:** Add `tier2/`, `tier3/`, `storage/`, implement `SimulationHook` trait

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
| **Pairs Trading** | V3 | 🔲 | Requires multi-symbol cointegration |
| **Factor Long-Short** | V3 | 🔲 | Requires `quant/factors.rs` (value, momentum, quality) |
| **News Reactive** | V3 | 🔲 | Requires Tier 2 wake system |
| **Sector Rotator** | V3 | 🔲 | Requires sector sentiment tracking |
| **RL Agent** | V4 | 🔲 | Requires gym + ONNX |

**Notes:**
- Momentum/TrendFollower have low activity — realistic for tick-level mean-reverting markets
- VWAP is an execution algorithm, not a strategy; consider restructuring in V3

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

**V3 Migration:** Strategies stay flat (no tier1/ folder). Tier 2/3 get separate modules when implemented.
