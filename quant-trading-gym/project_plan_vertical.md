# Quant Trading Gym: Vertical plan

Ideated with LLM assistance, summarised by Opus 4.5 Thinking.

Refer to [full plan here](project_plan.md)

---

## Philosophy

**Build vertically, not horizontally.**

Instead of completing all types → all matching → all agents → all viz, we build a thin slice through the entire stack each iteration. Every iteration produces something *runnable*.

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
 ├──► V1: Quant Layer
 │
 ├──► V2: Agent Scaling
 │
 ├──► V3: Persistence
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

## V2: Agent Scaling (+2 weeks)

**Add:** Tiered agent architecture for 100k+ scale

### V2.1: Tier 2 Reactive Agents (~5 days)
- `ReactiveAgent` struct (lightweight)
- Wake conditions (price cross, interval)
- `WakeConditionIndex` for O(1) lookups

### V2.2: Tier 3 Background Pool (~4 days)
- Statistical order generation (no individual agents)
- `BackgroundAgentPool` struct
- Configurable distributions (size, price, direction)

### V2.3: Performance Tuning (~3 days)
- Benchmark 100k agents
- Profile and optimize hot paths
- Memory budget validation

**Maps to Original:** Phase 6 (Agent Scaling) + Phase 12 (Scale Testing)

**Optional additions at V2:** More Tier 1 strategies for agent variety:
- Phase 7: MACD Crossover, Bollinger Reversion, Trend Following
- Phase 8: Pairs Trading, Factor Long-Short, VWAP Executor

These are independent — add based on interest, not obligation.

---

## V3: Persistence & Events (+2 weeks)

**Add:** Storage layer and news events

### V3.1: News Events (~4 days)
- `NewsEvent` type with sentiment
- `NewsGenerator` for random events
- Background pool reacts to news

### V3.2: SQLite Storage (~5 days)
- Trade history persistence
- Candle aggregation
- Portfolio snapshots
- **Game snapshots for save/resume** (`GameSnapshot`, `AgentSnapshot` structs)
- **Trade log** (append-only, for post-game analysis)

### V3.3: Hooks System (~3 days)
- `SimulationHook` trait
- Metrics hook, persistence hook
- TUI becomes a hook

**Maps to Original:** Phase 4 (News infrastructure) + Phase 10 (Storage)

**Optional addition at V3:** `news_reactive.rs` strategy (Phase 8) — now that news infra exists

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

---

## V5: Full Integration (+1 week)

If you did BOTH V4-RL and V4-Game:
- RL agents as game opponents
- Leaderboards with RL baselines
- "Beat the Bot" game mode


**Maps to Original:** Phase 23 (RL Game Integration)

---

## Iteration Timeline Summary

| Version | Focus | Duration | Cumulative |
|---------|-------|----------|------------|
| V0 | Steel Thread MVP | 4 wks | 4 wks |
| V1 | Quant Layer | 2 wks | 6 wks |
| V2 | Agent Scaling | 2 wks | 8 wks |
| V3 | Persistence & Events | 2 wks | 10 wks |
| V4-RL | RL Track | 5 wks | 15 wks |
| V4-Game | Game Track (4 services + frontend) | 8 wks | 18 wks |
| V5 | Full Integration | 1 wk | 19 wks |

**Key Decision Points:**
- After V0: Do you want to keep going? (4 wks invested)
- After V2: Is performance important, or skip to features? (8 wks invested)
- After V3: RL or Game? Pick one first. (10 wks invested)

---

## Crate Evolution Map

How crates grow across versions:

```
V0                  V1                  V2                  V3
─────────────────────────────────────────────────────────────────
types/              types/              types/              types/
  lib.rs              lib.rs              lib.rs              lib.rs
                                          constants.rs        constants.rs

sim-core/           sim-core/           sim-core/           sim-core/
  lib.rs              lib.rs              lib.rs              lib.rs
  order_book.rs       order_book.rs       order_book.rs       order_book.rs
  matching.rs         matching.rs         matching.rs         matching.rs
                                          pending_orders.rs   pending_orders.rs
                                          market.rs           market.rs

                    quant/              quant/              quant/
                      lib.rs              lib.rs              lib.rs
                      indicators.rs       indicators.rs       indicators.rs
                      risk.rs             risk.rs             risk.rs
                      stats.rs            stats.rs            stats.rs

                                                            news/
                                                              lib.rs
                                                              generator.rs
                                                              sectors.rs

agents/             agents/             agents/             agents/
  lib.rs              lib.rs              lib.rs              lib.rs
  traits.rs           traits.rs           traits.rs           traits.rs
  strategies/         strategies/         tiers.rs            tiers.rs
    mod.rs              mod.rs            context.rs          context.rs
    noise_trader.rs     noise_trader.rs   orchestrator.rs     orchestrator.rs
    market_maker.rs     market_maker.rs   tier1/              tier1/
                        momentum.rs         mod.rs              mod.rs
                                            agent.rs            agent.rs
                                            strategies/         strategies/
                                              mod.rs              mod.rs
                                              noise_trader.rs ←── (moved from strategies/)
                                              market_maker.rs
                                              momentum.rs
                                          tier2/              tier2/
                                            mod.rs              mod.rs
                                            agent.rs            agent.rs
                                            wake_index.rs       wake_index.rs
                                          tier3/              tier3/
                                            mod.rs              mod.rs
                                            pool.rs             pool.rs

simulation/         simulation/         simulation/         simulation/
  lib.rs              lib.rs              lib.rs              lib.rs
  runner.rs           runner.rs           runner.rs           runner.rs
  context.rs          context.rs          context.rs          context.rs
                                          config.rs           config.rs
                                          metrics.rs          metrics.rs
                                                              hooks.rs

tui/                tui/                tui/                tui/ (optional)
  lib.rs              lib.rs              lib.rs              lib.rs
  app.rs              app.rs              app.rs              app.rs
  price_chart.rs      price_chart.rs      price_chart.rs      price_chart.rs
  book_depth.rs       book_depth.rs       book_depth.rs       book_depth.rs
                      indicators.rs       agents_table.rs     agents_table.rs
                      risk_panel.rs       risk_panel.rs       risk_panel.rs

                                                            storage/
                                                              lib.rs
                                                              schema.rs
                                                              connection.rs
                                                              persistence_hook.rs
                                                              stores/
                                                                trades.rs
                                                                candles.rs
                                                                snapshots.rs
```

**Key Migration Points:**
- **V0→V1:** Add `quant/` crate, wire indicators into `context.rs`
- **V1→V2:** Restructure `agents/strategies/` into `agents/tier1/strategies/`, add tier2/tier3
- **V2→V3:** Add `news/`, `storage/`, implement `SimulationHook` trait, TUI becomes a hook
- **V3→V4:** Add `gym/` (RL track) or `services/` (Game track)

---

## What We're NOT Doing (Yet)

Explicitly deferred to keep V0-V3 lean:

| Feature | Deferred To | Reason |
|---------|-------------|--------|
| 100k agent scale | V2 | Optimization, not learning |
| Database persistence | V3 | Tedious plumbing |
| News/sentiment | V3 | Orthogonal domain |
| Multi-threading | V2+ | Single-threaded is simpler to debug |
| ONNX inference | V4-RL | Requires full gym first |
| HTTP services | V4-Game | Requires stable core first |
| React frontend | V4-Game | TUI is enough for learning |

---

## Strategy Roadmap

Strategies can be added independently once the indicator pipeline (V1) exists:

| Strategy | Full Plan Phase | When to Add | Prerequisites |
|----------|-----------------|-------------|---------------|
| **NoiseTrader** | Phase 5 | **V0** (Week 3) | None |
| **MarketMaker** | Phase 5 | **V0** (Week 3) | None |
| **RSI Momentum** | Phase 7 | **V1** | `quant/indicators.rs` |
| **Trend Following** | Phase 7 | V1+ (optional) | `quant/indicators.rs` |
| **MACD Crossover** | Phase 7 | V1+ (optional) | `quant/indicators.rs` |
| **Bollinger Reversion** | Phase 7 | V1+ (optional) | `quant/indicators.rs` |
| **Pairs Trading** | Phase 8 | V1+ (optional) | `quant/stats.rs` |
| **Factor Long-Short** | Phase 8 | V1+ (optional) | `quant/factors.rs` |
| **VWAP Executor** | Phase 8 | V1+ (optional) | Price history |
| **News Reactive** | Phase 8 | **V3+** | `news/` crate |
| **RL Agent** | Phase 18 | **V4-RL** | Full gym + ONNX |

**Key insight:** After V1, strategies are à la carte. Pick based on interest, not obligation.

---

## Success Metrics

- **V0 Success:** "I can watch agents trade in my terminal"
- **V1 Success:** "My agents use real indicators and I see risk metrics"
- **V2 Success:** "I can run 100k agents without OOM"
- **V3 Success:** "Trades persist across runs; news moves markets"
- **V4-RL Success:** "I trained an RL agent with observation parity"
- **V4-Game Success:** "I can play meaningfully — pause, analyze indicators, make informed trades"
- **V5 Success:** "I can play against my trained RL agent"

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

## File-to-Full-Plan Compatibility Checklist

| V0 File | Full Plan Equivalent | Migration |
|---------|---------------------|-----------|
| `types/lib.rs` | `types/lib.rs` | ✅ Direct |
| `sim-core/lib.rs` | `sim-core/lib.rs` | ✅ Direct |
| `sim-core/order_book.rs` | `sim-core/order_book.rs` | ✅ Direct |
| `sim-core/matching.rs` | `sim-core/matching.rs` | ✅ Direct |
| `agents/lib.rs` | `agents/lib.rs` | ✅ Direct |
| `agents/traits.rs` | `agents/traits.rs` | ✅ Direct |
| `agents/strategies/noise_trader.rs` | `agents/tier1/strategies/noise_trader.rs` | 🔄 Move in V2 |
| `agents/strategies/market_maker.rs` | `agents/tier1/strategies/market_maker.rs` | 🔄 Move in V2 |
| `simulation/lib.rs` | `simulation/lib.rs` | ✅ Direct |
| `simulation/runner.rs` | `simulation/runner.rs` | ✅ Direct |
| `simulation/context.rs` | `agents/context.rs` | 🔄 Move in V2 (context belongs with agents) |
| `tui/*` | (not in full plan) | 🔄 Becomes hook in V3, optional in V4 |
