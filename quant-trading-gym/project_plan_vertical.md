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

# Iterative Expansion: V0 → Full Plan

Each iteration adds a vertical slice of functionality.

```
V0 (MVP Simulation)
 │
 ├──► V1: Quant Layer (indicators, risk, strategies)
 │
 ├──► V2: Events & Market Realism (fundamentals, multi-symbol, position limits)
 │
 ├──► V3: Scaling & Persistence (tiers, storage, hooks)
 │
 ├──► V4: Web Frontend (Axum, React)
 │
 ├──► V5: Feature Engineering ML (PyO3, traditional ML)
 │
 ├──► V6: Reinforcement Learning (Gym env, RL training)
 │
 └──► V7: Portfolio Manager Game (Services, API)
```

---

## V0: MVP Simulation

**Goal:** Single-threaded simulation with TUI visualization showing agents trading.

### V0.1: Core Engine
- `OrderBook` using `BTreeMap<Price, VecDeque<Order>>` with price-time priority
- `LimitOrder`, `Trade`, `Price` (i64 fixed-point) types

### V0.2: Simulation Loop
- `Simulation` struct, tick-based `step()`, `Agent` trait
- `MarketData` snapshot for agents

### V0.3: Basic Agents
- `NoiseTrader`, `MarketMaker` with inventory management
- MarketMaker seeds initial spread to prevent zombie simulation

### V0.4: TUI Visualization
- `ratatui` live charts, order book depth, agent P&L
- Channel-based sim/render threads

**Maps to Original:** Core simulation foundation

---

## V1: Quant Layer

**Add:** Indicators, risk metrics, indicator-based strategies

### V1.1: Indicators
- `quant` crate: SMA, EMA, RSI, MACD, Bollinger, ATR
- Rolling window data structures, wire into `StrategyContext`

### V1.2: Risk Metrics
- `AgentRiskTracker`: Sharpe, Sortino, max drawdown, VaR per agent
- TUI RiskPanel with color-coded metrics

### V1.3: Indicator Strategies
- 5 strategies: Momentum (RSI), TrendFollower (SMA), MacdCrossover, BollingerReversion, VwapExecutor
- Tier configuration system with per-type agent counts

**Maps to Original:** Phase 3 (Quant Foundation) + Phase 7 (Strategies)

---

## V2: Events & Market Realism

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

## V3: Scaling & Persistence

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
**Modular:** Base image reused by V4/V7 service containers.
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

### V3.9: Minimal Storage (~2 days)

**Philosophy:** V3.9 is the **last common ancestor** before V4-V7 features. Build only infrastructure needed by **all** paths. Avoid path-specific features.

**Deliverables:**

1. **Trade History (Append-Only Event Log)**
   - Schema: `(tick, symbol, price, quantity, buyer_id, seller_id)`
   - Purpose: Post-simulation analysis for RL training evaluation and game replay
   - No deletion, no updates — pure event sourcing

2. **Candle Aggregation (Time-Series OLAP)**
   - Schema: `(symbol, timeframe, open, high, low, close, volume, tick_start)`
   - Timeframes: 1m, 5m, 1h (configurable via `SimulationConfig`)
   - Purpose: Chart rendering (Game), episode features (RL)

3. **Portfolio Snapshots (Analysis Checkpoints)**
   - Schema: `(tick, agent_id, cash, positions_json, realized_pnl, equity)`
   - Frequency: Configurable interval (default: every 1000 ticks)
   - Purpose: Performance analysis, NOT save/resume system
   - `positions_json`: `{"AAPL": {"qty": 100, "avg_cost": 15000}}`

4. **Docker Integration**
   - Update `docker-compose.yaml`: add `./data:/data` volume mount
   - CLI: `--storage-path ./data/sim.db` (default: in-memory `:memory:`)
   - Environment variable: `STORAGE_PATH`

5. **Implementation**
   - `crates/storage/` crate with `rusqlite` (sync, no async)
   - `StorageHook` implements `SimulationHook` trait (V3.6)
   - Hooks: `on_trade()`, `on_tick_end()` for snapshot writes

**Deferred to V4+ (Path-Specific Features):**
- ❌ **Game snapshots (save/resume)** → V7 (not required for idle-game model)
- ❌ **Fill-level events** (`on_fill(Fill)` API change) → V6 (if RL training demands it)
- ❌ **Real-time query APIs** → V4 Data Service (`/analytics/*`, `/portfolio/*`)
- ❌ **Agent-level trade attribution** → V7 (leaderboards) or V6 (per-agent reward)

**Why Minimal Scope:**
- RL path needs: trade history (reward engineering), candles (observations), snapshots (episode eval)
- Game path needs: same data exposed via REST APIs (Data Service queries storage)
- All paths **read** from storage; none need online writes during simulation
- Decouples V4-V7 development: V5/V6 extend `crates/gym/`, V4/V7 extend `services/` — zero file conflicts

**Maps to Original:** Phase 10 (Storage) — reduced scope for V4-V7 decoupling

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

## V4-V7: Development Tracks

At this point, you have a solid simulation. The following versions can be developed in any order:

### V4-V7 Decision Framework (Pre-Implementation Planning)

**Status:** V3.9 completes the last common infrastructure. V4-V7 paths are **largely decoupled**:
- V4 (Web Frontend): extends `services/`, Axum backend, React frontend
- V5 (Feature Engineering): extends `crates/gym/`, PyO3 bindings, feature extraction
- V6 (Reinforcement Learning): extends training scripts, reward functions, ONNX inference
- V7 (Portfolio Manager Game): extends `services/`, game API, interactive features
- **Minimal file conflicts** — parallel development ready

#### Game Path Requirements Refinement

**Core Question:** How does a human "play" against 100k AI agents?

**Idle Game Model (Recommended):**
- Human is a **portfolio manager**, not a tick-by-tick trader
- Gameplay: Adjust strategy parameters/formulas during simulation
- No save/resume needed (session-based, like Cookie Clicker)
- No strict time controls needed (simulation runs continuously, human tweaks settings)

**Human Player Tools:**

| Tool | Purpose | Implementation |
|------|---------|----------------|
| **Quant Dashboard** | Display all metrics AI agents see: Sharpe, RSI, MACD, Bollinger, momentum score, value score, volatility | Data Service → `/analytics/*` endpoints |
| **Formula Builder** | Combine metrics with adjustable weights: `buy_signal = 0.4*RSI + 0.3*momentum - 0.2*volatility` | Game Service → formula parser, live recalculation |
| **VWAP Executor Tool** | Human sets target: "accumulate 1000 AAPL shares via VWAP", simulation executes | Game Service → spawns `VwapExecutor` agent for human |
| **Risk Monitor** | Real-time: position size, exposure, VaR, drawdown | Data Service → `/risk/*` endpoints |
| **Event Feed** | Live news events with sentiment scores | Data Service → `/news/*` endpoints |

**Human Decision Loop:**
```
1. Monitor dashboard (all quant metrics visible)
2. Adjust formula weights or thresholds
3. Submit VWAP orders for execution
4. Watch portfolio equity curve
5. Repeat
```

**Competitive Modes:**
- **Sandbox:** Human vs AI agents, no time limit
- **Challenge:** Human vs specific AI strategy (beat MarketMaker, beat Momentum)
- **Leaderboard:** Best Sharpe ratio over 100k ticks

**What This Avoids:**
- ❌ Manual order entry every tick (impossible vs AI speed)
- ❌ Real-time pause/resume (human can tweak anytime)
- ❌ Save/load system (session-based gameplay)
- ❌ Complex matchmaking (single-player initially)

**Open Questions:**
1. Should formula adjustments apply instantly or at next rebalance interval?
2. Should human see AI agent positions (transparency vs competitive fairness)?
3. Should human manage multiple symbols or single-symbol focus?
4. Should there be capital limits (start with $100k) or unlimited?

#### RL Path Algorithm Selection

**Constraint:** Avoid GPU dependency (CPU-only training).

**Recommended Approach: Ensemble of Classical ML**

| Model | Purpose | Library | Training Time (CPU) |
|-------|---------|---------|---------------------|
| **Random Forest** | Non-linear patterns, feature importance | scikit-learn | ~5 min (1000 episodes) |
| **Linear Regression** | Trend following, risk-adjusted returns | scikit-learn | ~1 min |
| **SVM (RBF kernel)** | Regime detection (bull/bear/sideways) | scikit-learn | ~10 min |

**Ensemble Strategy:**
```python
# Each model votes: Buy (+1), Hold (0), Sell (-1)
rf_vote = random_forest.predict(features)
lr_vote = sign(linear_regression.predict(features))
svm_vote = svm.predict(features)

# Weighted ensemble
action = 0.5 * rf_vote + 0.3 * lr_vote + 0.2 * svm_vote
if action > 0.3: return BUY
if action < -0.3: return SELL
return HOLD
```

**Why This Works:**
- RF captures non-linear indicator interactions (RSI + momentum)
- Linear models capture trends and risk premia
- SVM handles regime shifts (mean-reversion vs trending markets)
- No exploration needed (supervised learning from profitable agent data)
- CPU-only, fast iteration

**Optional: 1-2 GPU Agents (If Available)**
- Use PPO/DQN for comparison benchmark
- Not required for V5/V6 success
- Can add later if ensemble proves insufficient

**Training Data Source:**
- Run V3 simulation with profitable agents (MarketMaker, PairsTrading)
- Extract state-action pairs from winning agents
- Label with forward-looking returns (imitation learning)

**Feature Engineering (V5.2):**
```python
features = [
    price_change_1tick, price_change_5tick, price_change_20tick,
    rsi_14, macd, macd_signal, bollinger_upper, bollinger_lower,
    current_position, cash_available, unrealized_pnl,
    order_book_imbalance, bid_ask_spread,
    momentum_score, value_score, volatility_score,
    news_sentiment (if recent event)
]
```

**Reward Function (V6.1):**
```python
reward = realized_pnl
         + 0.1 * unrealized_pnl_change
         - 0.01 * abs(position) * volatility  # Risk penalty
         - 0.001 * abs(action)  # Transaction cost
```

**Does Non-NN ML Work for Trading?**
- ✅ **Yes, historically successful:** Many quant funds use ensemble trees (XGBoost, LightGBM)
- ✅ **Interpretable:** Feature importance reveals which indicators matter
- ✅ **Fast:** Retrains in minutes, not hours
- ⚠️ **Limitation:** Requires manual feature engineering (no automatic representation learning like NNs)

**Trade-off vs Deep RL:**
| Aspect | Ensemble ML | Deep RL (PPO/DQN) |
|--------|-------------|-------------------|
| Training time | Minutes (CPU) | Hours (GPU) |
| Interpretability | High (feature importance) | Low (black box) |
| Hyperparameter tuning | Moderate | Extreme (learning rate, architecture, etc.) |
| Sample efficiency | High (learns from 1000 episodes) | Low (needs 100k+ steps) |
| State space | Requires features | Can learn from raw observations |

**Recommendation:** Start with Ensemble ML. If it beats noise traders, ship it. If not, debug features before trying Deep RL.

---



## V4-V7 Design Decisions Summary

### Decoupling Strategy

**V3.9 completes shared infrastructure.** After V3.9:
- RL path extends: `crates/gym/`, PyO3 bindings, Python scripts — **no simulation changes**
- Game path extends: `services/`, frontend, APIs — **no simulation changes**
- **Zero file conflicts** — paths are entirely independent

### Key Design Choices

| Decision | Rationale |
|----------|-----------|
| **Ensemble ML over Deep RL** | CPU-only, fast iteration, interpretable, proven in quant finance |
| **Optional Deep RL (V6.2)** | Only if ensemble plateaus; requires GPU (1-2 agents max) |
| **Idle game model** | Human is portfolio manager, not tick trader; adjusts formulas, uses VWAP execution |
| **Formula builder as core mechanic** | Human combines metrics (RSI, momentum, etc.) with adjustable weights |
| **Time controls required** | Pause/step/speed control essential for analysis |
| **Single-symbol MVP** | Simplifies initial gameplay; multi-symbol as advanced mode |
| **No save/resume in MVP** | Session-based gameplay (like idle games); can add later if needed |
| **No chatbot in MVP** | Direct UI controls sufficient; defer to V8 or post-launch |

### Technical Architecture Decisions

| Component | Decision | Why |
|-----------|----------|-----|
| **Storage (V3.9)** | Minimal: trades, candles, snapshots only | Both paths read from same data; avoid path-specific features |
| **ML Inference** | JSON export (trees/models) → Rust native | No ONNX needed for ensemble ML; only for optional Deep RL |
| **Services** | 3 services (Data, Game, Storage) | Removed Chatbot from MVP; Game Service is BFF |
| **Frontend** | React/TypeScript | Standard web stack; reuses existing TUI time control architecture |

### Validation Strategy

**RL path validates simulation realism:**
- If ML agents can profit → simulation is realistic → safe to build Game path
- If ML agents fail → debug features/simulation before Game investment
- Feature importance reveals which indicators matter → informs Game dashboard priorities

**Game path validates UX:**
- Formula builder tests if "portfolio manager" gameplay is engaging
- Leaderboard tests if Sharpe ratio competition motivates players
- Time controls test if speed control enables comprehension

### Dependency Graph

```
V3.9 (Storage) ← Last common ancestor
    │
    ├─► V4 (Web Frontend)
    │    │
    │    └─► Produces: Data visualization dashboard
    │
    ├─► V5 (Feature Engineering ML)
    │    │
    │    └─► Produces: PyO3 bindings, feature extraction, ensemble agents
    │
    ├─► V6 (Reinforcement Learning)
    │    │
    │    └─► Produces: Trained agents (EnsembleAgent, optionally NeuralAgent)
    │
    └─► V7 (Portfolio Manager Game) [requires V4]
         │
         └─► Requires: Agents to compete against (can use existing Tier 1, better with V5/V6 agents)

V8: Integration (if V5/V6 + V7 complete)
    └─► RL agents populate Game as intelligent opponents
```

**Recommendation:** V4 should be completed first as foundation for V7. V5/V6 can proceed independently and enhance V7 experience later.

---

## V4: Web Frontend

**Philosophy:** Build rich data visualization with Axum backend and React frontend. This provides the foundation for V7 game features. Start with Landing + Config pages to establish frontend infrastructure, then add simulation dashboard.

**Development Priority:**
1. **Phase 1 (V4.1):** Landing & Config Pages — React/Vite setup, preset management, config form
2. **Phase 2 (V4.2):** Services Foundation — Axum async services, channel bridge
3. **Phase 3 (V4.3):** Data Service — REST APIs for analytics, portfolio, risk
4. **Phase 4 (V4.4):** Simulation Dashboard — Real-time visualization, agent explorer

**Why Config-First:**
- Establishes frontend tooling (Vite, React Router, TypeScript) before complex visualizations
- Validates backend API patterns with simple CRUD (presets) before real-time data
- Allows simulation parameter tweaking from browser immediately
- Lower risk: if WebSocket/charting proves difficult, still have functional config UI

**Service Architecture:** 2 services initially (Data + Storage), Game Service added in V7

```
V4: Web Frontend (Weeks 1-4)
┌─────────────────────────────────────────────────────────────┐
│                      SIMULATION                             │
│  (sync, computes everything for agents)                     │
│  - Matching engine, agent tick loop                         │
│  - Emits: TickEvent { prices, trades, portfolios }          │
└──────────────────────────┬──────────────────────────────────┘
                           │ broadcast
                ┌──────────┴──────────┐
                ▼                     ▼
         ┌─────────┐           ┌─────────┐
         │  DATA   │           │ STORAGE │
         │ SERVICE │           │ SERVICE │
         │  :8001  │           │  :8003  │
         │         │           │         │
         │/api/config│         │/storage/*│
         │/api/presets│        │ Queries  │
         │/analytics│           │ History  │
         │/portfolio│           │          │
         │  /risk/* │           │          │
         │  /news/* │           │          │
         │WebSocket│           │          │
         └─────────┘           └─────────┘
              │
              ▼
      ┌──────────────┐
      │   FRONTEND   │
      │ React/TS UI  │
      │  3 Screens:  │
      │  - Landing   │
      │  - Config    │
      │  - Simulation│
      └──────────────┘

V7: Add Game Service (after V4)
         ┌─────────┐  ┌─────────┐
         │  DATA   │  │  GAME   │  (V7)
         │ SERVICE │  │ SERVICE │
         │  :8001  │  │  :8002  │
         │         │  │         │
         │         │  │ Formula │
         │         │  │ Builder │
         │         │  │  VWAP   │
         └─────────┘  └─────────┘
```

#### V4 Implementation Details

```
V4.1: Landing & Config Pages (~1 wk)
    └─► React/Vite/TypeScript scaffold
        - Vite for fast HMR and build
        - React Router: / (Landing), /config (Config), /sim (Simulation placeholder)
        - Tailwind CSS for styling (desktop-only, min-width: 1024px)
    └─► Landing Page:
        - Hero: "Quant Trading Gym" title + tagline
        - "Quick Start" button → loads default preset, navigates to /sim
        - "Configure" button → navigates to /config
        - Brief feature list (3-4 bullets)
    └─► Config Page:
        - Preset selector dropdown (built-in: Default, Demo, Stress Test, Low Activity, 
          High Volatility, Quant Heavy + custom presets from SQLite)
        - Accordion sections mirroring SimConfig:
          • Simulation Control: total_ticks, tick_delay_ms, max_cpu_percent, events_enabled
          • Symbols: editable list (name, initial_price, sector dropdown)
          • Tier 1 Agents: num_market_makers, num_noise_traders, num_momentum_traders, etc.
          • Tier 2 Agents: num_tier2_agents, t2_initial_cash, thresholds (collapsed by default)
          • Tier 3 Pool: enable_background_pool, background_pool_size, regime (collapsed)
          • Market Maker Params: mm_initial_cash, mm_half_spread, etc. (collapsed)
          • Noise Trader Params: nt_initial_cash, nt_order_probability, etc. (collapsed)
        - "Save Preset" button: name input, saves to SQLite via POST /api/presets
        - "Run Simulation" button: POST /api/config → navigates to /sim
    └─► Axum endpoints (minimal for V4.1):
        - GET /api/presets → list preset names (built-in + custom)
        - GET /api/presets/:name → full SimConfig JSON
        - POST /api/presets → save custom preset to SQLite
        - POST /api/config → accepts SimConfig, prepares simulation (V4.2 wires to runner)
        - Serves React build at / (SPA fallback)
    └─► SQLite preset storage:
        - Table: presets (name TEXT PRIMARY KEY, config_json TEXT, is_builtin BOOL)
        - Seed built-in presets from SimConfig::default(), ::demo(), etc.
    └─► No form validation in V4.1 (defer to V4.4 polish)

V4.2: Services Foundation (~1 wk)
    └─► Axum async services base
    └─► Channel bridge: Simulation (sync) ↔ Services (async)
    └─► Error handling, logging
    └─► Health check endpoints
    └─► WebSocket infrastructure for real-time updates

V4.3: Data Service (~1 wk)
    └─► /analytics/candles (OHLCV from V3.9 storage)
    └─► /analytics/indicators (SMA, RSI, MACD, Bollinger, ATR)
    └─► /analytics/factors (momentum score, value score, volatility)
    └─► /portfolio/agents (list all agents with P&L summary)
    └─► /portfolio/:agent_id (detailed positions, equity curve)
    └─► /risk/:agent_id (VaR, Sharpe, Sortino, max drawdown)
    └─► /news/active (current events with sentiment)
    └─► WebSocket /stream (real-time tick updates)
    └─► All endpoints query V3.9 storage + live simulation state

V4.4: Simulation Dashboard (~1 wk)
    └─► WebSocket connection to Data Service
    └─► Real-time data visualization:
        - Price chart (candlestick + line modes)
        - Order book depth heatmap
        - Indicator panel (multi-chart with all indicators)
        - Factor gauges (momentum, value, volatility)
        - Risk dashboard (VaR, Sharpe, drawdown)
        - News feed with sentiment tags
    └─► Agent Explorer:
        - Sortable table: all agents with P&L, Sharpe, positions
        - Click agent → detailed view (equity curve, trade history)
    └─► Time Controls UI:
        - Pause/Play toggle
        - Speed slider (1x, 10x, 100x, unlimited)
        - Step button (single tick advance)
    └─► Read-only mode: No human interaction yet
    └─► Form validation polish for Config page
```

**V4 Deliverable:** Portfolio-worthy demo showing 100k agents trading with full quant dashboard. Can showcase to stakeholders or use for analysis.

---

## V5: Feature Engineering ML

**Philosophy:** Build PyO3 bindings and feature engineering infrastructure for traditional ML training.

```
V5.1: Gym Environment (~5 days)
    └─► TradingEnv with step/reset (Rust)
    └─► Episode management, termination conditions
    └─► Logging infrastructure for training curves

V5.2: Feature Engineering (~5 days)
    └─► Extract 20-30 features from StrategyContext
    └─► Price indicators: SMA, EMA, RSI, MACD, Bollinger, ATR
    └─► Portfolio state: position, cash, unrealized P&L, equity
    └─► Market microstructure: bid-ask spread, order book imbalance
    └─► Fundamental: news sentiment, value score
    └─► Observation parity: human sees same data

V5.3: PyO3 Bindings (~3 days)
    └─► Python can call Rust TradingEnv
    └─► NumPy array interface for features
    └─► Batched episode collection (parallel Rust envs)

V5.4: Ensemble Training (~5 days)
    └─► Collect training data from profitable agents (imitation learning)
    └─► Train Random Forest (scikit-learn)
    └─► Train Linear Regression (trend following)
    └─► Train SVM (regime detection)
    └─► Implement weighted voting ensemble
    └─► Validate on held-out episodes

V5.5: Rust Inference (~2 days)
    └─► Export ensemble to JSON (decision trees, linear weights, SVM params)
    └─► Implement `EnsembleAgent` in Rust (crates/agents/ml_agents/)
    └─► No ONNX needed — interpret trees/models directly
    └─► Add to simulation as Tier 1 agent
```

**Total:** ~2-3 weeks for ensemble ML approach

---

## V6: Reinforcement Learning

**Philosophy:** Build on V5 infrastructure to add reward functions and deep RL training. Add Deep RL (GPU) optionally if ensemble proves insufficient.

```
V6.1: Reward Function (~2 days)
    └─► Realized P&L + unrealized P&L change
    └─► Risk penalties: volatility, drawdown
    └─► Transaction cost modeling
    └─► Sharpe ratio terminal bonus
```

**Optional V6.2: Deep RL with Neural Networks (~2 weeks, requires GPU)**

Only pursue if ensemble ML plateaus below profitability threshold. Requires 1-2 GPUs (per your constraint).

```
V6.2.1: Neural Network Architecture (~3 days)
    └─► Design network: feedforward or LSTM for sequence modeling
    └─► Input: V5.2 features (reuse feature engineering)
    └─► Output: discrete actions (buy/hold/sell) or continuous (position sizing)
    └─► Framework: PyTorch or stable-baselines3

V6.2.2: Deep RL Training (~1 week)
    └─► Algorithm: PPO (stable, sample-efficient) or DQN (simpler, discrete actions)
    └─► Hyperparameter tuning: learning rate, batch size, entropy coefficient
    └─► Parallel environment collection (leverage Rust speed)
    └─► Tensorboard logging, training curves

V6.2.3: ONNX Export + Rust Inference (~3 days)
    └─► Export trained model to ONNX format
    └─► Integrate ONNX runtime in Rust (tract or ort crate)
    └─► Implement `NeuralAgent` wrapper
    └─► Benchmark inference latency (target: <1ms per agent)
```

**Why Start with Ensemble ML:**
1. **Faster iteration:** Minutes to train vs hours for Deep RL
2. **No GPU required:** Runs on any machine
3. **Interpretable:** Feature importance reveals what works
4. **Proven:** Many quant funds use tree ensembles
5. **Baseline:** If ensemble fails, features are bad (fix before trying Deep RL)

**When to Add Deep RL:**
- Ensemble ML consistently loses to baseline strategies
- Need to model sequential dependencies (LSTM for market regimes)
- Have GPU resources available for training
- Want to explore complex action spaces (continuous position sizing)

**Maps to Original:** Phases 13-18 (RL Track) — updated for CPU-first approach with optional GPU path


## V7: Portfolio Manager Game

**Philosophy:** Build on V4 frontend to add interactive game mechanics. Human becomes portfolio manager competing against AI agents.

```
V7.1: Game Service (~1.5 wks)
    └─► Formula Builder API:
        - POST /game/formula (parse: "0.4*RSI + 0.3*momentum - 0.2*volatility")
        - Validate formula (safe eval, whitelist metrics only)
        - Real-time signal calculation (every tick or rebalance interval)
        - Returns: signal strength (-1 to +1)
    └─► VWAP Execution Tool:
        - POST /game/vwap (symbol, target_qty, max_ticks)
        - Spawns VwapExecutor agent on behalf of human
        - Returns execution progress updates via WebSocket
    └─► Human Agent Management:
        - Create human agent with starting capital ($100k)
        - Track human P&L separately from AI agents
        - Submit orders generated by formula
    └─► Session management (start/stop simulation)
    └─► Leaderboard persistence (SQLite: player_id, sharpe_ratio, timestamp)

V7.2: Interactive Frontend (~0.5 wk)
    └─► Formula Builder UI:
        - Metric selector (dropdown: RSI, MACD, momentum, etc.)
        - Weight sliders for each metric
        - Live signal preview (bar: Strong Sell ← → Strong Buy)
        - "Apply Formula" button → updates on backend
    └─► VWAP Tool UI:
        - Symbol dropdown, quantity input
        - "Execute VWAP" button
        - Progress notification (toast: "Executed 450/1000 shares")
    └─► Human Portfolio Panel:
        - Highlight human agent in agent table
        - Show human equity curve vs AI benchmarks (50th/75th percentile)
    └─► Leaderboard Modal:
        - Top 10 players by Sharpe ratio
        - Human's rank highlighted
```

**V7.2 Deliverable:** Playable game where human can compete against AI agents using formula-based strategies.

#### V7.3: Polish (~1 week)

```
V7.3: Competitive Features
    └─► Challenge modes:
        - "Beat MarketMaker" (fixed 10k ticks)
        - "Beat Momentum" (trending market)
        - "Beat PairsTrading" (multi-symbol)
    └─► Sandbox mode (unlimited time, no pressure)
    └─► Tutorial mode (guided formula building)
    └─► Advanced time controls:
        - "Skip to next event" button
        - "Run until market regime change"
    └─► Export functionality:
        - Download equity curve CSV
        - Export formula as JSON
```

**Total V7:** ~3 weeks (V7.1: 1.5 weeks, V7.2: 0.5 week, V7.3: 1 week)

#### Deployment Architecture (Docker-Based)

**Frontend + Backend Integration:**

```
Browser
   │
   └──► http://localhost:8001
         │
         ├─► /            → Serves React build (index.html, bundle.js)
         ├─► /api/analytics/*  → REST endpoints
         ├─► /api/portfolio/*  → REST endpoints
         ├─► /api/risk/*       → REST endpoints
         ├─► /ws               → WebSocket for real-time updates
         │
         └─► Data Service (Axum) :8001
                  │
                  ├──► V3.9 Storage (SQLite)
                  └──► Simulation (channel bridge)
```

**Dockerfile (Multi-Stage Build):**
```dockerfile
# Stage 1: Build TypeScript/React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build
# Output: /frontend/dist (index.html, bundle.js, etc.)

# Stage 2: Build Rust backend
FROM rust:1.75 AS backend-builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin data-service

# Stage 3: Runtime (distroless)
FROM gcr.io/distroless/cc-debian12
COPY --from=backend-builder /app/target/release/data-service /
COPY --from=frontend-builder /frontend/dist /frontend/dist
EXPOSE 8001
CMD ["/data-service", "--frontend-path", "/frontend/dist"]
```

**Axum Route Setup:**
```rust
// services/data/src/main.rs
use axum::{Router, routing::get};
use tower_http::services::ServeDir;

let app = Router::new()
    // API routes
    .route("/api/analytics/candles", get(get_candles))
    .route("/api/portfolio/agents", get(list_agents))
    .route("/ws", get(websocket_handler))
    // Serve React app (must be last)
    .nest_service("/", ServeDir::new("/frontend/dist"))
    .fallback_service(ServeFile::new("/frontend/dist/index.html"));
    //                 ↑ SPA fallback (for client-side routing)
```

**How It Works:**
1. **TypeScript/React** code lives in `frontend/` directory
2. **Docker build** compiles TS → JS bundle (no local Node needed)
3. **Axum** serves both:
   - Static files (React app) at `/`
   - REST APIs at `/api/*`
   - WebSocket at `/ws`
4. **Browser** loads from single origin `http://localhost:8001`

**Benefits:**
- **Single port (8001)** - no CORS issues, frontend and API same origin
- **No local Node install** - all TS compilation happens in Docker
- **Simple deployment** - one container for Phase 1
- **Development workflow:**
  - Frontend: `cd frontend && npm run dev` (hot reload, proxies API calls to :8001)
  - Backend: `cargo run --bin data-service` (serves API + pre-built frontend)

**Phase 2: Add Game Service Container**
```yaml
# docker-compose.yaml
services:
  data-service:
    build: ./services/data
    ports:
      - "8001:8001"
    volumes:
      - ./data:/data

  game-service:  # Phase 2
    build: ./services/game
    ports:
      - "8002:8002"
```

Browser would then call `:8001` for viz, `:8002` for game actions (formula, VWAP).

#### Time Control System (Required)

Human players need control over simulation speed to analyze markets and adjust strategies.

| Mode | Speed | Use Case | Implementation |
|------|-------|----------|----------------|
| **Paused** | 0 ticks/sec | Formula adjustments, analysis | Simulation blocks, UI remains responsive |
| **Step** | Manual advance | Learning, debugging | Single tick on button press |
| **Slow** | 1 tick/sec | Comfortable analysis | Good for watching indicators evolve |
| **Normal** | 10 tick/sec | Standard gameplay | Balanced between speed and comprehension |
| **Fast** | 100 tick/sec | Skip boring periods | When waiting for market regime change |
| **Ultra** | Unlimited | Backtesting mode | Run to completion, review results |

**Implementation:** Game Service controls simulation tick rate via channel commands (existing TUI architecture reused).

#### Dashboard Panels (Information Parity with AI Agents)

| Panel | Shows | Source (via BFF) |
|-------|-------|------------------|
| Indicator Panel | SMA, EMA, RSI, MACD, Bollinger, ATR | Data :8001 → /analytics/* |
| Factor Gauges | Momentum, Value, Volatility scores | Data :8001 → /analytics/* |
| Risk Dashboard | VaR, Sharpe, max drawdown | Data :8001 → /risk/* |
| Portfolio | Holdings, P&L, equity curve | Data :8001 → /portfolio/* |
| Signal Summary | Strong Buy → Strong Sell | Game :8002 → aggregated |
| News Feed | Active events with sentiment | Data :8001 → /news/* |

#### Game Design Decisions (Answers to Open Questions)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Formula application timing** | Instant (every tick) with optional rebalance interval | Flexible: aggressive (every tick) or conservative (every N ticks) |
| **Human visibility of AI positions** | ❌ Hidden | Fair competition; human sees only aggregated market data (order book, trades) |
| **Symbol management** | Single-symbol focus initially | Simplifies MVP; multi-symbol can be added as "Advanced Mode" |
| **Capital limits** | ✅ Start with $100k | Enforces risk management; leaderboard normalized by starting capital |
| **Rebalance interval** | Configurable (default: every 10 ticks) | Prevents overtrading; realistic execution delay |

**Competitive Modes:**
- **Sandbox:** Unlimited time, $100k starting capital, goal: beat 50th percentile AI
- **Challenge:** Fixed 10k ticks, beat specific AI strategy (MarketMaker, Momentum, PairsTrading)
- **Leaderboard:** Persistent top 10 by risk-adjusted returns (Sharpe ratio) over 100k ticks

**Maps to Original:** Phases 13-22 (Game Track) + Part 11 (Human Player Interface) — refined for portfolio manager gameplay

#### Containerization (V4/V7)

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

## V8: Full Integration

If you completed V5/V6 (ML/RL) AND V7 (Game):
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
| V5 | PyO3 GIL blocking | `py.allow_threads()` for Rust computation |
| V4/V7 | Async/sync boundary | Channel-based `SimulationBridge` |

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
| V3 | Scaling & Persistence | ✅ Complete |
| V4 | Web Frontend | ✅ Complete |
| V5 | Feature Engineering ML | 🔲 Planned |
| V6 | Reinforcement Learning | 🔲 Planned |
| V7 | Portfolio Manager Game | 🔲 Planned |

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
| ONNX inference | V6 | Requires full gym first |
| HTTP services | V4 | Requires stable core first |
| React frontend | V4 | TUI is enough for learning |

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
| **RL Agent** | V6 | 🔲 | Requires gym + ONNX |

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
| V3 | "100k agents without OOM; trades persist across runs" | ✅ Achieved |
| V4 | "I can see rich visualization of simulation data in browser" | 🔲 Planned |
| V5 | "I have PyO3 bindings and feature engineering for ML" | 🔲 Planned |
| V6 | "I trained an RL agent that beats noise traders" | 🔲 Planned |
| V7 | "I can play, pause, analyze, and make informed trades" | 🔲 Planned |

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


## V3.x Migration Notes
- **V3.1:** Refactor `state.rs` for multi-symbol positions; update trait in `traits.rs`
- **V3.2:** Add `tiers.rs`, `tier2/` module with `agent.rs`, `wake_index.rs`, `strategies.rs`; add `orchestrator.rs` to simulation
- **V3.3:** Add `tier1/strategies/pairs_trading.rs`, `tier2/strategies/sector_rotator.rs`, extend `quant/stats.rs`
- **V3.4:** Add `tier3/` module with `pool.rs`
- **V3.5:** Performance tuning pass (no new files)
- **V3.6:** Add `hooks.rs` to simulation; refactor TUI to implement `SimulationHook`
- **V3.7:** Add `dockerfile/`, `docker-compose.yaml`, CI workflow
- **V3.8:** Add `storage/` crate
