# Quant Trading Gym — Technical Summary

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    simulation crate                      │
│  (tick loop, event processing, order routing)           │
├─────────────┬─────────────┬─────────────┬───────────────┤
│   agents    │  sim-core   │    news     │     quant     │
│  (traits,   │ (order book,│ (events,    │ (indicators,  │
│  strategies)│  matching)  │ fundamentals)│  risk metrics)│
├─────────────┴─────────────┴─────────────┴───────────────┤
│                      types crate                         │
│        (Order, Trade, Price, Symbol, Sector)            │
└─────────────────────────────────────────────────────────┘
│                       tui crate                          │
│            (visualization, user input)                   │
└─────────────────────────────────────────────────────────┘
```

## Crate Responsibilities

| Crate | Purpose | Key Files |
|-------|---------|-----------|
| `types` | Shared types, no logic | `order.rs`, `config.rs` |
| `sim-core` | Order book, matching engine | `order_book.rs`, `matching.rs` |
| `agents` | Agent trait + 7 strategies | `strategies/*.rs` |
| `news` | Events, fundamentals, sectors | `generator.rs`, `fundamentals.rs` |
| `quant` | Indicators, risk metrics | `indicators/`, `tracker.rs` |
| `simulation` | Tick loop coordination | `runner.rs` |
| `tui` | Terminal UI | `app.rs`, `widgets/` |

## Simulation Loop

```
for tick in 0..max_ticks:
    1. Generate news events (probabilistic)
    2. Apply permanent fundamentals changes (earnings, guidance, rates)
    3. Build StrategyContext for each agent
    4. Collect orders from agents (on_tick)
    5. Validate against position limits
    6. Match orders through engine
    7. Notify agents of fills (on_fill)
    8. Update candles, indicators, risk metrics
    9. Send SimUpdate to TUI
```

## Fair Value Model

Prices anchor to intrinsic value via Gordon Growth Model:

```
fair_value = D1 / (r - g)

D1 = EPS × payout × (1 + g)     # Next dividend
r  = risk_free + equity_premium  # Required return (9%)
g  = growth_estimate             # Growth rate (≤7%)
```

With defaults: $5 EPS, 40% payout, 5% growth → **$52.50 fair value**

## Agent Strategies

| Strategy | Signal | Behavior |
|----------|--------|----------|
| MarketMaker | Always | Two-sided quotes around mid |
| NoiseTrader | Random | Random buys/sells near fair value |
| Momentum | RSI < 30 / > 70 | Buy oversold, sell overbought |
| TrendFollower | SMA crossover | Buy golden cross, sell death cross |
| MacdCrossover | MACD/Signal | Buy bullish cross, sell bearish |
| BollingerReversion | Band touch | Buy lower band, sell upper band |
| VwapExecutor | Time-sliced | Execute target qty over horizon |

## Key Design Decisions

1. **Mean-reverting prices**: Realistic for tick-level liquid markets; momentum strategies struggle (as in real HFT)
2. **Fixed-point arithmetic**: `Price` and `Cash` use i64 with implicit decimals for financial precision
3. **Event-value-first generation**: Events generate magnitude before selecting symbol to prevent seed-based bias
4. **Growth cap at 7%**: Prevents Gordon Growth Model breakdown when g ≥ r

## Build & Run

```bash
cargo build --release      # Build
cargo test --all           # 213 tests
cargo run --release        # TUI (Space to start)
```

## TUI Controls

| Key | Action |
|-----|--------|
| `Space` | Start/Pause |
| `Tab`/`1-4` | Switch symbol |
| `O` | Overlay mode |
| `↑↓` | Scroll |
| `q` | Quit |
