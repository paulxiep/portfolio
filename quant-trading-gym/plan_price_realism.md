# Plan: Price Realism - Fair Value Drift

## Problem

Currently, prices are extremely flat between news events because:
1. Fair value is deterministic (Gordon Growth Model with known parameters)
2. All agents estimate similar fair values
3. No uncertainty about fundamentals between events

Result: Unrealistic "flat line with occasional jumps" price patterns.

## Solution

Add a bounded random walk to fair value, simulating continuous uncertainty about fundamentals.

## Design

### Approach
Drift the **output** fair value directly, not the Gordon Growth inputs (EPS, g, r). This keeps the model internally consistent while adding realistic uncertainty.

```rust
// Each tick, apply small drift to fair value
let drift = rng.gen_range(-drift_pct..drift_pct);
fair_value = (fair_value * (1.0 + drift))
    .max(min_fair_value)
    .min(max_fair_value);
```

### Safeguards

| Risk | Mitigation |
|------|------------|
| Goes to 0 or negative | Floor at 10% of initial fair value |
| Explodes to infinity | Cap at 10x initial fair value |
| Unrealistic volatility | Default drift ~0.1% per tick (configurable) |
| Breaks existing tests | Make drift optional, default off for tests |

### Configuration

Add to `SimConfig` or symbol fundamentals:
```rust
pub struct FairValueDriftConfig {
    /// Enable fair value drift (default: true for sim, false for tests)
    pub enabled: bool,
    /// Max drift per tick as fraction (e.g., 0.001 = 0.1%)
    pub drift_pct: f64,
    /// Floor as fraction of initial (e.g., 0.1 = 10%)
    pub min_pct: f64,
    /// Cap as multiple of initial (e.g., 10.0 = 10x)
    pub max_multiple: f64,
}
```

## Files to Modify

| File | Change |
|------|--------|
| `crates/news/src/fundamentals.rs` | Add drift logic to `FairValueModel` |
| `crates/types/src/config.rs` | Add `FairValueDriftConfig` |
| `crates/simulation/src/runner.rs` | Apply drift each tick before agent decisions |

## Implementation Steps

1. Add `FairValueDriftConfig` to types
2. Store initial fair values per symbol (for bounds calculation)
3. Add `apply_drift()` method to `FairValueModel`
4. Call drift in simulation tick loop (before `collect_orders`)
5. Add config flag to disable for deterministic tests

## Verification

1. Run simulation, observe price chart - should show gradual wandering
2. Verify prices stay within bounds (never <10% initial, never >10x initial)
3. Run existing tests with drift disabled - should pass unchanged
4. Check that news events still cause sharp moves on top of drift

## Future Considerations

- Per-symbol drift rates (volatile vs stable stocks)
- Correlation between symbols (market-wide sentiment drift)
- Mean-reversion in drift (Ornstein-Uhlenbeck process)
- Configurable via SimConfig TOML/JSON
