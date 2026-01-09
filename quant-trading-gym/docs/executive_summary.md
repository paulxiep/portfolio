# Quant Trading Gym — Executive Summary

## What It Is

A high-performance trading simulation that models realistic market microstructure with 1,000 autonomous agents trading across 4 symbols. Built in Rust for speed and reliability.

## Why It Matters

- **Strategy Testing**: Safe environment to test trading strategies before deploying real capital
- **Market Research**: Study price discovery, liquidity dynamics, and agent behavior
- **ML/RL Training**: Train reinforcement learning models against realistic market conditions

## Key Features

- **Multi-Symbol Trading**: 4 symbols across different sectors (Consumer, Energy, Real Estate, Finance)
- **11 Agent Strategies**: Market makers, momentum, MACD, Bollinger, VWAP, pairs trading, sector rotation
- **Tiered Architecture**: T1 (full logic), T2 (reactive), T3 (45k+ background agents)
- **News Events**: Earnings surprises, guidance changes, rate decisions, sector news
- **Fair Value Model**: Gordon Growth Model anchors prices to fundamentals
- **Real-time Visualization**: Terminal UI with price charts, order book, agent P&L

## Quick Start

```bash
cargo build --release
cargo run --release    # Press Space to start
cargo run --release -- --headless --ticks 10000  # Benchmark mode
docker compose up      # Containerized run (port 7681)
```

## Current State

**V3 Complete** — 100k+ agent scaling with tiered architecture, SQLite persistence, batch auction matching, and Docker containerization.

## Technology

- **Language**: Rust (100% safe, no external dependencies for core)
- **Precision**: Fixed-point arithmetic for financial accuracy
- **Architecture**: Modular crates (types, sim-core, agents, news, quant, simulation, tui, storage)
- **Parallelism**: Rayon-based parallel agent execution with batch auction
- **Persistence**: SQLite for trade history, candle aggregation, portfolio snapshots
