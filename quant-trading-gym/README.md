# Quant Trading Gym

## Project Overview

Quant Trading Gym is a high-performance quantitative trading simulation platform built primarily in Rust. It is designed to support large-scale simulations with over 100,000 agents using a tiered architecture. The system integrates Reinforcement Learning (RL) training capabilities, modular quantitative strategies, risk management, and a microservices architecture.

## Development Roadmap

The project is built vertically in iterative stages, ensuring a runnable simulation at each step.

| Version | Focus | Goal |
|---------|-------|------|
| **V0** | **The MVP Simulation** | Single-threaded simulation with TUI visualization showing agents trading. |
| **V1** | **Quant Layer** | Add indicators (SMA, RSI), risk metrics (Sharpe, Drawdown), and real strategies. |
| **V2** | **Agent Scaling** | Implement tiered architecture (Smart, Reactive, Background) for 100k+ agent scale. |
| **V3** | **Persistence & Events** | Add SQLite storage for trade history and news event generation. |
| **V4** | **RL or Game Track** | **RL:** Gym env, PyO3 bindings, training. <br> **Game:** Services, API, Web Frontend. |
| **V5** | **Full Integration** | RL agents as opponents in the multiplayer game. |

## Key Features

- **High Performance:** Built with Rust for core simulation and services, ensuring low latency and high throughput.
- **Scalability:** Supports 100,000+ agents through a tiered architecture:
  - **Tier 1 (Smart):** Full strategy agents (e.g., RL agents, complex quant strategies).
  - **Tier 2 (Reactive):** Event-triggered agents with lightweight logic.
  - **Tier 3 (Background):** Statistically modeled background liquidity pool.
- **RL Integration:** Python bindings for training scripts and experiments.
- **Financial Precision:** Uses fixed-point arithmetic (`i64`) for all monetary values to ensure accuracy.
- **Modular Design:** Strategies, observations, and rewards are implemented as plugins.
- **Microservices:** Async services for non-critical paths, bridged to the synchronous simulation core.

## Architecture

The project follows a strict separation of concerns where crates communicate through traits. It is designed to fit within a 2GB memory budget while maintaining realistic latency and order execution.

## Tech Stack

- **Rust:** Core simulation, types, quant strategies, and services.
- **Python:** Training scripts and experiments (via PyO3).
- **TypeScript:** Frontend interface.

