# Copilot Instructions for quant-trading-gym

This document outlines the commands that must be run to verify code quality before committing changes.

## Prerequisites

- **Docker** and **Docker Compose** are required for all checks
- **Cargo** (Rust toolchain) for backend checks
- No Node.js installation needed - frontend checks run in Docker

## Backend Checks (Rust)

### Clippy (Linting)
```bash
cargo clippy --all-targets --all-features -- -D warnings
```
Run this to catch common Rust mistakes and enforce code quality.

### Format Check
```bash
cargo fmt --all -- --check
```
Run this to verify code formatting. To auto-fix formatting:
```bash
cargo fmt --all
```

### Tests
```bash
cargo test
```

## Frontend Checks (Docker-based)

All frontend checks run via Docker Compose using `docker-compose.frontend.yaml`.

### TypeScript Type Check
```bash
docker compose -f docker-compose.frontend.yaml run --rm typecheck
```
Runs `tsc --noEmit` to verify TypeScript types without producing output.

### ESLint
```bash
docker compose -f docker-compose.frontend.yaml run --rm lint
```
Runs ESLint on the frontend codebase.

### Integration Tests
```bash
docker compose -f docker-compose.frontend.yaml build integration-test
docker compose -f docker-compose.frontend.yaml run --rm integration-test
```
**Important**: The backend server must be running at `localhost:8001` for integration tests to pass.

To start the backend server:
```bash
cargo run --release
```
Or use Docker:
```bash
docker compose up server
```

## Quick Reference - All Checks

### Before committing, run:
```bash
# Backend
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

# Frontend (with backend running)
docker compose -f docker-compose.frontend.yaml run --rm typecheck
docker compose -f docker-compose.frontend.yaml run --rm lint
docker compose -f docker-compose.frontend.yaml run --rm integration-test
```

## Project Structure

- **Backend**: Rust/Axum server in `src/` and `crates/`
- **Frontend**: React/Vite/TypeScript in `frontend/`
- **Docker configs**: `dockerfile/` directory
- **Compose files**: 
  - `docker-compose.yaml` - Main stack
  - `docker-compose.frontend.yaml` - Frontend dev/test services
  - `docker-compose.tui.yaml` - TUI interface

## Integration Test Details

The integration tests in `frontend/src/*.integration.test.ts` verify the API contract between frontend and backend:

- **API tests** (`api.integration.test.ts`): Tests all REST endpoints
- **WebSocket tests** (`websocket.integration.test.ts`): Tests real-time tick data

Tests are designed to pass even when the simulation is paused (WebSocket tests will timeout gracefully).
