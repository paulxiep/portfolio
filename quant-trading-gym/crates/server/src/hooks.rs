//! SimulationHook implementations for broadcasting updates (V4.2).
//!
//! Provides the bridge from sync simulation to async server via hooks.
//!
//! # Architecture
//!
//! ```text
//! Simulation (sync)          BroadcastHook           Server (async)
//!       │                         │                       │
//!       │── on_tick_end() ───────▶│                       │
//!       │                         │── tick_tx.send() ────▶│
//!       │                         │                       │── ws broadcast
//! ```
//!
//! # Design Principles
//!
//! - **Declarative**: Hook declares what events it handles
//! - **Modular**: Hook is self-contained, no dependencies on server internals
//! - **SoC**: Hook observes simulation, server distributes updates

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use simulation::{HookContext, SimulationHook, SimulationStats};
use tokio::sync::broadcast;

use crate::bridge::{SymbolData, TickData};

/// Hook that broadcasts simulation updates to the server.
///
/// Implements SimulationHook to observe tick events and broadcast
/// via tokio broadcast channel to async WebSocket handlers.
pub struct BroadcastHook {
    /// Broadcast sender for tick data.
    tick_tx: broadcast::Sender<TickData>,
    /// Total agent count (set once at start).
    total_agents: AtomicU64,
    /// Whether simulation is running.
    running: AtomicBool,
    /// Whether simulation has finished.
    finished: AtomicBool,
}

impl BroadcastHook {
    /// Create a new broadcast hook.
    pub fn new(tick_tx: broadcast::Sender<TickData>) -> Self {
        Self {
            tick_tx,
            total_agents: AtomicU64::new(0),
            running: AtomicBool::new(false),
            finished: AtomicBool::new(false),
        }
    }

    /// Set total agent count.
    pub fn set_agents(&self, count: u64) {
        self.total_agents.store(count, Ordering::Relaxed);
    }

    /// Set running state.
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// Set finished state.
    pub fn set_finished(&self, finished: bool) {
        self.finished.store(finished, Ordering::Relaxed);
    }

    /// Get sender for server state.
    pub fn sender(&self) -> broadcast::Sender<TickData> {
        self.tick_tx.clone()
    }

    /// Build TickData from hook context and stats.
    fn build_tick_data(&self, stats: &SimulationStats, ctx: &HookContext) -> TickData {
        let mut symbols = HashMap::new();

        // Build per-symbol data from market snapshot
        for (symbol, book) in &ctx.market.books {
            let data = SymbolData::new(symbol.clone())
                .with_prices(book.last_price, book.best_bid, book.best_ask)
                .with_depth(book.bid_depth.0, book.ask_depth.0);
            symbols.insert(symbol.clone(), data);
        }

        TickData {
            tick: ctx.tick,
            timestamp: ctx.timestamp,
            symbols,
            trades_this_tick: 0, // TODO: track from on_trades
            total_trades: stats.total_trades,
            total_orders: stats.total_orders,
            agents_called: stats.agents_called_this_tick,
        }
    }
}

impl SimulationHook for BroadcastHook {
    fn name(&self) -> &str {
        "BroadcastHook"
    }

    fn on_tick_end(&self, stats: &SimulationStats, ctx: &HookContext) {
        let tick_data = self.build_tick_data(stats, ctx);

        // Fire-and-forget: if no receivers, drop the message
        let _ = self.tick_tx.send(tick_data);
    }

    fn on_simulation_end(&self, _final_stats: &SimulationStats) {
        self.set_finished(true);
        self.set_running(false);
    }
}

// Required for Arc<dyn SimulationHook>
unsafe impl Send for BroadcastHook {}
unsafe impl Sync for BroadcastHook {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_hook_creation() {
        let (tx, _rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        assert_eq!(hook.name(), "BroadcastHook");
    }

    #[test]
    fn test_broadcast_hook_state() {
        let (tx, _rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        hook.set_agents(25000);
        hook.set_running(true);

        assert_eq!(hook.total_agents.load(Ordering::Relaxed), 25000);
        assert!(hook.running.load(Ordering::Relaxed));
    }

    #[test]
    fn test_tick_data_broadcast() {
        let (tx, mut rx) = broadcast::channel(16);
        let hook = BroadcastHook::new(tx);

        // Create minimal context
        let ctx = HookContext::new(100, 1000);
        let stats = SimulationStats {
            tick: 100,
            total_trades: 500,
            total_orders: 1000,
            ..Default::default()
        };

        hook.on_tick_end(&stats, &ctx);

        // Verify message was sent
        let received = rx.try_recv().unwrap();
        assert_eq!(received.tick, 100);
        assert_eq!(received.total_trades, 500);
    }
}
