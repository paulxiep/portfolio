//! Shared server state (V4.2).
//!
//! Contains channels and metrics shared across handlers.
//!
//! # Design Principles
//!
//! - **Declarative**: State is data, handlers extract what they need
//! - **Modular**: State independent of route logic
//! - **SoC**: State holds references, doesn't own simulation

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::broadcast;

use crate::bridge::{SimCommand, TickData};

/// Shared state for all route handlers.
///
/// Cloned into each handler via Axum's State extractor.
#[derive(Clone)]
pub struct ServerState {
    /// Broadcast channel for tick updates (simulation → clients).
    pub tick_tx: broadcast::Sender<TickData>,

    /// Command sender (server → simulation).
    pub cmd_tx: crossbeam_channel::Sender<SimCommand>,

    /// Server start time.
    pub start_time: Instant,

    /// Shared metrics.
    pub metrics: Arc<ServerMetrics>,
}

impl ServerState {
    /// Create new server state with channels.
    pub fn new(
        tick_tx: broadcast::Sender<TickData>,
        cmd_tx: crossbeam_channel::Sender<SimCommand>,
    ) -> Self {
        Self {
            tick_tx,
            cmd_tx,
            start_time: Instant::now(),
            metrics: Arc::new(ServerMetrics::new()),
        }
    }

    /// Get uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Subscribe to tick updates.
    pub fn subscribe_ticks(&self) -> broadcast::Receiver<TickData> {
        self.tick_tx.subscribe()
    }

    /// Send command to simulation.
    pub fn send_command(
        &self,
        cmd: SimCommand,
    ) -> Result<(), crossbeam_channel::SendError<SimCommand>> {
        self.cmd_tx.send(cmd)
    }
}

/// Server-side metrics.
pub struct ServerMetrics {
    /// Current tick from simulation.
    pub current_tick: AtomicU64,
    /// Total agents in simulation.
    pub total_agents: AtomicU64,
    /// Whether simulation is running.
    pub sim_running: AtomicBool,
    /// Whether simulation has finished.
    pub sim_finished: AtomicBool,
    /// Active WebSocket connections.
    pub ws_connections: AtomicU64,
}

impl ServerMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self {
            current_tick: AtomicU64::new(0),
            total_agents: AtomicU64::new(0),
            sim_running: AtomicBool::new(false),
            sim_finished: AtomicBool::new(false),
            ws_connections: AtomicU64::new(0),
        }
    }

    /// Update from simulation update.
    pub fn update_from_tick(&self, tick: u64, agents: u64, running: bool, finished: bool) {
        self.current_tick.store(tick, Ordering::Relaxed);
        self.total_agents.store(agents, Ordering::Relaxed);
        self.sim_running.store(running, Ordering::Relaxed);
        self.sim_finished.store(finished, Ordering::Relaxed);
    }

    /// Increment WebSocket connection count.
    pub fn ws_connect(&self) {
        self.ws_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement WebSocket connection count.
    pub fn ws_disconnect(&self) {
        self.ws_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current tick.
    pub fn tick(&self) -> u64 {
        self.current_tick.load(Ordering::Relaxed)
    }

    /// Get agent count.
    pub fn agents(&self) -> u64 {
        self.total_agents.load(Ordering::Relaxed)
    }

    /// Check if simulation is running.
    pub fn is_running(&self) -> bool {
        self.sim_running.load(Ordering::Relaxed)
    }

    /// Check if simulation has finished.
    pub fn is_finished(&self) -> bool {
        self.sim_finished.load(Ordering::Relaxed)
    }

    /// Get WebSocket connection count.
    pub fn ws_count(&self) -> u64 {
        self.ws_connections.load(Ordering::Relaxed)
    }
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_update() {
        let metrics = ServerMetrics::new();
        metrics.update_from_tick(100, 25000, true, false);

        assert_eq!(metrics.tick(), 100);
        assert_eq!(metrics.agents(), 25000);
        assert!(metrics.is_running());
        assert!(!metrics.is_finished());
    }

    #[test]
    fn test_ws_connections() {
        let metrics = ServerMetrics::new();
        assert_eq!(metrics.ws_count(), 0);

        metrics.ws_connect();
        metrics.ws_connect();
        assert_eq!(metrics.ws_count(), 2);

        metrics.ws_disconnect();
        assert_eq!(metrics.ws_count(), 1);
    }
}
