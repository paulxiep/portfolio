//! ML agent spawning: decision trees, random forests, gradient boosted.

use agents::{
    Agent, DecisionTree, GradientBoosted, MlModel, RandomForest, TreeAgent, TreeAgentConfig,
};
use types::{AgentId, Price};

use crate::Simulation;
use crate::sim_config::{SimConfig, SymbolSpec};

/// Pre-loaded ML models, ready for agent creation.
///
/// The binary (CLI) populates this from JSON files on disk.
/// The gym crate can populate it from episode config or bundled models.
#[derive(Default)]
pub struct MlModels {
    pub decision_trees: Vec<DecisionTree>,
    pub random_forests: Vec<RandomForest>,
    pub gradient_boosteds: Vec<GradientBoosted>,
}

impl MlModels {
    /// True if any models are loaded.
    pub fn has_models(&self) -> bool {
        !self.decision_trees.is_empty()
            || !self.random_forests.is_empty()
            || !self.gradient_boosteds.is_empty()
    }
}

/// Spawn ML agents from a collection of models, distributing with round-robin.
fn spawn_ml_agents<M: Clone + MlModel + 'static>(
    models: &[M],
    count: usize,
    start_id: u64,
    agent_config: TreeAgentConfig,
    warning_label: &str,
) -> Vec<Box<dyn Agent>> {
    if models.is_empty() {
        if count > 0 {
            eprintln!("  Warning: No {} models found in models/", warning_label);
        }
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            Box::new(TreeAgent::new(
                AgentId(start_id + i as u64),
                models[i % models.len()].clone(),
                agent_config.clone(),
            )) as Box<dyn Agent>
        })
        .collect()
}

/// Spawn tree-based ML agents and register models with the simulation.
///
/// Models must be pre-loaded (see `MlModels`). This function:
/// 1. Registers unique models with the simulation for centralized prediction caching
/// 2. Creates agents distributed across the loaded models via round-robin
pub(crate) fn spawn_tree_agents(
    sim: &mut Simulation,
    config: &SimConfig,
    symbols: &[SymbolSpec],
    start_id: u64,
    models: &MlModels,
) -> (Vec<Box<dyn Agent>>, u64) {
    // Register unique models for centralized prediction caching (O(M×S) vs O(N))
    for model in &models.decision_trees {
        sim.register_ml_model(model.clone());
    }
    for model in &models.random_forests {
        sim.register_ml_model(model.clone());
    }
    for model in &models.gradient_boosteds {
        sim.register_ml_model(model.clone());
    }
    if sim.has_ml_models() {
        eprintln!(
            "  Registered {} unique ML models for centralized caching",
            sim.ml_model_count()
        );
    }

    let agent_config = TreeAgentConfig {
        symbols: symbols.iter().map(|s| s.symbol.clone()).collect(),
        buy_threshold: config.tree_agent_buy_threshold,
        sell_threshold: config.tree_agent_sell_threshold,
        order_size: config.tree_agent_order_size,
        max_long_position: config.tree_agent_max_long_position,
        max_short_position: config.tree_agent_max_short_position,
        initial_cash: config.tree_agent_initial_cash,
        initial_price: symbols
            .first()
            .map(|s| s.initial_price)
            .unwrap_or(Price::from_float(100.0)),
    };

    let decision_tree_agents = spawn_ml_agents(
        &models.decision_trees,
        config.num_decision_tree_agents,
        start_id,
        agent_config.clone(),
        "decision tree",
    );

    let random_forest_agents = spawn_ml_agents(
        &models.random_forests,
        config.num_random_forest_agents,
        start_id + decision_tree_agents.len() as u64,
        agent_config.clone(),
        "random forest",
    );

    let gradient_boosted_agents = spawn_ml_agents(
        &models.gradient_boosteds,
        config.num_gradient_boosted_agents,
        start_id + decision_tree_agents.len() as u64 + random_forest_agents.len() as u64,
        agent_config,
        "gradient boosted",
    );

    let total =
        decision_tree_agents.len() + random_forest_agents.len() + gradient_boosted_agents.len();
    let agents = decision_tree_agents
        .into_iter()
        .chain(random_forest_agents)
        .chain(gradient_boosted_agents)
        .collect();

    (agents, start_id + total as u64)
}
