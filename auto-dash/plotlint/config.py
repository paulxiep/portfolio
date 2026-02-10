"""Convergence loop configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configurable parameters for the convergence loop.

    Declarative: behavior is driven by these values, not if-else chains.
    """

    max_iterations: int = 3
    target_score: float = 1.0
    stagnation_window: int = 2
    score_improvement_threshold: float = 0.01
