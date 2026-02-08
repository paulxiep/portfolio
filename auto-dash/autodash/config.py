"""Pipeline configuration.

Composes all sub-configs. Each module receives the relevant slice.
G2: Uses field(default_factory=...) for mutable defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plotlint.config import ConvergenceConfig
from plotlint.core.config import LLMConfig, SandboxConfig


@dataclass(frozen=True)
class ProfileConfig:
    """Thresholds for semantic type detection and profiling."""

    # Categorical: dual guard — BOTH conditions must hold
    categorical_max_cardinality: float = 0.5
    categorical_max_unique: int = 20
    identifier_min_cardinality: float = 0.95
    boolean_max_unique: int = 2
    boolean_string_values: frozenset[str] = frozenset(
        {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}
    )
    date_parse_sample_size: int = 100
    date_parse_success_threshold: float = 0.8
    top_values_count: int = 10
    sample_rows_count: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the AutoDash pipeline."""

    output_dir: str = "output"
    output_format: str = "png"
    max_charts: int = 1
    max_analysis_steps: int = 1
    max_exploration_attempts: int = 3
    inline_data_max_rows: int = 50

    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
