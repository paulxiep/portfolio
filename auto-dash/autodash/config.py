"""Pipeline configuration.

Composes all sub-configs. Each module receives the relevant slice.
G2: Uses field(default_factory=...) for mutable defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plotlint.config import ConvergenceConfig
from plotlint.core.config import LLMConfig, SandboxConfig


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
