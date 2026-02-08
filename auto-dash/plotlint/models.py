"""plotlint data models and convergence state.

Contains all dataclasses that cross module boundaries within plotlint.
ConvergenceState is the LangGraph TypedDict for the convergence loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypedDict


# --- Enums ---


class Severity(str, Enum):
    """Defect severity levels. Ordered for comparison."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DefectType(str, Enum):
    """Registry of known defect types.

    MVP starts with 2. PL-1 adds 3 more. PL-2 adds 4 more.
    New values are ADDED — existing values never change.
    """

    LABEL_OVERLAP = "label_overlap"
    ELEMENT_CUTOFF = "element_cutoff"


class RenderStatus(str, Enum):
    """Outcome of a render attempt."""

    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"
    NO_FIGURE = "no_figure"


# --- Dataclasses ---


@dataclass(frozen=True)
class Issue:
    """A single visual defect detected by the Inspector.

    Immutable. Communication contract between Inspector and Patcher.
    """

    defect_type: DefectType
    severity: Severity
    details: str
    suggestion: str
    element_ids: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format for inclusion in Patcher LLM prompts."""
        return (
            f"[{self.severity.value.upper()}] {self.defect_type.value}: "
            f"{self.details}. Suggestion: {self.suggestion}"
        )


@dataclass(frozen=True)
class InspectionResult:
    """Complete output from the Inspector.

    Contains all detected issues and an overall numeric score.
    """

    issues: list[Issue]
    score: float
    element_count: int = 0

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def highest_severity_issue(self) -> Optional[Issue]:
        """The most severe issue, for the Patcher to focus on."""
        if not self.issues:
            return None
        severity_order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
        return min(self.issues, key=lambda i: severity_order[i.severity])


@dataclass(frozen=True)
class FixAttempt:
    """Record of a single patch attempt.

    Used for history tracking and deduplication (PL-1.2).
    """

    iteration: int
    target_issue: DefectType
    description: str
    code_hash: str
    score_before: float
    score_after: float

    @property
    def improved(self) -> bool:
        return self.score_after > self.score_before


@dataclass(frozen=True)
class RenderResult:
    """Output of rendering chart code. Renderer-agnostic.

    Both matplotlib and Plotly renderers return this same type.
    G11: field `figure_data` maps to ConvergenceState `figure_pickle`.
    """

    status: RenderStatus
    png_bytes: Optional[bytes] = None
    figure_data: Optional[bytes] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: int = 0
    dpi: int = 100
    figsize: Optional[tuple[float, float]] = None

    @property
    def succeeded(self) -> bool:
        return self.status == RenderStatus.SUCCESS and self.png_bytes is not None


# --- Convergence Loop State (LangGraph TypedDict) ---


class ConvergenceState(TypedDict, total=False):
    """LangGraph state for the plotlint convergence loop.

    total=False: all fields optional, enabling incremental building.
    Nodes read/write only their fields. Future phases add fields additively.
    """

    # Core state (set by graph initialization)
    source_code: str
    original_code: str

    # Renderer output (set by render node - MVP.6)
    png_bytes: bytes
    render_error: Optional[str]
    figure_pickle: Optional[bytes]

    # Inspector output (set by inspect node - MVP.7)
    inspection: Optional[InspectionResult]
    score: float

    # Convergence tracking
    iteration: int
    max_iterations: int
    score_history: list[float]
    fix_history: list[FixAttempt]

    # Best-seen tracking (PL-1.2 rollback)
    best_code: str
    best_score: float

    # Patcher output (set by patch node - MVP.8)
    patch_applied: bool

    # Renderer bundle reference
    renderer_type: str

    # Critic output (PL-1.3)
    critic_feedback: Optional[str]
    critic_invoked: bool

    # Spec context (PL-1.3)
    spec_context: Optional[str]
