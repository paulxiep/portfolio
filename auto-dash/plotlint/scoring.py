# plotlint/scoring.py

from __future__ import annotations

from plotlint.models import Issue, Severity


# Severity weights for score calculation.
# Declarative: change weights without changing logic.
SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.HIGH: 1.0,
    Severity.MEDIUM: 0.5,
    Severity.LOW: 0.2,
}

# Maximum weighted demerits before score floors at 0.0
MAX_DEMERITS: float = 5.0


def compute_score(issues: list[Issue]) -> float:
    """Compute a 0.0-1.0 score from a list of issues.

    Score = 1.0 - (total_weighted_demerits / MAX_DEMERITS), clamped to [0, 1].

    Each issue contributes its severity weight as a demerit.
    A chart with no issues scores 1.0.
    A chart with 5+ high-severity issues scores 0.0.

    This is a pure function — deterministic, no side effects.
    """
    total_demerits = sum(
        SEVERITY_WEIGHTS.get(issue.severity, 0.5)
        for issue in issues
    )
    score = 1.0 - (total_demerits / MAX_DEMERITS)
    return max(0.0, min(1.0, score))
