# plotlint/inspector.py

from __future__ import annotations

from plotlint.elements import ElementMap, Extractor
from plotlint.models import InspectionResult, Issue
from plotlint.scoring import compute_score
from plotlint.checks import get_registered_checks


def inspect(elements: ElementMap) -> InspectionResult:
    """Run all registered checks on the extracted elements.

    This is the core orchestration function.
    It does NOT know which checks exist — it queries the registry.
    New checks are added by registering them (in checks/*.py),
    not by modifying this function.

    Args:
        elements: Extracted element map from any renderer's extractor.

    Returns:
        InspectionResult with all issues and computed score.
    """
    all_issues: list[Issue] = []
    checks = get_registered_checks()

    for check_name, check_fn in checks.items():
        issues = check_fn(elements)
        all_issues.extend(issues)

    score = compute_score(all_issues)

    return InspectionResult(
        issues=all_issues,
        score=score,
        element_count=len(elements.elements),
    )


def inspect_from_figure(
    figure_data: bytes,
    extractor: Extractor,
) -> InspectionResult:
    """Convenience: extract elements then inspect.

    This is what the convergence loop's inspect_node calls.
    """
    elements = extractor.extract(figure_data)
    return inspect(elements)
