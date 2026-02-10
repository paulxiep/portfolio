# plotlint/checks/overlap.py

from __future__ import annotations

from plotlint.checks import check
from plotlint.elements import ElementMap, ElementCategory
from plotlint.models import Issue, DefectType, Severity


@check("label_overlap")
class LabelOverlapCheck:
    """Detect overlapping tick labels on x and y axes.

    Algorithm:
    1. Get all tick labels (grouped by axis)
    2. For each adjacent pair, check bbox overlap
    3. Compute overlap fraction for severity
    4. Report: how many labels collide, which ones, severity

    Severity heuristic:
    - > 50% of labels overlap → HIGH
    - > 20% of labels overlap → MEDIUM
    - any overlap → LOW
    """
    name = "label_overlap"

    def __call__(self, elements: ElementMap) -> list[Issue]:
        issues = []
        for axis in ("x", "y"):
            labels = elements.tick_labels(axis=axis)
            if len(labels) < 2:
                continue

            # Sort by position along the axis
            if axis == "x":
                labels.sort(key=lambda e: e.bbox.x0)
            else:
                labels.sort(key=lambda e: e.bbox.y0)

            collisions = 0
            for i in range(len(labels) - 1):
                if labels[i].bbox.overlaps(labels[i + 1].bbox):
                    collisions += 1

            if collisions > 0:
                total_pairs = len(labels) - 1
                collision_ratio = collisions / total_pairs

                if collision_ratio > 0.5:
                    severity = Severity.HIGH
                elif collision_ratio > 0.2:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(Issue(
                    defect_type=DefectType.LABEL_OVERLAP,
                    severity=severity,
                    details=(
                        f"{axis.upper()}-axis labels overlap: "
                        f"{collisions} of {total_pairs} adjacent pairs collide"
                    ),
                    suggestion=(
                        f"Rotate {axis}-axis labels 45-90 degrees"
                        if axis == "x"
                        else f"Increase figure height or reduce {axis}-axis label count"
                    ),
                    element_ids=[l.element_id for l in labels],
                ))
        return issues
