# plotlint/checks/cutoff.py

from __future__ import annotations

from plotlint.checks import check
from plotlint.elements import ElementMap, ElementCategory
from plotlint.models import Issue, DefectType, Severity


@check("element_cutoff")
class ElementCutoffCheck:
    """Detect elements that extend beyond the figure boundaries.

    Algorithm:
    1. Get the figure bounding box
    2. For each element, check if it's fully within the figure
    3. Compute cutoff fraction for severity
    4. Report: which elements are cut off, by how much

    Severity heuristic:
    - cutoff_fraction > 0.5 → HIGH (more than half of element outside)
    - cutoff_fraction > 0.1 → MEDIUM
    - cutoff_fraction > 0.0 → LOW
    """
    name = "element_cutoff"

    def __call__(self, elements: ElementMap) -> list[Issue]:
        issues = []
        figure_bbox = elements.figure_bbox

        # Check all non-figure elements
        for element in elements.elements:
            if element.category == ElementCategory.FIGURE:
                continue
            if element.bbox.area == 0:
                continue  # skip zero-size elements

            cutoff = element.bbox.cutoff_fraction(figure_bbox)
            if cutoff > 0.0:
                if cutoff > 0.5:
                    severity = Severity.HIGH
                elif cutoff > 0.1:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                issues.append(Issue(
                    defect_type=DefectType.ELEMENT_CUTOFF,
                    severity=severity,
                    details=(
                        f"Element '{element.element_id}' extends beyond figure: "
                        f"{cutoff:.0%} outside boundaries"
                    ),
                    suggestion="Add plt.tight_layout() or increase figure size",
                    element_ids=[element.element_id],
                ))
        return issues
