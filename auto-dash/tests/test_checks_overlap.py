"""Tests for label overlap detection check."""

from plotlint.checks.overlap import LabelOverlapCheck
from plotlint.elements import ElementMap, ElementInfo, ElementCategory
from plotlint.geometry import BoundingBox
from plotlint.models import DefectType, Severity


def test_no_overlap():
    # Adjacent labels with 1px gap
    elements = [
        ElementInfo("tick.0", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10),
                    metadata={"axis": "x"}),
        ElementInfo("tick.1", ElementCategory.TICK_LABEL, BoundingBox(11, 0, 21, 10),
                    metadata={"axis": "x"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 0


def test_single_overlap():
    # Two overlapping labels
    elements = [
        ElementInfo("tick.0", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10),
                    metadata={"axis": "x"}),
        ElementInfo("tick.1", ElementCategory.TICK_LABEL, BoundingBox(5, 0, 15, 10),
                    metadata={"axis": "x"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].defect_type == DefectType.LABEL_OVERLAP
    assert issues[0].severity == Severity.HIGH  # 1 of 1 = 100%, >50%
    assert "1 of 1" in issues[0].details


def test_high_severity():
    # 6 out of 10 adjacent pairs overlap (>50%)
    elements = []
    for i in range(11):
        # Every other pair overlaps
        x = i * 5 if i % 2 == 0 else (i - 1) * 5 + 3
        elements.append(
            ElementInfo(f"tick.{i}", ElementCategory.TICK_LABEL,
                       BoundingBox(x, 0, x + 10, 10),
                       metadata={"axis": "x"})
        )
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].severity == Severity.HIGH


def test_medium_severity():
    # 3 out of 10 pairs overlap (30%, >20%)
    elements = [
        # 3 overlapping pairs
        ElementInfo("tick.0", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10), metadata={"axis": "x"}),
        ElementInfo("tick.1", ElementCategory.TICK_LABEL, BoundingBox(5, 0, 15, 10), metadata={"axis": "x"}),
        ElementInfo("tick.2", ElementCategory.TICK_LABEL, BoundingBox(10, 0, 20, 10), metadata={"axis": "x"}),
        ElementInfo("tick.3", ElementCategory.TICK_LABEL, BoundingBox(15, 0, 25, 10), metadata={"axis": "x"}),
        # 7 non-overlapping pairs
        ElementInfo("tick.4", ElementCategory.TICK_LABEL, BoundingBox(30, 0, 40, 10), metadata={"axis": "x"}),
        ElementInfo("tick.5", ElementCategory.TICK_LABEL, BoundingBox(45, 0, 55, 10), metadata={"axis": "x"}),
        ElementInfo("tick.6", ElementCategory.TICK_LABEL, BoundingBox(60, 0, 70, 10), metadata={"axis": "x"}),
        ElementInfo("tick.7", ElementCategory.TICK_LABEL, BoundingBox(75, 0, 85, 10), metadata={"axis": "x"}),
        ElementInfo("tick.8", ElementCategory.TICK_LABEL, BoundingBox(90, 0, 100, 10), metadata={"axis": "x"}),
        ElementInfo("tick.9", ElementCategory.TICK_LABEL, BoundingBox(105, 0, 115, 10), metadata={"axis": "x"}),
        ElementInfo("tick.10", ElementCategory.TICK_LABEL, BoundingBox(120, 0, 130, 10), metadata={"axis": "x"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 200, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].severity == Severity.MEDIUM  # 3/10 = 30%


def test_separate_axes():
    # X-axis overlaps, Y-axis clean - should report only X
    elements = [
        # X-axis: overlapping
        ElementInfo("x.0", ElementCategory.TICK_LABEL, BoundingBox(0, 90, 10, 100),
                    metadata={"axis": "x"}),
        ElementInfo("x.1", ElementCategory.TICK_LABEL, BoundingBox(5, 90, 15, 100),
                    metadata={"axis": "x"}),
        # Y-axis: clean
        ElementInfo("y.0", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10),
                    metadata={"axis": "y"}),
        ElementInfo("y.1", ElementCategory.TICK_LABEL, BoundingBox(0, 15, 10, 25),
                    metadata={"axis": "y"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert "X-axis" in issues[0].details


def test_single_label():
    # Single label should not produce issues
    elements = [
        ElementInfo("tick.0", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10),
                    metadata={"axis": "x"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = LabelOverlapCheck()
    issues = check(elem_map)
    assert len(issues) == 0
