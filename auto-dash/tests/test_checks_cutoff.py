"""Tests for element cutoff detection check."""

from plotlint.checks.cutoff import ElementCutoffCheck
from plotlint.elements import ElementMap, ElementInfo, ElementCategory
from plotlint.geometry import BoundingBox
from plotlint.models import DefectType, Severity


def test_fully_inside():
    # Element fully within figure
    elements = [
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(10, 10, 90, 30)),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 0


def test_partial_cutoff_low():
    # Title extends slightly beyond top edge (9% cutoff)
    elements = [
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(10, -2, 90, 20)),
    ]
    # bbox area = 80 * 22 = 1760
    # inside = 80 * 20 = 1600
    # cutoff = 1 - 1600/1760 = 0.091 (9.1%, which is LOW)
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].defect_type == DefectType.ELEMENT_CUTOFF
    assert issues[0].severity == Severity.LOW


def test_partial_cutoff_medium():
    # Title extends 20% beyond edge
    elements = [
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(10, -5, 90, 20)),
    ]
    # Cutoff fraction = 1 - (inside_area / total_area)
    # inside_area = 80 * 20 = 1600
    # total_area = 80 * 25 = 2000
    # cutoff = 1 - 1600/2000 = 0.2 (20%)
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].severity == Severity.MEDIUM


def test_partial_cutoff_high():
    # Element more than 50% outside
    elements = [
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(10, -15, 90, 10)),
    ]
    # inside_area = 80 * 10 = 800
    # total_area = 80 * 25 = 2000
    # cutoff = 1 - 800/2000 = 0.6 (60%)
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].severity == Severity.HIGH


def test_fully_outside():
    # Element completely outside figure
    elements = [
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(110, 110, 190, 130)),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 1
    assert issues[0].severity == Severity.HIGH
    assert "100%" in issues[0].details


def test_zero_area_element_skipped():
    # Zero-area elements should be skipped
    elements = [
        ElementInfo("point", ElementCategory.TICK_LABEL, BoundingBox(5, 5, 5, 5)),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 0


def test_figure_element_skipped():
    # FIGURE category should be skipped
    elements = [
        ElementInfo("figure", ElementCategory.FIGURE, BoundingBox(0, 0, 100, 100)),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    check = ElementCutoffCheck()
    issues = check(elem_map)
    assert len(issues) == 0
