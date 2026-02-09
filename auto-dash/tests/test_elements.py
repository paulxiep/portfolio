"""Tests for plotlint.elements (ElementMap queries)."""

from plotlint.elements import ElementMap, ElementInfo, ElementCategory
from plotlint.geometry import BoundingBox


def test_by_category():
    elements = [
        ElementInfo("tick1", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10)),
        ElementInfo("tick2", ElementCategory.TICK_LABEL, BoundingBox(11, 0, 21, 10)),
        ElementInfo("title", ElementCategory.TITLE, BoundingBox(0, 0, 100, 20)),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    tick_labels = elem_map.by_category(ElementCategory.TICK_LABEL)
    assert len(tick_labels) == 2
    assert all(e.category == ElementCategory.TICK_LABEL for e in tick_labels)

    titles = elem_map.by_category(ElementCategory.TITLE)
    assert len(titles) == 1
    assert titles[0].element_id == "title"


def test_tick_labels():
    elements = [
        ElementInfo("x1", ElementCategory.TICK_LABEL, BoundingBox(0, 0, 10, 10),
                    metadata={"axis": "x"}),
        ElementInfo("x2", ElementCategory.TICK_LABEL, BoundingBox(11, 0, 21, 10),
                    metadata={"axis": "x"}),
        ElementInfo("y1", ElementCategory.TICK_LABEL, BoundingBox(0, 11, 10, 21),
                    metadata={"axis": "y"}),
    ]
    elem_map = ElementMap(elements, BoundingBox(0, 0, 100, 100))

    x_labels = elem_map.tick_labels(axis="x")
    assert len(x_labels) == 2
    assert all(e.metadata.get("axis") == "x" for e in x_labels)

    y_labels = elem_map.tick_labels(axis="y")
    assert len(y_labels) == 1
    assert y_labels[0].element_id == "y1"


def test_empty_element_map():
    elem_map = ElementMap([], BoundingBox(0, 0, 100, 100))
    assert len(elem_map.by_category(ElementCategory.TICK_LABEL)) == 0
    assert len(elem_map.tick_labels(axis="x")) == 0
