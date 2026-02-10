"""Tests for plotlint.geometry (BoundingBox operations)."""

import pytest
from plotlint.geometry import BoundingBox


class TestBoundingBox:
    def test_properties(self):
        bbox = BoundingBox(x0=10, y0=20, x1=50, y1=60)
        assert bbox.width == 40
        assert bbox.height == 40
        assert bbox.area == 1600
        assert bbox.center == (30, 40)

    def test_overlaps_non_overlapping(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(11, 0, 21, 10)
        assert not bbox1.overlaps(bbox2)

    def test_overlaps_edge_adjacent(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(10, 0, 20, 10)
        assert not bbox1.overlaps(bbox2)

    def test_overlaps_partial(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(5, 0, 15, 10)
        assert bbox1.overlaps(bbox2)

    def test_overlaps_full(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(2, 2, 8, 8)
        assert bbox1.overlaps(bbox2)

    def test_intersection_area_zero(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(11, 0, 21, 10)
        assert bbox1.intersection_area(bbox2) == 0

    def test_intersection_area_partial(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(5, 0, 15, 10)
        assert bbox1.intersection_area(bbox2) == 50

    def test_intersection_area_full(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(2, 2, 8, 8)
        assert bbox1.intersection_area(bbox2) == 36

    def test_overlap_fraction_zero(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(11, 0, 21, 10)
        assert bbox1.overlap_fraction(bbox2) == 0.0

    def test_overlap_fraction_partial(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(5, 0, 15, 10)
        assert bbox1.overlap_fraction(bbox2) == 0.5

    def test_overlap_fraction_full(self):
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(2, 2, 8, 8)
        assert bbox1.overlap_fraction(bbox2) == 1.0

    def test_is_within_fully_inside(self):
        bbox = BoundingBox(2, 2, 8, 8)
        container = BoundingBox(0, 0, 10, 10)
        assert bbox.is_within(container)

    def test_is_within_partially_outside(self):
        bbox = BoundingBox(5, 5, 15, 15)
        container = BoundingBox(0, 0, 10, 10)
        assert not bbox.is_within(container)

    def test_is_within_fully_outside(self):
        bbox = BoundingBox(11, 11, 21, 21)
        container = BoundingBox(0, 0, 10, 10)
        assert not bbox.is_within(container)

    def test_cutoff_fraction_fully_inside(self):
        bbox = BoundingBox(2, 2, 8, 8)
        container = BoundingBox(0, 0, 10, 10)
        assert bbox.cutoff_fraction(container) == 0.0

    def test_cutoff_fraction_partial(self):
        bbox = BoundingBox(5, 5, 15, 15)
        container = BoundingBox(0, 0, 10, 10)
        # bbox area = 100, inside area = 25, cutoff = 75%
        assert bbox.cutoff_fraction(container) == 0.75

    def test_cutoff_fraction_fully_outside(self):
        bbox = BoundingBox(11, 11, 21, 21)
        container = BoundingBox(0, 0, 10, 10)
        assert bbox.cutoff_fraction(container) == 1.0

    def test_negative_dimensions(self):
        # Bbox with x1 < x0 should handle gracefully
        bbox = BoundingBox(10, 10, 5, 5)
        assert bbox.width == -5
        assert bbox.height == -5
        assert bbox.area == 0  # max(0, width) * max(0, height)

    def test_zero_area(self):
        # Point
        bbox = BoundingBox(5, 5, 5, 5)
        assert bbox.area == 0
        # Horizontal line
        bbox = BoundingBox(0, 5, 10, 5)
        assert bbox.area == 0
        # Vertical line
        bbox = BoundingBox(5, 0, 5, 10)
        assert bbox.area == 0

    def test_coordinate_system(self):
        # Top-left origin (screen coordinates)
        bbox = BoundingBox(x0=10, y0=20, x1=50, y1=60)
        assert bbox.width == 40
        assert bbox.height == 40
        assert bbox.center == (30, 40)
        # y increases downward
        assert bbox.y1 > bbox.y0
