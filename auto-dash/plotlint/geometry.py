# plotlint/geometry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Origin: top-left of the figure (consistent with screen coordinates).
    All values in pixels.

    This is the fundamental geometric primitive shared by ALL checks
    and ALL extractors. Renderer-agnostic.
    """
    x0: float    # left edge
    y0: float    # top edge
    x1: float    # right edge
    y1: float    # bottom edge

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def overlaps(self, other: BoundingBox) -> bool:
        """True if this bbox overlaps with another."""
        return (
            self.x0 < other.x1 and self.x1 > other.x0 and
            self.y0 < other.y1 and self.y1 > other.y0
        )

    def intersection_area(self, other: BoundingBox) -> float:
        """Compute the area of overlap between two bounding boxes."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        return max(0, ix1 - ix0) * max(0, iy1 - iy0)

    def overlap_fraction(self, other: BoundingBox) -> float:
        """Overlap area as a fraction of the smaller bbox's area.

        Returns 0.0 if no overlap, up to 1.0 if one fully contains the other.
        Used for severity calculation.
        """
        inter = self.intersection_area(other)
        smaller_area = min(self.area, other.area)
        if smaller_area == 0:
            return 0.0
        return inter / smaller_area

    def is_within(self, container: BoundingBox) -> bool:
        """True if this bbox is fully contained within the container."""
        return (
            self.x0 >= container.x0 and self.y0 >= container.y0 and
            self.x1 <= container.x1 and self.y1 <= container.y1
        )

    def cutoff_fraction(self, container: BoundingBox) -> float:
        """Fraction of this bbox's area that falls outside the container.

        Returns 0.0 if fully inside, up to 1.0 if fully outside.
        Used for element_cutoff severity.
        """
        if self.area == 0:
            return 0.0
        inside = self.intersection_area(container)
        return 1.0 - (inside / self.area)

    @classmethod
    def from_matplotlib_bbox(cls, mpl_bbox) -> BoundingBox:
        """Convert a matplotlib Bbox to our BoundingBox.

        matplotlib Bbox uses (x0, y0) as bottom-left with y increasing upward.
        We use (x0, y0) as top-left with y increasing downward (screen coords).

        Args:
            mpl_bbox: matplotlib.transforms.Bbox instance.
        """
        # matplotlib: y0 = bottom, y1 = top (cartesian)
        # Our convention: y0 = top, y1 = bottom (screen)
        # Need the figure height to flip. This is handled by the extractor,
        # which knows the figure dimensions.
        # This classmethod receives already-flipped coordinates.
        return cls(
            x0=mpl_bbox.x0,
            y0=mpl_bbox.y0,
            x1=mpl_bbox.x1,
            y1=mpl_bbox.y1,
        )
