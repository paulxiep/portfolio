# plotlint/elements.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from plotlint.geometry import BoundingBox


class ElementCategory(str, Enum):
    """Categories of chart elements.

    Used by checks to select which elements to operate on.
    A check for label_overlap only looks at TICK_LABEL elements.
    """
    TICK_LABEL = "tick_label"
    AXIS_LABEL = "axis_label"
    TITLE = "title"
    SUPTITLE = "suptitle"
    LEGEND = "legend"
    DATA_ELEMENT = "data_element"   # bars, lines, points
    ANNOTATION = "annotation"
    FIGURE = "figure"               # the figure boundary itself


@dataclass(frozen=True)
class ElementInfo:
    """A single chart element with its bounding box and metadata.

    Renderer-agnostic. Both matplotlib and Plotly extractors
    produce ElementInfo objects.
    """
    element_id: str                            # unique identifier (e.g. "xaxis.tick.3")
    category: ElementCategory
    bbox: BoundingBox
    text: Optional[str] = None                 # label text, if applicable
    axis_index: int = 0                        # which axes (for multi-axes charts)
    metadata: dict[str, Any] = field(default_factory=dict)  # renderer-specific extras


@dataclass
class ElementMap:
    """Collection of all extracted elements from a rendered chart.

    The intermediate representation between extractors and checks.
    Checks query elements by category — they never touch renderer internals.
    """
    elements: list[ElementInfo]
    figure_bbox: BoundingBox                   # the figure boundary (for cutoff detection)

    def by_category(self, category: ElementCategory) -> list[ElementInfo]:
        """Get all elements of a given category."""
        return [e for e in self.elements if e.category == category]

    def tick_labels(self, axis: str = "x") -> list[ElementInfo]:
        """Get tick labels for a specific axis."""
        return [
            e for e in self.elements
            if e.category == ElementCategory.TICK_LABEL
            and e.metadata.get("axis") == axis
        ]


@runtime_checkable
class Extractor(Protocol):
    """Protocol for extracting element bounding boxes from a rendered figure.

    OCP: New renderers implement this protocol.
    - MatplotlibExtractor walks the artist tree.
    - PlotlyExtractor (PL-1.6) injects JS and parses getBoundingClientRect().
    Both produce the same ElementMap.
    """

    def extract(self, figure_data: bytes) -> ElementMap:
        """Extract all elements from a rendered figure.

        Args:
            figure_data: Opaque bytes from RenderResult.figure_data.
                         For matplotlib: pickled Figure.
                         For Plotly: HTML string.

        Returns:
            ElementMap with all detected elements and their bounding boxes.
        """
        ...
