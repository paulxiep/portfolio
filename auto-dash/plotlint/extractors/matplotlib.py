# plotlint/extractors/matplotlib.py

from __future__ import annotations

import pickle
from typing import Any

from plotlint.elements import ElementMap, ElementInfo, ElementCategory
from plotlint.geometry import BoundingBox
from plotlint.core.errors import ExtractionError


class MatplotlibExtractor:
    """Extract element bounding boxes from a pickled matplotlib Figure.

    Walks the artist tree:
        fig → axes → xaxis/yaxis → ticklabels, title, legend, patches, lines

    Calls get_window_extent(renderer) on each artist to get pixel-space bboxes.
    Converts matplotlib's bottom-left origin to screen top-left origin.

    CRITICAL: fig.canvas.draw() MUST have been called before extraction
    to ensure layout is finalized. The renderer (MVP.6) handles this.
    """

    def extract(self, figure_data: bytes) -> ElementMap:
        """Unpickle the Figure and walk its artist tree.

        Args:
            figure_data: Pickled matplotlib.figure.Figure bytes.
        """
        try:
            fig = pickle.loads(figure_data)
            fig.canvas.draw()  # ensure layout is computed
            renderer = fig.canvas.get_renderer()
        except Exception as e:
            raise ExtractionError(f"Failed to load or draw figure: {e}") from e

        try:
            fig_height = fig.get_size_inches()[1] * fig.dpi

            elements: list[ElementInfo] = []

            # Figure bbox
            fig_bbox = BoundingBox(
                x0=0, y0=0,
                x1=fig.get_size_inches()[0] * fig.dpi,
                y1=fig_height,
            )

            for ax_idx, ax in enumerate(fig.get_axes()):
                # Tick labels
                elements.extend(
                    self._extract_tick_labels(ax, renderer, fig_height, ax_idx, "x")
                )
                elements.extend(
                    self._extract_tick_labels(ax, renderer, fig_height, ax_idx, "y")
                )

                # Title
                title = ax.get_title()
                if title:
                    title_bbox = self._mpl_to_bbox(
                        ax.title.get_window_extent(renderer), fig_height
                    )
                    if title_bbox.area > 0:
                        elements.append(ElementInfo(
                            element_id=f"axes.{ax_idx}.title",
                            category=ElementCategory.TITLE,
                            bbox=title_bbox,
                            text=title,
                            axis_index=ax_idx,
                        ))

                # Legend
                legend = ax.get_legend()
                if legend is not None:
                    legend_bbox = self._mpl_to_bbox(
                        legend.get_window_extent(renderer), fig_height
                    )
                    if legend_bbox.area > 0:
                        elements.append(ElementInfo(
                            element_id=f"axes.{ax_idx}.legend",
                            category=ElementCategory.LEGEND,
                            bbox=legend_bbox,
                            axis_index=ax_idx,
                        ))

                # Axis labels
                for axis_name, label_artist in [("x", ax.xaxis.label), ("y", ax.yaxis.label)]:
                    label_text = label_artist.get_text()
                    if label_text:
                        label_bbox = self._mpl_to_bbox(
                            label_artist.get_window_extent(renderer), fig_height
                        )
                        if label_bbox.area > 0:
                            elements.append(ElementInfo(
                                element_id=f"axes.{ax_idx}.{axis_name}_label",
                                category=ElementCategory.AXIS_LABEL,
                                bbox=label_bbox,
                                text=label_text,
                                axis_index=ax_idx,
                            ))

            # Suptitle
            if fig._suptitle is not None and fig._suptitle.get_text():
                suptitle_bbox = self._mpl_to_bbox(
                    fig._suptitle.get_window_extent(renderer), fig_height
                )
                if suptitle_bbox.area > 0:
                    elements.append(ElementInfo(
                        element_id="suptitle",
                        category=ElementCategory.SUPTITLE,
                        bbox=suptitle_bbox,
                        text=fig._suptitle.get_text(),
                    ))

            return ElementMap(elements=elements, figure_bbox=fig_bbox)
        except Exception as e:
            raise ExtractionError(f"Failed to extract elements: {e}") from e

    def _extract_tick_labels(
        self, ax, renderer, fig_height: float, ax_idx: int, axis: str,
    ) -> list[ElementInfo]:
        """Extract tick label elements for one axis."""
        axis_obj = ax.xaxis if axis == "x" else ax.yaxis
        labels = axis_obj.get_ticklabels()
        results = []

        for i, label in enumerate(labels):
            text = label.get_text()
            if not text or not label.get_visible():
                continue
            try:
                mpl_bbox = label.get_window_extent(renderer)
                bbox = self._mpl_to_bbox(mpl_bbox, fig_height)
                if bbox.area > 0:
                    results.append(ElementInfo(
                        element_id=f"axes.{ax_idx}.{axis}axis.tick.{i}",
                        category=ElementCategory.TICK_LABEL,
                        bbox=bbox,
                        text=text,
                        axis_index=ax_idx,
                        metadata={"axis": axis, "tick_index": i},
                    ))
            except Exception:
                # Skip labels that fail to get extent (e.g., malformed artists)
                continue
        return results

    def _mpl_to_bbox(self, mpl_bbox, fig_height: float) -> BoundingBox:
        """Convert matplotlib Bbox (bottom-left origin) to screen coords (top-left origin)."""
        return BoundingBox(
            x0=mpl_bbox.x0,
            y0=fig_height - mpl_bbox.y1,   # flip Y axis
            x1=mpl_bbox.x1,
            y1=fig_height - mpl_bbox.y0,   # flip Y axis
        )
