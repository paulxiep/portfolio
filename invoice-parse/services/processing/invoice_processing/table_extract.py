"""Table extraction — reconstruct table structure from raw OCR or layout models.

Single responsibility: take raw OCR lines (text + coordinates) and/or images,
produce structured table regions. Multiple strategies available:

- SpatialClusterExtractor: dynamic gap-based row/column clustering from coordinates
- PPStructureExtractor: PaddleOCR PPStructureV3 layout detection + HTML table parsing

Both output the same TableExtractionOutput, so the LLM extraction step
can receive either, both, or neither.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from html.parser import HTMLParser

import numpy as np
from PIL import Image

from .ocr import OcrLine, OcrPage, RawOcrOutput

logger = logging.getLogger(__name__)


# --- Data structures ---


@dataclass
class TableRegion:
    """A detected table or section on the page."""

    label: str  # "table", "title", "text"
    content: str  # formatted text (pipe-delimited rows for tables)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class TablePage:
    """Table extraction results for one page."""

    page_number: int
    regions: list[TableRegion] = field(default_factory=list)


@dataclass
class TableExtractionOutput:
    """Complete table extraction output for a document."""

    pages: list[TablePage]
    method: str  # "spatial_cluster", "ppstructure"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "pages": [
                {
                    "page_number": p.page_number,
                    "regions": [
                        {"label": r.label, "content": r.content, "rows": r.rows}
                        for r in p.regions
                    ],
                }
                for p in self.pages
            ],
        }

    def to_prompt_text(self) -> str:
        """Format for LLM consumption."""
        parts: list[str] = []
        for page in self.pages:
            if len(self.pages) > 1:
                parts.append(f"--- Page {page.page_number} ---")
            for region in page.regions:
                if region.label == "title":
                    parts.append(f"\n## {region.content}\n")
                elif region.label == "table" and region.rows:
                    for row in region.rows:
                        parts.append("| " + " | ".join(row) + " |")
                    parts.append("")
                elif region.label == "separator":
                    parts.append("---")
                else:
                    if region.content.strip():
                        parts.append(region.content)
                        parts.append("")
        return "\n".join(parts)


# --- HTML table parser (for PPStructure) ---


class _TableHtmlParser(HTMLParser):
    """Minimal parser to extract rows/cells from PPStructure table HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append("".join(self._current_cell).strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data)


def _parse_table_html(html: str) -> list[list[str]]:
    parser = _TableHtmlParser()
    parser.feed(html)
    return parser.rows


# --- Gap detection (shared utility) ---


def _detect_gaps(values: list[int]) -> list[list[int]]:
    """Cluster sorted values by detecting natural gaps.

    Computes gaps between consecutive values, uses the median gap as
    baseline. A gap > 2× median signals a cluster boundary.
    Returns list of clusters (each a list of original indices).
    """
    if len(values) <= 1:
        return [list(range(len(values)))]

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
    threshold = max(median_gap * 2, 1)

    clusters: list[list[int]] = [[0]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            clusters.append([])
        clusters[-1].append(i + 1)
    return clusters


# --- Extractor interface ---


class TableExtractor(ABC):
    """Abstract interface for table extraction strategies."""

    @abstractmethod
    def extract(
        self,
        raw_ocr: RawOcrOutput,
        images: list[Image.Image] | None = None,
    ) -> TableExtractionOutput: ...


# --- Spatial cluster extractor ---


class SpatialClusterExtractor(TableExtractor):
    """Reconstruct tables from raw OCR coordinates using gap detection.

    Groups lines into rows by y-coordinate gaps, joins same-row items
    with tabs. Inserts region separators at large y-gaps.
    """

    def extract(
        self,
        raw_ocr: RawOcrOutput,
        images: list[Image.Image] | None = None,
    ) -> TableExtractionOutput:
        pages: list[TablePage] = []
        for ocr_page in raw_ocr.pages:
            regions = self._cluster_page(ocr_page)
            pages.append(TablePage(page_number=ocr_page.page_number, regions=regions))
        return TableExtractionOutput(pages=pages, method="spatial_cluster")

    def _cluster_page(self, page: OcrPage) -> list[TableRegion]:
        if not page.lines:
            return []

        # Cluster lines into rows by y-gaps
        y_values = [line.y for line in page.lines]
        row_clusters = _detect_gaps(y_values)

        # Build row groups
        row_groups: list[list[OcrLine]] = []
        for cluster_indices in row_clusters:
            row_lines = sorted(
                [page.lines[i] for i in cluster_indices],
                key=lambda l: l.x,
            )
            row_groups.append(row_lines)

        # Detect region boundaries (large y-gaps between rows)
        row_y_values = [group[0].y for group in row_groups]
        region_break_indices: set[int] = set()
        if len(row_y_values) > 1:
            inter_row_gaps = [
                row_y_values[i + 1] - row_y_values[i]
                for i in range(len(row_y_values) - 1)
            ]
            median_row_gap = sorted(inter_row_gaps)[len(inter_row_gaps) // 2]
            region_threshold = max(median_row_gap * 3, 1)
            for i, gap in enumerate(inter_row_gaps):
                if gap > region_threshold:
                    region_break_indices.add(i + 1)

        # Format into regions
        regions: list[TableRegion] = []
        for i, row in enumerate(row_groups):
            if i in region_break_indices:
                regions.append(TableRegion(label="separator", content="---"))
            line_text = "\t".join(line.text for line in row)
            regions.append(TableRegion(label="text", content=line_text))

        return regions


# --- PPStructure extractor ---


class PPStructureExtractor(TableExtractor):
    """Extract tables using PaddleOCR PPStructureV3 layout detection.

    Requires images (runs its own inference). Raw OCR is not used.
    Detects titled sections + tables with HTML structure.
    """

    def extract(
        self,
        raw_ocr: RawOcrOutput,
        images: list[Image.Image] | None = None,
    ) -> TableExtractionOutput:
        if not images:
            raise ValueError("PPStructureExtractor requires images")

        from paddleocr import PPStructureV3

        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
            device="cpu",
        )

        pages: list[TablePage] = []
        for page_num, img in enumerate(images, start=1):
            results = list(pipeline.predict(input=np.array(img)))
            regions: list[TableRegion] = []
            for result in results:
                blocks = result.json["res"].get("parsing_res_list", [])
                for block in blocks:
                    region = self._build_region(block)
                    if region:
                        regions.append(region)
            pages.append(TablePage(page_number=page_num, regions=regions))

        return TableExtractionOutput(pages=pages, method="ppstructure")

    @staticmethod
    def _build_region(block: dict) -> TableRegion | None:
        label = block.get("block_label", "text")
        content = block.get("block_content", "")

        if label == "table":
            rows = _parse_table_html(content) if content else []
            text = "\n".join(" | ".join(row) for row in rows)
            return TableRegion(label="table", content=text, rows=rows)

        if label == "paragraph_title":
            return TableRegion(label="title", content=content.strip())

        if content.strip():
            return TableRegion(label="text", content=content.strip())

        return None


# --- Factory ---


def create_table_extractor(method: str = "spatial_cluster") -> TableExtractor:
    """Create a table extractor by method name."""
    match method:
        case "spatial_cluster":
            return SpatialClusterExtractor()
        case "ppstructure":
            return PPStructureExtractor()
        case _:
            raise ValueError(f"Unknown table extraction method: {method}")
