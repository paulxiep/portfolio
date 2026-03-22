"""OCR substep — PDF to structured text via PaddleOCR PPStructureV3.

Responsibilities:
- Convert PDF bytes to images (PyMuPDF)
- Run layout-aware OCR (PPStructureV3)
- Produce structured output with typed regions (header, table, text)
- Structural completeness check for degraded OCR detection
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MONETARY_PATTERN = re.compile(r"\d+[.,\s]\d{2}")


# --- Data structures ---


@dataclass
class OcrRegion:
    """A single detected region on a page."""

    type: str  # "title", "table", "text"
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class OcrPage:
    """OCR results for one page."""

    page_number: int
    regions: list[OcrRegion] = field(default_factory=list)


@dataclass
class OcrOutput:
    """Complete OCR output for a document."""

    pages: list[OcrPage]
    avg_confidence: float
    has_table_regions: bool
    has_monetary_patterns: bool

    def to_dict(self) -> dict:
        """Serialize for blob storage."""
        return {
            "pages": [
                {
                    "page_number": page.page_number,
                    "regions": [
                        {
                            "type": r.type,
                            "text": r.text,
                            "rows": r.rows,
                            "confidence": r.confidence,
                        }
                        for r in page.regions
                    ],
                }
                for page in self.pages
            ],
            "avg_confidence": self.avg_confidence,
            "has_table_regions": self.has_table_regions,
            "has_monetary_patterns": self.has_monetary_patterns,
        }

    def to_prompt_text(self) -> str:
        """Flatten OCR output into LLM-readable text.

        Titles rendered as uppercase headers, tables as pipe-delimited rows,
        text blocks as paragraphs.
        """
        parts: list[str] = []
        for page in self.pages:
            if len(self.pages) > 1:
                parts.append(f"--- Page {page.page_number} ---")
            for region in page.regions:
                if region.type == "title":
                    parts.append(f"\n## {region.text.strip()}\n")
                elif region.type == "table" and region.rows:
                    for row in region.rows:
                        parts.append("| " + " | ".join(row) + " |")
                    parts.append("")  # blank line after table
                elif region.type == "raw_text":
                    parts.append("\n--- Full page text (raw OCR) ---\n")
                    parts.append(region.text.strip())
                    parts.append("")
                else:
                    text = region.text.strip()
                    if text:
                        parts.append(text)
                        parts.append("")
        return "\n".join(parts)


# --- HTML table parser ---


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


def parse_table_html(html: str) -> list[list[str]]:
    """Extract rows from an HTML table string."""
    parser = _TableHtmlParser()
    parser.feed(html)
    return parser.rows


# --- PDF conversion ---


def pdf_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        images.append(img)
    doc.close()
    return images


# --- OCR engine ---


def _build_region(block: dict) -> OcrRegion:
    """Convert a PPStructureV3 parsing block to an OcrRegion.

    PPStructureV3 blocks have:
    - block_label: "paragraph_title", "table", "text", "figure", etc.
    - block_content: str (plain text for titles/text, HTML for tables)
    """
    label = block.get("block_label", "text")
    content = block.get("block_content", "")

    if label == "table":
        rows = parse_table_html(content) if content else []
        text = "\n".join(" | ".join(row) for row in rows)
        return OcrRegion(type="table", text=text, rows=rows, confidence=1.0)

    if label == "paragraph_title":
        return OcrRegion(type="title", text=content.strip(), confidence=1.0)

    # text, figure, or any other type
    return OcrRegion(type="text", text=content.strip(), confidence=1.0)


def _run_raw_ocr(images: list[Image.Image]) -> list[list[tuple[str, int, int]]]:
    """Run basic PaddleOCR on images, return text lines with coordinates.

    Returns per-page list of (text, x, y) tuples sorted by y then x.
    Used as a supplement to PPStructureV3 which may miss regions
    at the bottom of the page (e.g. Misc sections, VAT summaries).
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en")
    all_pages: list[list[tuple[str, int, int]]] = []
    for img in images:
        results = list(ocr.predict(input=np.array(img)))
        lines: list[tuple[str, int, int]] = []
        for r in results:
            for text, poly in zip(r["rec_texts"], r["dt_polys"]):
                x = int(min(p[0] for p in poly))
                y = int(min(p[1] for p in poly))
                lines.append((text, x, y))
        lines.sort(key=lambda t: (t[2], t[1]))  # sort by y, then x
        all_pages.append(lines)
    return all_pages


def _detect_gaps(values: list[int]) -> list[list[int]]:
    """Cluster sorted values by detecting natural gaps.

    Computes all pairwise gaps between consecutive values, then uses
    the median gap as the baseline. A gap > 2× median signals a cluster
    boundary. Returns list of clusters (each a list of original indices).
    """
    if len(values) <= 1:
        return [list(range(len(values)))]

    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
    threshold = max(median_gap * 2, 1)  # at least 1px to avoid zero-threshold

    clusters: list[list[int]] = [[0]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            clusters.append([])
        clusters[-1].append(i + 1)
    return clusters


def _format_raw_lines(
    lines: list[tuple[str, int, int]],
    structure_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> str:
    """Format raw OCR lines into structured text for LLM consumption.

    1. Filter out lines that fall inside PPStructureV3-detected regions
       (those are already represented as structured tables/titles).
    2. Cluster remaining lines into rows by y-coordinate gaps.
    3. Within each row, cluster into columns by x-coordinate gaps.
    4. Detect region boundaries (large y-gaps) and insert separators.

    Args:
        lines: (text, x, y) tuples sorted by y then x.
        structure_bboxes: [(x1, y1, x2, y2), ...] of PPStructureV3 regions
            to exclude from raw output (already covered).
    """
    if not lines:
        return ""

    # Note: we intentionally do NOT filter lines by PPStructureV3 bboxes.
    # PPStructureV3 bboxes can be larger than the content they actually
    # detected (e.g. Job table bbox swallows the Misc section below it).
    # The raw text section provides the full page view; the LLM uses both
    # the structured tables above and this raw text to extract all fields.
    _ = structure_bboxes  # reserved for future use

    if not lines:
        return ""

    # Cluster into rows by y-coordinate gaps
    y_values = [line[2] for line in lines]
    row_clusters = _detect_gaps(y_values)

    # Detect region separators: find row boundaries where the y-gap is
    # significantly larger than within-region gaps (> 3× median row gap)
    row_y_centers = []
    row_groups: list[list[tuple[str, int, int]]] = []
    for cluster_indices in row_clusters:
        row_lines = sorted([lines[i] for i in cluster_indices], key=lambda t: t[1])
        row_groups.append(row_lines)
        row_y_centers.append(row_lines[0][2])

    region_gaps: list[int] = []
    if len(row_y_centers) > 1:
        inter_row_gaps = [
            row_y_centers[i + 1] - row_y_centers[i]
            for i in range(len(row_y_centers) - 1)
        ]
        median_row_gap = sorted(inter_row_gaps)[len(inter_row_gaps) // 2]
        region_threshold = max(median_row_gap * 3, 1)
        for i, gap in enumerate(inter_row_gaps):
            if gap > region_threshold:
                region_gaps.append(i)

    # Format output with tab-separated columns and region dividers
    output_lines: list[str] = []
    for i, row in enumerate(row_groups):
        if i in region_gaps:
            output_lines.append("---")
        output_lines.append("\t".join(item[0] for item in row))

    return "\n".join(output_lines)


def run_ocr(images: list[Image.Image]) -> OcrOutput:
    """Run PaddleOCR PPStructureV3 + raw OCR on images.

    PPStructureV3 provides table structure but may miss page regions.
    Raw OCR captures all text. Both are included so the LLM gets
    complete information.
    """
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

    # Run raw OCR for full-page text with coordinates
    raw_pages = _run_raw_ocr(images)

    pages: list[OcrPage] = []
    has_tables = False
    has_monetary = False

    for page_num, img in enumerate(images, start=1):
        img_array = np.array(img)
        results = list(pipeline.predict(input=img_array))

        regions: list[OcrRegion] = []
        structure_bboxes: list[tuple[int, int, int, int]] = []
        for result in results:
            blocks = result.json["res"].get("parsing_res_list", [])
            for block in blocks:
                region = _build_region(block)
                regions.append(region)

                if region.type == "table":
                    has_tables = True

                # Collect bboxes of detected regions
                bbox = block.get("block_bbox", [])
                if len(bbox) == 4:
                    structure_bboxes.append(
                        (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    )

        # Raw OCR for lines outside PPStructureV3-detected regions
        raw_lines = raw_pages[page_num - 1] if page_num <= len(raw_pages) else []
        if raw_lines:
            raw_text = _format_raw_lines(raw_lines, structure_bboxes)
            if raw_text.strip():
                has_monetary = bool(_MONETARY_PATTERN.search(raw_text))
                regions.append(OcrRegion(
                    type="raw_text",
                    text=raw_text,
                    confidence=1.0,
                ))

        pages.append(OcrPage(page_number=page_num, regions=regions))

    avg_confidence = 1.0 if has_tables else 0.5

    # Structural completeness check (FM-1.1)
    if not has_tables and has_monetary:
        logger.warning(
            "No table regions detected but monetary patterns found in text. "
            "OCR layout detection may be degraded."
        )

    return OcrOutput(
        pages=pages,
        avg_confidence=avg_confidence,
        has_table_regions=has_tables,
        has_monetary_patterns=has_monetary,
    )


# --- Top-level entry point ---


def process_ocr(pdf_bytes: bytes) -> OcrOutput:
    """PDF bytes → structured OCR output."""
    images = pdf_to_images(pdf_bytes)
    logger.info("Converted PDF to %d page image(s)", len(images))
    return run_ocr(images)
