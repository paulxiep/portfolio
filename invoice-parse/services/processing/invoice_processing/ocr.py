"""OCR substep — raw text extraction from PDF with coordinates.

Single responsibility: convert PDF to images, run OCR, output
(text, x, y) per detected line. No table reconstruction, no formatting.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --- Data structures ---


@dataclass
class OcrLine:
    """A single detected text line with position."""

    text: str
    x: int
    y: int


@dataclass
class OcrPage:
    """Raw OCR results for one page."""

    page_number: int
    width: int
    height: int
    lines: list[OcrLine] = field(default_factory=list)


@dataclass
class RawOcrOutput:
    """Complete raw OCR output for a document."""

    pages: list[OcrPage]

    def to_dict(self) -> dict:
        """Serialize for blob storage."""
        return {
            "pages": [
                {
                    "page_number": p.page_number,
                    "width": p.width,
                    "height": p.height,
                    "lines": [
                        {"text": l.text, "x": l.x, "y": l.y}
                        for l in p.lines
                    ],
                }
                for p in self.pages
            ],
        }


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


# --- Raw OCR ---


def run_raw_ocr(images: list[Image.Image]) -> RawOcrOutput:
    """Run PaddleOCR on images, return text lines with coordinates."""
    from paddleocr import PaddleOCR

    import os
    ocr = PaddleOCR(
        lang="en",
        text_detection_model_name=os.environ.get("OCR_DET_MODEL", "PP-OCRv5_server_det"),
        text_recognition_model_name=os.environ.get("OCR_REC_MODEL", "en_PP-OCRv5_server_rec"),
    )
    pages: list[OcrPage] = []

    for page_num, img in enumerate(images, start=1):
        img_array = np.array(img)
        results = list(ocr.predict(input=img_array))
        lines: list[OcrLine] = []
        for r in results:
            for text, poly in zip(r["rec_texts"], r["dt_polys"]):
                x = int(min(p[0] for p in poly))
                y = int(min(p[1] for p in poly))
                lines.append(OcrLine(text=text, x=x, y=y))
        lines.sort(key=lambda l: (l.y, l.x))
        pages.append(OcrPage(
            page_number=page_num,
            width=img.size[0],
            height=img.size[1],
            lines=lines,
        ))

    return RawOcrOutput(pages=pages)


# --- Top-level entry point ---


def process_ocr(file_bytes: bytes, filename: str = "") -> tuple[RawOcrOutput, list[Image.Image]]:
    """File bytes → raw OCR output + images.

    Supports PDF and image files (PNG, JPG, WEBP, etc.).
    Returns both because table extractors may need the images.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if filename else ""

    if ext == "pdf" or file_bytes[:5] == b"%PDF-":
        images = pdf_to_images(file_bytes)
        logger.info("Converted PDF to %d page image(s)", len(images))
    else:
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        images = [img]
        logger.info("Loaded image (%dx%d)", img.size[0], img.size[1])

    raw_ocr = run_raw_ocr(images)
    logger.info(
        "Raw OCR: %d lines across %d page(s)",
        sum(len(p.lines) for p in raw_ocr.pages),
        len(raw_ocr.pages),
    )
    return raw_ocr, images
