"""LLM extraction substep — extract structured invoice data from OCR output.

Single responsibility: build prompts from OCR inputs, call LLM, return
InvoiceExtraction. Receives either or both:
- Raw OCR text (tab-separated lines with spatial layout)
- Table extraction text (structured tables from PPStructure or spatial clustering)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from invoice_shared.models import InvoiceExtraction

from .ocr import RawOcrOutput
from .table_extract import TableExtractionOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an invoice data extraction assistant. Extract structured data from the \
OCR output of an invoice. You may receive two views of the same document:

1. STRUCTURED TABLES: Tables detected by layout analysis, with pipe-delimited \
columns. These have reliable column structure but may miss some page regions.
2. RAW OCR TEXT: All text on the page with tab-separated items on the same row. \
This captures everything but has less structure.

Use both views together for best accuracy. Follow these rules exactly:

- Number formats: European invoices use comma as decimal separator and \
space/period as thousands separator (e.g., "2 305,00" means 2305.00). \
Parse numbers according to the currency context.
- Subtotals: Do NOT include subtotal, summary, or "in total" rows as line items. \
Only extract individual transaction/charge rows.
- Sections: Invoices may have multiple sections (e.g., Job/Labor, Miscellaneous, \
Materials). Preserve section names in the `section` field.
- Dates: Convert all dates to YYYY-MM-DD format. If the invoice shows a date range, \
use the first date as `invoice_date` and the second as `invoice_date_end`. \
Preserve original text in `invoice_date_raw`.
- Currency: Extract currency as a 3-letter ISO 4217 code (e.g., CZK, EUR, USD).
- VAT rate: Express as a percentage integer (e.g., 20 for 20%), not a decimal.
"""


def _format_raw_ocr_for_prompt(raw_ocr: RawOcrOutput) -> str:
    """Format raw OCR output as tab-separated lines for LLM prompt."""
    parts: list[str] = []
    for page in raw_ocr.pages:
        if len(raw_ocr.pages) > 1:
            parts.append(f"--- Page {page.page_number} ---")
        for line in page.lines:
            parts.append(line.text)
    return "\n".join(parts)


def build_extraction_prompt(
    raw_ocr: RawOcrOutput | None = None,
    table_extraction: TableExtractionOutput | None = None,
) -> str:
    """Build the user prompt from available OCR inputs.

    Accepts either or both inputs. The LLM gets whichever views
    are provided, with clear section headers.
    """
    if raw_ocr is None and table_extraction is None:
        raise ValueError("At least one of raw_ocr or table_extraction must be provided")

    schema = InvoiceExtraction.model_json_schema()
    sections: list[str] = []

    if table_extraction is not None:
        sections.append(
            "## Structured Tables\n"
            f"(method: {table_extraction.method})\n\n"
            f"{table_extraction.to_prompt_text()}"
        )

    if raw_ocr is not None:
        sections.append(
            "## Raw OCR Text\n"
            "(tab-separated items on same row indicate spatial alignment)\n\n"
            f"{_format_raw_ocr_for_prompt(raw_ocr)}"
        )

    ocr_text = "\n\n".join(sections)

    return (
        "Extract the invoice data from the following OCR output.\n\n"
        f"{ocr_text}\n\n"
        "---\n\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


# --- LLM abstraction ---


class LLMExtractor(ABC):
    """Abstract interface for LLM-based invoice extraction."""

    @abstractmethod
    def extract(
        self,
        raw_ocr: RawOcrOutput | None = None,
        table_extraction: TableExtractionOutput | None = None,
    ) -> InvoiceExtraction: ...


class GeminiExtractor(LLMExtractor):
    """Gemini Flash via google-genai SDK with structured JSON output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ["GEMINI_API_KEY"]
        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

    def extract(
        self,
        raw_ocr: RawOcrOutput | None = None,
        table_extraction: TableExtractionOutput | None = None,
    ) -> InvoiceExtraction:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        prompt = build_extraction_prompt(raw_ocr, table_extraction)

        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=InvoiceExtraction.model_json_schema(),
                temperature=0.0,
            ),
        )

        logger.debug("Gemini raw response: %s", response.text)
        return InvoiceExtraction.model_validate_json(response.text)


class ClaudeExtractor(LLMExtractor):
    """Claude Haiku via tool_use. Stub — not implemented for MVP."""

    def extract(
        self,
        raw_ocr: RawOcrOutput | None = None,
        table_extraction: TableExtractionOutput | None = None,
    ) -> InvoiceExtraction:
        raise NotImplementedError("ClaudeExtractor is a production fallback, not in MVP scope")


class OpenAIExtractor(LLMExtractor):
    """GPT-4o-mini via response_format. Stub — not implemented for MVP."""

    def extract(
        self,
        raw_ocr: RawOcrOutput | None = None,
        table_extraction: TableExtractionOutput | None = None,
    ) -> InvoiceExtraction:
        raise NotImplementedError("OpenAIExtractor is a production fallback, not in MVP scope")


# --- Factory ---


def create_extractor(provider: str = "gemini") -> LLMExtractor:
    """Create an LLM extractor by provider name."""
    match provider:
        case "gemini":
            return GeminiExtractor()
        case "claude":
            return ClaudeExtractor()
        case "openai":
            return OpenAIExtractor()
        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")
