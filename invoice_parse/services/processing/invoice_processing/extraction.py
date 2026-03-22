"""LLM extraction substep — extract structured invoice data from OCR text.

Responsibilities:
- Define the LLM abstraction (LLMExtractor ABC)
- Implement GeminiExtractor (MVP primary)
- Build prompts with schema and critical instructions
- Stub ClaudeExtractor / OpenAIExtractor for future fallback
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

from invoice_shared.models import InvoiceExtraction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an invoice data extraction assistant. Extract structured data from the \
OCR text of an invoice. Follow these rules exactly:

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


def build_extraction_prompt(ocr_text: str) -> str:
    """Build the user prompt with OCR text and expected JSON schema."""
    schema = InvoiceExtraction.model_json_schema()
    return (
        "Extract the invoice data from the following OCR text.\n\n"
        "OCR Text:\n---\n"
        f"{ocr_text}\n"
        "---\n\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )


# --- LLM abstraction ---


class LLMExtractor(ABC):
    """Abstract interface for LLM-based invoice extraction."""

    @abstractmethod
    def extract(self, ocr_text: str) -> InvoiceExtraction: ...


class GeminiExtractor(LLMExtractor):
    """Gemini Flash via google-genai SDK with structured JSON output."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ["GEMINI_API_KEY"]
        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def extract(self, ocr_text: str) -> InvoiceExtraction:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        response = client.models.generate_content(
            model=self._model,
            contents=build_extraction_prompt(ocr_text),
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

    def extract(self, ocr_text: str) -> InvoiceExtraction:
        raise NotImplementedError("ClaudeExtractor is a production fallback, not in MVP scope")


class OpenAIExtractor(LLMExtractor):
    """GPT-4o-mini via response_format. Stub — not implemented for MVP."""

    def extract(self, ocr_text: str) -> InvoiceExtraction:
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
