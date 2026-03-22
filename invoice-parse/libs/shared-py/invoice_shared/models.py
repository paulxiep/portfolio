"""Pydantic models for extraction schema, queue messages, and job status."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, field_validator


class JobStatus(StrEnum):
    QUEUED = "queued"
    OCR_PROCESSING = "ocr_processing"
    OCR_DONE = "ocr_done"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    VALIDATING = "validating"
    DONE = "done"
    OUTPUT_GENERATED = "output_generated"
    DELIVERED = "delivered"
    OCR_FAILED = "ocr_failed"
    EXTRACTION_FAILED = "extraction_failed"
    NEEDS_REVIEW = "needs_review"
    DELIVERY_FAILED = "delivery_failed"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"


# --- Extraction Schema ---

_ISO_4217_CODES = {
    "CZK", "EUR", "USD", "GBP", "CHF", "PLN", "HUF", "SEK", "NOK", "DKK",
    "JPY", "CNY", "KRW", "AUD", "CAD", "NZD", "BRL", "MXN", "INR", "RUB",
    "TRY", "ZAR", "SGD", "HKD", "THB", "TWD", "ILS", "AED", "SAR", "PHP",
}

_CURRENCY_ALIASES: dict[str, str] = {
    "kc": "CZK", "kč": "CZK", "czk": "CZK",
    "eur": "EUR", "€": "EUR",
    "usd": "USD", "$": "USD",
    "gbp": "GBP", "£": "GBP",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LineItem(BaseModel):
    section: str | None = None
    date: str | None = None
    item: str
    quantity: float | None = None
    unit: str | None = None
    start_time: str | None = None
    finish_time: str | None = None
    hours: float | None = None
    total_hours: float | None = None
    tariff: float | None = None
    tariff_unit: str | None = None
    total: float


class InvoiceExtraction(BaseModel):
    supplier_name: str
    supplier_address: str | None = None
    client_name: str
    client_address: str | None = None
    invoice_number: str
    invoice_date: str
    invoice_date_end: str | None = None
    invoice_date_raw: str
    location: str | None = None
    total_excl_vat: float
    vat_amount: float
    vat_rate: float | None = None  # percentage integer, e.g. 20
    total_incl_vat: float
    currency: str  # ISO 4217
    line_items: list[LineItem]

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper in _ISO_4217_CODES:
            return upper
        alias = _CURRENCY_ALIASES.get(v.strip().lower())
        if alias:
            return alias
        raise ValueError(
            f"Unknown currency '{v}'. Expected ISO 4217 code (e.g. CZK, EUR, USD)."
        )

    @field_validator("invoice_date", "invoice_date_end")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not _DATE_RE.match(v):
            raise ValueError(
                f"Date '{v}' must be in YYYY-MM-DD format."
            )
        return v


# --- Queue Message Contracts ---

class QueueAMessage(BaseModel):
    """Ingestion → Processing."""
    job_id: str
    tenant_id: str
    blob_path: str
    source_channel: str  # "telegram" or "email"
    source_identifier: str  # chat_id or email address
    created_at: datetime


class QueueBMessage(BaseModel):
    """Processing → Output."""
    job_id: str
    tenant_id: str
    extraction: InvoiceExtraction
    confidence_score: float
    output_blob_path: str
    source_channel: str
    source_identifier: str
