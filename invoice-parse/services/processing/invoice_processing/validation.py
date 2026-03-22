"""Validation substep — business logic checks and confidence scoring.

Responsibilities:
- VAT arithmetic validation
- Line items sum validation with adaptive tolerance
- Date sanity checks
- Field completeness assessment
- Confidence scoring from validation signals (not LLM self-report)

All functions are pure — no I/O, no side effects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from invoice_shared.models import InvoiceExtraction

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.7

_OPTIONAL_FIELDS = (
    "supplier_address",
    "client_address",
    "invoice_date_end",
    "location",
    "vat_rate",
)


# --- Data structures ---


@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    name: str
    passed: bool
    skipped: bool = False
    detail: str = ""


@dataclass
class ValidationResult:
    """Aggregated validation outcome."""

    checks: list[ValidationCheck] = field(default_factory=list)
    confidence_score: float = 0.0
    needs_review: bool = False
    summary: str = ""

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.skipped)

    @property
    def checks_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.skipped)


# --- Individual validators ---


def validate_vat_math(extraction: InvoiceExtraction) -> list[ValidationCheck]:
    """Check VAT arithmetic consistency."""
    checks: list[ValidationCheck] = []

    # excl + vat_amount ≈ incl
    expected_total = extraction.total_excl_vat + extraction.vat_amount
    diff = abs(expected_total - extraction.total_incl_vat)
    checks.append(ValidationCheck(
        name="vat_sum",
        passed=diff < 0.02,
        detail=(
            f"excl({extraction.total_excl_vat}) + vat({extraction.vat_amount}) "
            f"= {expected_total}, incl={extraction.total_incl_vat}, diff={diff:.2f}"
        ),
    ))

    # If vat_rate provided, verify vat_amount ≈ excl × rate/100
    if extraction.vat_rate is not None:
        expected_vat = extraction.total_excl_vat * extraction.vat_rate / 100
        vat_diff = abs(expected_vat - extraction.vat_amount)
        checks.append(ValidationCheck(
            name="vat_rate_consistency",
            passed=vat_diff < 0.02,
            detail=(
                f"expected_vat={expected_vat:.2f}, "
                f"actual={extraction.vat_amount}, diff={vat_diff:.2f}"
            ),
        ))
    else:
        derived = ""
        if extraction.total_excl_vat > 0:
            derived_rate = (extraction.vat_amount / extraction.total_excl_vat) * 100
            derived = f"derived rate = {derived_rate:.1f}%"
        checks.append(ValidationCheck(
            name="vat_rate_consistency",
            passed=True,
            skipped=True,
            detail=f"vat_rate not provided; {derived}",
        ))

    return checks


def validate_line_items_sum(extraction: InvoiceExtraction) -> ValidationCheck:
    """Check that line items sum ≈ total_excl_vat with adaptive tolerance."""
    items_sum = sum(item.total for item in extraction.line_items)
    diff = abs(items_sum - extraction.total_excl_vat)
    n = len(extraction.line_items)
    abs_tolerance = max(0.01, 0.005 * n)
    rel_ok = (
        abs(diff / extraction.total_excl_vat) < 0.005
        if extraction.total_excl_vat
        else diff < 0.01
    )
    passed = diff <= abs_tolerance or rel_ok
    return ValidationCheck(
        name="line_items_sum",
        passed=passed,
        detail=(
            f"items_sum={items_sum:.2f}, "
            f"total_excl_vat={extraction.total_excl_vat:.2f}, "
            f"diff={diff:.2f}, tolerance={abs_tolerance:.3f}"
        ),
    )


def validate_dates(extraction: InvoiceExtraction) -> list[ValidationCheck]:
    """Date sanity: not in future, not ancient (year >= 2000)."""
    checks: list[ValidationCheck] = []
    today = date.today()

    for field_name in ("invoice_date", "invoice_date_end"):
        val = getattr(extraction, field_name)
        if val is None:
            continue
        d = date.fromisoformat(val)
        checks.append(ValidationCheck(
            name=f"{field_name}_not_future",
            passed=d <= today,
            detail=f"{field_name}={val}",
        ))
        checks.append(ValidationCheck(
            name=f"{field_name}_not_ancient",
            passed=d.year >= 2000,
            detail=f"{field_name}={val}",
        ))

    return checks


def validate_field_completeness(extraction: InvoiceExtraction) -> ValidationCheck:
    """Count how many optional fields are populated (informational)."""
    filled = sum(1 for f in _OPTIONAL_FIELDS if getattr(extraction, f) is not None)
    ratio = filled / len(_OPTIONAL_FIELDS)
    return ValidationCheck(
        name="field_completeness",
        passed=True,  # informational — always passes
        detail=f"{filled}/{len(_OPTIONAL_FIELDS)} optional fields filled ({ratio:.0%})",
    )


# --- Confidence scoring ---


def compute_confidence(
    checks: list[ValidationCheck],
    ocr_confidence: float,
    field_completeness_ratio: float,
) -> float:
    """Compute confidence from validation signals.

    Weights: 50% check pass rate, 30% OCR confidence, 20% field completeness.
    Does NOT use LLM self-reported confidence (FM-8.1).
    """
    applicable = [c for c in checks if not c.skipped]
    if applicable:
        check_score = sum(1 for c in applicable if c.passed) / len(applicable)
    else:
        check_score = 0.5

    score = 0.50 * check_score + 0.30 * ocr_confidence + 0.20 * field_completeness_ratio
    return round(min(1.0, max(0.0, score)), 3)


# --- Top-level entry point ---


def validate_extraction(
    extraction: InvoiceExtraction,
    ocr_avg_confidence: float,
) -> ValidationResult:
    """Run all validation checks and compute confidence score.

    Args:
        extraction: Parsed invoice data from LLM.
        ocr_avg_confidence: Average OCR confidence (0.0–1.0) from OcrOutput.
    """
    checks: list[ValidationCheck] = []
    checks.extend(validate_vat_math(extraction))
    checks.append(validate_line_items_sum(extraction))
    checks.extend(validate_dates(extraction))
    checks.append(validate_field_completeness(extraction))

    filled = sum(1 for f in _OPTIONAL_FIELDS if getattr(extraction, f) is not None)
    completeness_ratio = filled / len(_OPTIONAL_FIELDS)

    confidence = compute_confidence(checks, ocr_avg_confidence, completeness_ratio)
    needs_review = confidence < REVIEW_THRESHOLD

    failed = [c for c in checks if not c.passed and not c.skipped]
    summary = (
        "Failed checks: " + ", ".join(c.name for c in failed)
        if failed
        else "All checks passed"
    )

    return ValidationResult(
        checks=checks,
        confidence_score=confidence,
        needs_review=needs_review,
        summary=summary,
    )
