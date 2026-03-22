"""CLI for testing the invoice processing pipeline on a single PDF.

Usage:
    python -m invoice_processing.cli path/to/invoice.pdf [--provider gemini] [--output-dir ./output] [--ocr-only] [-v]

No Redis or Postgres required. Uses local filesystem and skips DB state transitions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ocr import process_ocr
from .extraction import create_extractor
from .validation import validate_extraction


def main() -> None:
    # Load .env file if present (for GEMINI_API_KEY etc.)
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Process a single invoice PDF")
    parser.add_argument("pdf_path", type=Path, help="Path to invoice PDF")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory for output files (default: ./output)",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Run only OCR step, skip LLM extraction and validation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Read PDF
    pdf_bytes = args.pdf_path.read_bytes()
    logger.info("Read %d bytes from %s", len(pdf_bytes), args.pdf_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- OCR ---
    logger.info("Running OCR...")
    ocr_output = process_ocr(pdf_bytes)

    ocr_path = args.output_dir / "ocr_output.json"
    ocr_path.write_text(json.dumps(ocr_output.to_dict(), indent=2, ensure_ascii=False))
    logger.info("OCR output written to %s", ocr_path)
    logger.info(
        "OCR avg confidence: %.3f, tables found: %s",
        ocr_output.avg_confidence,
        ocr_output.has_table_regions,
    )

    if args.ocr_only:
        print("\n=== OCR TEXT ===")
        print(ocr_output.to_prompt_text())
        return

    # --- Extraction ---
    logger.info("Running LLM extraction with %s...", args.provider)
    extractor = create_extractor(args.provider)
    ocr_text = ocr_output.to_prompt_text()
    extraction = extractor.extract(ocr_text)
    extraction_dict = extraction.model_dump()

    extraction_path = args.output_dir / "extraction.json"
    extraction_path.write_text(json.dumps(extraction_dict, indent=2, default=str, ensure_ascii=False))
    logger.info("Extraction written to %s", extraction_path)

    # --- Validation ---
    logger.info("Running validation...")
    validation = validate_extraction(extraction, ocr_output.avg_confidence)

    validation_path = args.output_dir / "validation.json"
    validation_data = {
        "confidence_score": validation.confidence_score,
        "needs_review": validation.needs_review,
        "summary": validation.summary,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "skipped": c.skipped,
                "detail": c.detail,
            }
            for c in validation.checks
        ],
    }
    validation_path.write_text(json.dumps(validation_data, indent=2))
    logger.info("Validation written to %s", validation_path)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Invoice:      {extraction.invoice_number}")
    print(f"Supplier:     {extraction.supplier_name}")
    print(f"Client:       {extraction.client_name}")
    print(f"Total:        {extraction.total_incl_vat} {extraction.currency}")
    print(f"Line items:   {len(extraction.line_items)}")
    print(f"Confidence:   {validation.confidence_score:.1%}")
    print(f"Needs review: {validation.needs_review}")
    print(f"Validation:   {validation.summary}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
