"""CLI for testing the invoice processing pipeline on a single PDF.

Usage:
    python -m invoice_processing.cli path/to/invoice.pdf [--provider gemini] [--output-dir ./output] [--ocr-only] [-v]
    python -m invoice_processing.cli path/to/invoice.pdf --table-method ppstructure
    python -m invoice_processing.cli path/to/invoice.pdf --table-method spatial_cluster
    python -m invoice_processing.cli path/to/invoice.pdf --raw-only  (send only raw OCR to LLM)

No Redis or Postgres required. Uses local filesystem and skips DB state transitions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .ocr import process_ocr
from .table_extract import create_table_extractor
from .extraction import create_extractor
from .validation import validate_extraction


def main() -> None:
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
        "--table-method",
        default="spatial_cluster",
        choices=["spatial_cluster", "ppstructure", "none"],
        help="Table extraction method (default: spatial_cluster)",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Send only raw OCR to LLM (no table extraction)",
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
        help="Run only OCR + table extraction, skip LLM and validation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    pdf_bytes = args.pdf_path.read_bytes()
    logger.info("Read %d bytes from %s", len(pdf_bytes), args.pdf_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- OCR ---
    logger.info("Running OCR...")
    raw_ocr, images = process_ocr(pdf_bytes, filename=args.pdf_path.name)

    ocr_path = args.output_dir / "raw_ocr.json"
    ocr_path.write_text(json.dumps(raw_ocr.to_dict(), indent=2, ensure_ascii=False))
    logger.info("Raw OCR written to %s", ocr_path)

    # --- Table extraction ---
    table_extraction = None
    if not args.raw_only and args.table_method != "none":
        logger.info("Running table extraction (%s)...", args.table_method)
        table_extractor = create_table_extractor(args.table_method)
        table_extraction = table_extractor.extract(raw_ocr, images)

        table_path = args.output_dir / "table_extraction.json"
        table_path.write_text(json.dumps(table_extraction.to_dict(), indent=2, ensure_ascii=False))
        logger.info("Table extraction written to %s", table_path)

    if args.ocr_only:
        if table_extraction:
            print("\n=== TABLE EXTRACTION ===")
            print(table_extraction.to_prompt_text())
        print("\n=== RAW OCR ===")
        for page in raw_ocr.pages:
            for line in page.lines:
                print(f"  x={line.x:5d} y={line.y:5d}  {line.text}")
        return

    # --- LLM Extraction ---
    logger.info("Running LLM extraction with %s...", args.provider)
    extractor = create_extractor(args.provider)
    extraction = extractor.extract(
        raw_ocr=raw_ocr,
        table_extraction=table_extraction,
    )
    extraction_dict = extraction.model_dump()

    extraction_path = args.output_dir / "extraction.json"
    extraction_path.write_text(json.dumps(extraction_dict, indent=2, default=str, ensure_ascii=False))
    logger.info("Extraction written to %s", extraction_path)

    # --- Validation ---
    logger.info("Running validation...")
    ocr_confidence = 1.0 if table_extraction else 0.5
    validation = validate_extraction(extraction, ocr_confidence)

    validation_path = args.output_dir / "validation.json"
    validation_data = {
        "confidence_score": validation.confidence_score,
        "needs_review": validation.needs_review,
        "summary": validation.summary,
        "checks": [
            {"name": c.name, "passed": c.passed, "skipped": c.skipped, "detail": c.detail}
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
