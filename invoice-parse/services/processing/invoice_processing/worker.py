"""Queue consumer loop — pulls from Queue A, runs pipeline, publishes to Queue B.

Responsibilities:
- Orchestrate OCR → table extraction → LLM extraction → validation
- Manage DB state transitions (optional — skipped in CLI mode)
- Queue consumption with graceful shutdown

All domain logic is delegated to ocr, table_extract, extraction, and validation modules.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
from pathlib import Path

from invoice_shared.adapters.blob_store import BlobStore
from invoice_shared.adapters.factory import create_blob_store, create_queue
from invoice_shared.adapters.queue import MessageQueue
from invoice_shared.config import load_config
from invoice_shared.db import get_session, session_factory, transition_job
from invoice_shared.models import (
    InvoiceExtraction,
    JobStatus,
    QueueAMessage,
    QueueBMessage,
)

from .extraction import LLMExtractor, create_extractor
from .ocr import process_ocr
from .table_extract import TableExtractor, create_table_extractor
from .validation import ValidationCheck, ValidationResult, validate_extraction

logger = logging.getLogger(__name__)

QUEUE_A_TOPIC = "queue:a"
QUEUE_B_TOPIC = "queue:b"


# --- Pipeline result cache ---
# Caches extraction + validation by input file hash (SHA-256).
# Avoids re-running OCR + LLM on repeated load tests with the same invoices.
# Cache lives on local filesystem (not blob storage, which enforces UUID paths).


def _cache_dir() -> Path:
    """Return the cache directory, creating it if needed."""
    base = Path(os.environ.get("PIPELINE_CACHE_DIR", "./data/pipeline_cache"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cache_key(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def _load_cache(pdf_bytes: bytes) -> tuple[dict, ValidationResult] | None:
    """Try to load cached extraction + validation for this input file."""
    cache_file = _cache_dir() / f"{_cache_key(pdf_bytes)}.json"
    if not cache_file.exists():
        return None
    try:
        cached = json.loads(cache_file.read_text())
        extraction_dict = cached["extraction"]
        validation = ValidationResult(
            checks=[ValidationCheck(**c) for c in cached["validation"]["checks"]],
            confidence_score=cached["validation"]["confidence_score"],
            needs_review=cached["validation"]["needs_review"],
            summary=cached["validation"]["summary"],
        )
        logger.info("Cache hit for input hash %s", _cache_key(pdf_bytes)[:12])
        return extraction_dict, validation
    except Exception:
        logger.warning("Cache read failed, running full pipeline", exc_info=True)
        return None


def _save_cache(pdf_bytes: bytes, extraction_dict: dict, validation: ValidationResult) -> None:
    """Persist extraction + validation result keyed by input hash."""
    cache_file = _cache_dir() / f"{_cache_key(pdf_bytes)}.json"
    payload = {
        "extraction": extraction_dict,
        "validation": {
            "confidence_score": validation.confidence_score,
            "needs_review": validation.needs_review,
            "summary": validation.summary,
            "checks": [
                {"name": c.name, "passed": c.passed, "skipped": c.skipped, "detail": c.detail}
                for c in validation.checks
            ],
        },
    }
    cache_file.write_text(json.dumps(payload, default=str, ensure_ascii=False))
    logger.info("Cached result for input hash %s", _cache_key(pdf_bytes)[:12])


# --- Pipeline orchestration ---


def run_pipeline(
    pdf_bytes: bytes,
    job_id: str,
    tenant_id: str,
    blob_store: BlobStore,
    db_session_factory=None,
    extractor: LLMExtractor | None = None,
    table_extractor: TableExtractor | None = None,
    use_cache: bool = False,
) -> tuple[dict, ValidationResult]:
    """Run the full OCR → table extraction → LLM extraction → validation pipeline.

    Args:
        pdf_bytes: Raw PDF file content.
        job_id: Job UUID.
        tenant_id: Tenant UUID.
        blob_store: Where to write intermediate artifacts.
        db_session_factory: SQLAlchemy session factory. None to skip DB transitions.
        extractor: LLM extractor instance. Defaults to Gemini.
        table_extractor: Table extraction strategy. Defaults to SpatialCluster.

    Returns:
        (extraction_dict, validation_result)
    """
    blob_prefix = f"{tenant_id}/{job_id}"

    def _transition(status: JobStatus) -> None:
        if db_session_factory is None:
            return
        with get_session(db_session_factory) as session:
            transition_job(session, job_id, status)

    # --- Cache check ---
    if use_cache:
        cached = _load_cache(pdf_bytes)
        if cached is not None:
            extraction_dict, validation = cached
            # Write artifacts to job-specific blob path so downstream sees them
            blob_store.put(
                f"{blob_prefix}/extraction.json",
                json.dumps(extraction_dict, default=str, ensure_ascii=False).encode(),
            )
            # Fast-forward through state transitions
            for status in (
                JobStatus.OCR_PROCESSING, JobStatus.OCR_DONE,
                JobStatus.EXTRACTING, JobStatus.EXTRACTED,
                JobStatus.VALIDATING,
            ):
                _transition(status)
            target = JobStatus.NEEDS_REVIEW if validation.needs_review else JobStatus.DONE
            if db_session_factory is not None:
                with get_session(db_session_factory) as session:
                    job = transition_job(session, job_id, target)
                    job.extraction_data = extraction_dict
                    job.confidence_score = validation.confidence_score
            return extraction_dict, validation

    # --- Substep 1: Raw OCR ---
    _transition(JobStatus.OCR_PROCESSING)
    raw_ocr, images = process_ocr(pdf_bytes)
    blob_store.put(
        f"{blob_prefix}/raw_ocr.json",
        json.dumps(raw_ocr.to_dict(), ensure_ascii=False).encode(),
    )
    _transition(JobStatus.OCR_DONE)
    logger.info("OCR complete — %d page(s), %d lines",
                len(raw_ocr.pages), sum(len(p.lines) for p in raw_ocr.pages))

    # --- Substep 1b: Table extraction ---
    if table_extractor is None:
        table_extractor = create_table_extractor("spatial_cluster")
    table_extraction = table_extractor.extract(raw_ocr, images)
    blob_store.put(
        f"{blob_prefix}/table_extraction.json",
        json.dumps(table_extraction.to_dict(), ensure_ascii=False).encode(),
    )
    logger.info("Table extraction complete (method=%s)", table_extraction.method)

    # --- Substep 2: LLM extraction ---
    _transition(JobStatus.EXTRACTING)
    if extractor is None:
        extractor = create_extractor("gemini")
    extraction = extractor.extract(raw_ocr=raw_ocr, table_extraction=table_extraction)
    extraction_dict = extraction.model_dump()
    blob_store.put(
        f"{blob_prefix}/extraction.json",
        json.dumps(extraction_dict, default=str, ensure_ascii=False).encode(),
    )
    _transition(JobStatus.EXTRACTED)
    logger.info("Extraction complete — %d line items", len(extraction.line_items))

    # --- Substep 3: Validation ---
    _transition(JobStatus.VALIDATING)
    ocr_confidence = 1.0  # TODO: derive from table extraction quality signals
    validation = validate_extraction(extraction, ocr_confidence)

    target = JobStatus.NEEDS_REVIEW if validation.needs_review else JobStatus.DONE
    if db_session_factory is not None:
        with get_session(db_session_factory) as session:
            job = transition_job(session, job_id, target)
            job.extraction_data = extraction_dict
            job.confidence_score = validation.confidence_score
    logger.info("Validation: %s (confidence=%.3f)", validation.summary, validation.confidence_score)

    # --- Cache save ---
    if use_cache:
        _save_cache(pdf_bytes, extraction_dict, validation)

    return extraction_dict, validation


# --- Queue message processing ---


def process_message(
    message: QueueAMessage,
    blob_store: BlobStore,
    queue: MessageQueue,
    db_sf,
) -> None:
    """Process a single queue message end-to-end."""
    pdf_bytes = blob_store.get(message.blob_path)

    use_cache = os.environ.get("PIPELINE_CACHE", "").strip() == "1"
    extraction_dict, validation = run_pipeline(
        pdf_bytes=pdf_bytes,
        job_id=message.job_id,
        tenant_id=message.tenant_id,
        blob_store=blob_store,
        db_session_factory=db_sf,
        use_cache=use_cache,
    )

    queue_b_msg = QueueBMessage(
        job_id=message.job_id,
        tenant_id=message.tenant_id,
        extraction=InvoiceExtraction.model_validate(extraction_dict),
        confidence_score=validation.confidence_score,
        output_blob_path=f"{message.tenant_id}/{message.job_id}/output.xlsx",
        source_channel=message.source_channel,
        source_identifier=message.source_identifier,
    )
    queue.publish(QUEUE_B_TOPIC, queue_b_msg.model_dump(mode="json"))
    logger.info("Published to Queue B for job %s", message.job_id)


# --- Queue consumer loop ---


def run_worker() -> None:
    """Start the queue consumer loop. Entry point for production."""
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config()
    blob_store = create_blob_store(config)
    queue = create_queue(config)
    db_sf = session_factory(config)

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received, finishing current job...")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("Worker started, consuming from %s", QUEUE_A_TOPIC)
    while not shutdown:
        messages = queue.consume(QUEUE_A_TOPIC, count=1, block_ms=5000)
        for msg_id, msg_data in messages:
            try:
                message = QueueAMessage.model_validate(msg_data)
                logger.info("Processing job %s", message.job_id)
                process_message(message, blob_store, queue, db_sf)
                queue.ack(QUEUE_A_TOPIC, msg_id)
                logger.info("Job %s completed", message.job_id)
            except Exception:
                logger.exception("Failed to process job from message %s", msg_id)

    logger.info("Worker shut down gracefully")


if __name__ == "__main__":
    run_worker()
