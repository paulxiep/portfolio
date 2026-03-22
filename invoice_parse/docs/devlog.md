# Development Log

## Session 1: Design + Failure Mode Analysis + Infra Scaffolding

### Design phase
- Wrote initial architecture doc with service separation rationale, component designs, data models, scalability plan
- Wrote detailed plans for each service (ingestion, processing, output, dashboard) and infrastructure (local, cloud)

### Failure mode analysis
- Stress-tested every major design decision for failure modes
- Identified 20 concrete failure modes across OCR fidelity (FM-1.x), queue semantics (FM-2.x), service separation (FM-3.x), multi-tenancy (FM-4.x), validation (FM-5.x), schema design (FM-6.x), LLM fallback (FM-7.x), confidence scoring (FM-8.x), local-cloud equivalence (FM-9.x), state machine (FM-10.x), Telegram integration (FM-11.x), Excel output (FM-12.x), and cross-cutting concerns (FM-CC.x)
- Updated all 6 plan documents with mitigations — schema changes, prompt engineering, state machine expansions, queue semantics, idempotency, delivery tracking, golden test set

### Structural decisions for production-readiness

| Decision | Rationale |
|----------|-----------|
| Independent crates, no Cargo workspace | Each service is a standalone repo in production. Path deps mimic registry deps. No root Cargo.toml. |
| Per-service .venv (conda) | Mimics production where each service has its own dependency environment |
| `libs/` for shared packages | Publishable to private crate registry / PyPI in production |
| `services/` with own Dockerfile each | Each service is independently deployable |
| Shared `migrations/`, `config/`, `seed.sql` | Single source of truth for DB schema and config structure |
| psycopg (v3) over psycopg2 | psycopg2-binary doesn't build on MSYS2/MinGW; psycopg3 works with `postgresql+psycopg` dialect |

### Infra local implementation
- Docker Compose: Postgres 18, Redis 8
- SQL migration with full job status enum (16 states), jobs + tenants tables, indexes, updated_at triggers
- Seed data: two test tenants for isolation testing
- Config: local.yaml (Postgres + local FS + Redis Streams) and production.yaml placeholder
- Python shared library (`libs/shared-py/`): config loader, Pydantic models (InvoiceExtraction, LineItem with field validators for ISO 4217 currency + YYYY-MM-DD dates), SQLAlchemy ORM with state machine transition validation, BlobStore ABC + LocalFsBlobStore (path traversal protection), MessageQueue ABC + RedisStreamQueue (Streams with consumer groups)
- Rust shared crate (`libs/shared-rs/`): mirror of Python types with serde/sqlx derives, config loader, PgPool factory, state machine transitions, BlobStore trait + LocalFsBlobStore, MessageQueue trait + RedisStreamQueue
- Service placeholders: ingestion (Rust), processing (Python), output (Rust), dashboard (Python/Streamlit) — each with Cargo.toml/pyproject.toml + Dockerfile
- Tests: 7 Python blob store tests pass, 3 Rust blob store tests pass, all Rust crates compile

### Version choices (adoption-driven, not latest-chasing)

| Component | Version | Signal |
|-----------|---------|--------|
| Postgres | 18 | Latest stable |
| Redis | 8 | Latest stable |
| pydantic | 2.12 | 710M downloads/mo |
| SQLAlchemy | 2.0 | 331M downloads/mo |
| redis-py | 7.3 | 176M downloads/mo |
| psycopg | 3.3 | Newer but works; psycopg2-binary incompatible with MSYS2 |
| sqlx (Rust) | 0.8 | 20M recent downloads, 0.9 is alpha |
| redis (Rust) | 1.0 | Stable, adopted |
| serde_yaml | 0.9 | 48M downloads; replacement serde_yml at 0.0.12 not adopted |

---

## Session 2: Processing Service — OCR Implementation + Pipeline Scaffolding

### Code written
Scaffolded the full processing pipeline across 4 modules + CLI entry point. **OCR module tested end-to-end; extraction, validation, and full pipeline not yet tested.**

| Module | Status |
|--------|--------|
| `ocr.py` | **Tested** — hybrid PPStructureV3 + raw PaddleOCR with dynamic spatial clustering |
| `extraction.py` | Written — GeminiExtractor with structured JSON output. Not yet tested (needs API key). |
| `validation.py` | Written — VAT math, line items sum, dates, confidence scoring. Not yet tested. |
| `worker.py` | Written — pipeline orchestration + queue consumer. Not yet tested. |
| `cli.py` | **Tested** — OCR-only mode works. Full pipeline mode not yet tested. |

### Design decisions

| Decision | Rationale |
|----------|-----------|
| PyMuPDF over pdf2image | Pure Python wheel — no system `poppler` dependency. Simpler on Windows and Docker. |
| PPStructureV3 (not legacy PPStructure) | PaddleOCR 3.x API: `parsing_res_list` with `block_label`/`block_content`. |
| Hybrid OCR: PPStructureV3 + raw PaddleOCR | PPStructureV3 misses page regions (see below). Raw OCR with coordinates fills gaps. |
| Dynamic spatial clustering for raw OCR | Row grouping by y-gap detection, region dividers by large y-gaps. No hardcoded thresholds. |
| `google-genai` SDK (not `google-generativeai`) | New unified SDK is GA. Uses `response_json_schema` + `response_mime_type`. |
| `gemini-2.5-flash` as default model | Configurable via `GEMINI_MODEL` env var. |
| `validate_extraction` takes `ocr_avg_confidence: float` not full `OcrOutput` | Validation module stays pure — no dependency on OCR data structures. |
| CLI writes directly to output dir (not via BlobStore) | BlobStore enforces UUID path segments. CLI uses human-readable paths. |
| `run_pipeline()` with optional `db_session_factory` | Same function serves worker (with DB) and CLI (without). |
| HTML table parsing via stdlib `html.parser` | No BeautifulSoup dependency. |
| Confidence = 50% checks + 30% OCR + 20% completeness | Deterministic, auditable. No LLM self-report per FM-8.1. |

### Version choices (latest stable as of Mar 2026)

| Component | Version | Source |
|-----------|---------|--------|
| paddleocr | 3.4.0 | PyPI, Jan 2026 |
| paddlepaddle | 3.2.2 (pinned <3.3) | PyPI; 3.3.0 has PIR/oneDNN CPU bug |
| PyMuPDF | 1.27.2 | PyPI, Mar 2026 |
| google-genai | 1.56.0 | PyPI, Mar 2026 |
| anthropic | 0.86.0 | PyPI, Mar 2026 |
| openai | 2.29.0 | PyPI, Mar 2026 |
| Pillow | 12.1.1 | PyPI, Feb 2026 |

### OCR testing and iteration

**PaddlePaddle 3.3.0 CPU inference bug:**
- PaddlePaddle 3.3.0 introduced a PIR (Paddle Intermediate Representation) executor that breaks oneDNN CPU inference: `ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`
- Affects both Windows and Linux Docker. `FLAGS_use_mkldnn=0` and `paddle.set_flags()` do not fix it — the C++ runtime ignores the Python-level flag.
- [GitHub issue #77340](https://github.com/PaddlePaddle/Paddle/issues/77340). Fix: pin `paddlepaddle>=3.2.0,<3.3`.

**PPStructureV3 misses page regions:**
- On sample invoice, PPStructureV3 detects 5 blocks: 3 titles + 2 tables (header table + Job table). The Misc section (3 line items), VAT summary, and bottom totals are undetected — the layout model cuts off at ~84% page height.
- PPStructureV3 bbox for the Job table (y=1645–2938) is larger than the content it actually extracted, swallowing the Misc section spatially without extracting it.

**Separation of concerns refactor — ocr.py / table_extract.py / extraction.py:**

Initial implementation had `ocr.py` doing raw OCR + PPStructureV3 + spatial clustering + LLM prompt formatting — too many concerns in one module. Refactored into:

| Module | Responsibility |
|--------|---------------|
| `ocr.py` | Pure raw OCR. PDF/image → `RawOcrOutput` with `list[OcrLine(text, x, y)]` per page. No table logic. |
| `table_extract.py` | Table reconstruction from raw OCR or layout models. `TableExtractor` ABC with two strategies: `SpatialClusterExtractor` (gap-based coordinate clustering) and `PPStructureExtractor` (PPStructureV3 layout detection). Both output `TableExtractionOutput`. |
| `extraction.py` | LLM extraction. Receives **either or both** raw OCR and table extraction. Prompt builder composes whichever inputs are available with clear section headers. |

Design rationale:
- LLM is likely more robust at inferring table structure from coordinates than hardcoded clustering, especially across diverse invoice layouts. Separating concerns allows swapping strategies or sending raw coordinates directly to LLM.
- `extraction.py` accepts `raw_ocr: RawOcrOutput | None` and `table_extraction: TableExtractionOutput | None`. CLI flags control which inputs to provide: `--table-method spatial_cluster|ppstructure|none`, `--raw-only`.
- `SpatialClusterExtractor`: dynamic gap detection for row grouping (gap > 2× median y-gap = new row) and region boundaries (gap > 3× median row gap = separator). No hardcoded pixel thresholds.
- PPStructureV3 bboxes unreliable as coverage claims (Job table bbox swallowed Misc section without extracting it). Each extractor is self-contained rather than trying to filter/merge.

**Image format support:**
- `process_ocr()` now accepts PDF and image files (PNG, JPG, WEBP). Detects format by filename extension or PDF magic bytes.
- Tested on 17 invoice files (1 PDF, 16 images in various formats). All processed successfully, 37–106 OCR lines per file.

**Dockerfile iterations:**
- `python:3.12-slim` base requires `libgl1 libglib2.0-0 libgomp1` for OpenCV + PaddlePaddle.
- `paddlex[ocr]` extra required for PPStructureV3 (not installed by `paddleocr` alone).
- Build context must be project root (`invoice_parse/`) for shared lib access. Docker Compose profile `processing` added.
- `.env.sample` for API keys; `python-dotenv` loads `.env` in CLI and worker entry points.

### End-to-end extraction testing

Tested full pipeline (OCR → spatial cluster table extraction → Gemini 2.5 Flash → validation) on 3 invoices:

| Invoice | Items | Confidence | Validation | Notes |
|---------|-------|------------|------------|-------|
| `sample_invoice.pdf` (CZK, Job+Misc) | 7/7 correct | 100% | All passed | Dates `2.7.` vs `12.7.` — OCR boundary splits `1` into shift label. LLM correctly reads what OCR gives it. |
| `invoice-example.webp` (EUR, spices) | 4/4 correct | 96% | All passed | Clean extraction. European format handled. |
| `Purchase-Invoice.webp` (INR, electronics) | 3 extracted | 68% | **Failed: vat_sum, line_items_sum** | `total_incl_vat` misread (2.1M vs expected 504K). Power Strips tariff×qty math wrong. Correctly flagged `needs_review`. |

Observations:
- Validation catches real problems: VAT arithmetic and line items sum failures correctly flag bad extractions for review.
- OCR text boundary issues (date digits absorbed into adjacent text) are an inherent limitation of line-level OCR — not fixable at extraction layer.
- Indian invoice with large numbers (INR 280,000+) had OCR/extraction errors on totals. Likely OCR misread on dense number-heavy layout.
