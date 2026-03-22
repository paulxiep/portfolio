# Invoice Document Parsing System

Multi-tenant, async document parsing pipeline that extracts structured data from invoice documents in any language. Accepts invoices via email (IMAP polling), extracts 10+ key fields and line items using OCR + LLM, validates the output, generates Excel, and delivers the result back to the sender.

## Design Reasoning

**Problem framing.** The requirements are intentionally high-level. The architecture doc (`docs/architecture.md`) documents every ambiguity and the decision made: async vs sync, single vs multi-tenant, OCR+text LLM vs vision model, confidence handling, schema flexibility. 20 failure modes identified and mitigated before writing code.

**Data reasoning.** Invoices have structural patterns (header fields, sectioned line items) that layout-aware OCR preserves. The pipeline extracts spatial coordinates, clusters them into table rows by y-gap analysis, and feeds structured text to the LLM — 10-50x cheaper than image tokens. Validation catches real extraction errors (VAT arithmetic, line item sums) with adaptive tolerance.

**Modeling choices.** Gemini Flash for cost-efficient structured extraction with provider abstraction (Claude, OpenAI ready to swap in). Confidence scoring from validation signals (50% checks + 30% OCR + 20% completeness), not LLM self-report. Deterministic pipeline with conditional fallbacks rather than agentic loops — invoice extraction is structured and repeatable.

**Code clarity.** Each module has a single responsibility: `ocr.py` (raw OCR), `table_extract.py` (table reconstruction), `extraction.py` (LLM), `validation.py` (business rules), `excel_gen.rs` (output formatting). Adapter pattern for infrastructure (blob store, queue) with identical interfaces across Rust and Python.

**Accuracy–simplicity trade-offs.** Hybrid OCR (PPStructureV3 + spatial clustering) compensates for layout model coverage gaps without adding complexity to downstream modules. Section-aware column layouts in Excel output (Job vs Misc columns) preserve fidelity without sparse tables. Cache-by-hash avoids LLM cost on repeat inputs without complicating the pipeline.

**First steps.** Infrastructure-first: Postgres + Redis + adapters + state machine before any ML code. This meant the processing pipeline integrated into a working system immediately, not as a standalone script requiring rework.

## Load Testing

```bash
# 1. Start infra (Postgres + Redis) + application services (processing, output, dashboard)
docker compose -f infra/docker-compose.yaml --profile app up -d

# 2. Enqueue invoices (1 round = 17 jobs)
docker compose -f infra/docker-compose.yaml --profile ingest run --rm ingest

# Load test: repeat N times
# PowerShell:
1..100 | % { docker compose -f infra/docker-compose.yaml --profile ingest run --rm ingest }
# Bash:
for i in $(seq 1 100); do docker compose -f infra/docker-compose.yaml --profile ingest run --rm ingest; done

# 3. Monitor at http://localhost:8501
```

Each round enqueues 17 jobs (one per invoice in `invoices/`). With `PIPELINE_CACHE=1` on the processing worker, only the first run per unique file hits the LLM — subsequent rounds reuse cached results.

```bash
# Tear down everything (containers + volumes)
docker compose -f infra/docker-compose.yaml --profile app --profile ingest down -v
```

## Quick Start (Docker)

Only requires **Docker** and a **Gemini API key**.

```bash
# 1. Set API key
cp .env.sample .env
# Edit .env: set GEMINI_API_KEY=your_key

# 2. Start everything (infra + all services)
docker compose -f infra/docker-compose.yaml --profile app up -d

# 3. Dashboard at http://localhost:8501
```

Postgres auto-runs migrations and seed on first start. A `model-init` container downloads PaddleOCR models (~500MB) into a persistent volume on first run — subsequent starts are instant. Processing worker, output worker, and dashboard start automatically.

OCR models default to **mobile** tier in Docker (faster). Override in `.env`:
```
OCR_DET_MODEL=PP-OCRv5_server_det      # or PP-OCRv5_mobile_det (default)
OCR_REC_MODEL=en_PP-OCRv5_server_rec   # or en_PP-OCRv5_mobile_rec (default)
```

## Quick Start (Local Development)

### Prerequisites
- Docker (for Postgres + Redis)
- Rust 1.85+, Python 3.12+
- Gemini API key

```bash
# 1. Start infra
docker compose -f infra/docker-compose.yaml up -d

# 2. Install Python deps
pip install -e libs/shared-py && pip install -e services/processing

# 3. Set API key
cp .env.sample .env   # edit: GEMINI_API_KEY=your_key

# 4. Single invoice (CLI)
GEMINI_API_KEY=xxx python -m invoice_processing.cli invoices/sample_invoice.pdf -v

# 5. Full pipeline (queue mode) — all from project root
cargo run --manifest-path services/ingestion/Cargo.toml -- invoices/   # ingest
PIPELINE_CACHE=1 python -m invoice_processing.worker                   # process
cargo run --manifest-path services/output/Cargo.toml                   # excel gen
streamlit run services/dashboard/dashboard/app.py                      # dashboard

# Monitor
redis-cli XLEN queue:a && redis-cli XLEN queue:b
```

## Architecture

```
          Rust                    Python                   Rust
          ────                    ──────                   ────

┌───────────────┐  Queue A  ┌──────────────────┐  Queue B  ┌───────────────┐
│   INGESTION   │──────────▶│    PROCESSING    │─────────▶│    OUTPUT     │
│───────────────│           │──────────────────│          │───────────────│
│ • IMAP poll   │           │ • OCR            │          │ • Excel gen   │
│ • Auth/tenant │           │ • LLM extraction │          │ • Send reply  │
│ • Enqueue job │           │ • Validation     │          │               │
└───────────────┘           └──────────────────┘          └───────────────┘
                                    │
                              Postgres + Blob
```

**Services are separated by scaling needs, not by function:**

| Service | Language | Why |
|---------|----------|-----|
| Ingestion | Rust | I/O-bound, handles burst connections with minimal memory |
| Processing | Python | CPU/API-bound; OCR + LLM libraries are Python-native |
| Output | Rust | I/O-bound delivery; Excel gen is stable plumbing that rarely changes |

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate OCR from LLM** | Text tokens are 10-50x cheaper than image tokens. Self-hosted PaddleOCR → text LLM (Gemini Flash). |
| **Deterministic pipeline, not agentic** | Invoice extraction is structured and repeatable. Conditional fallbacks (vision model, provider failover, VAT derivation) provide adaptability without LLM decision loops. |
| **Queue-based async** | Redis Streams locally, SQS in production. Workers scale independently. Queue absorbs bursts. |
| **Confidence from validation, not LLM self-report** | 50% validation checks + 30% OCR confidence + 20% field completeness. Deterministic and auditable. |
| **Adapter pattern** | `BlobStore` (local FS / S3) and `MessageQueue` (Redis / SQS) abstractions. Same code, swap config. |
| **Layout-aware OCR** | PaddleOCR with spatial clustering preserves table row/column structure critical for invoices. |
| **LLM result cache** | SHA-256 hash of input file → cached extraction. First run hits LLM, subsequent runs are free. Enables load testing without API cost. |

## Storage

| Store | Purpose | Local | Cloud |
|-------|---------|-------|-------|
| Blob storage | Documents + intermediates (input.pdf, ocr.json, extraction.json, output.xlsx) | Local filesystem | S3 |
| Postgres | Job state machine (16 states), extraction data (JSONB), tenant config | Docker container | RDS |
| Redis | Message queues (Streams with consumer groups) | Docker container | ElastiCache / SQS |

## Scalability

| Volume | Approach |
|--------|----------|
| ~100/day | Single worker |
| ~1k-10k/day | Multiple workers, queue-based horizontal scaling |
| ~100k+/day | Auto-scaling workers, LLM batch API, multiple API keys |

**Bottleneck analysis:** OCR (1-5s) + LLM (1-5s) dominate at 95%+ of total latency. Inter-service network overhead is <1%. Queue absorbs bursts; workers catch up. Excel gen (~20ms) and delivery (~50ms) are negligible.

**Horizontal scaling:** Processing is the only service worth scaling. Redis Streams consumer groups handle this natively — deploy N processing workers and each gets different messages with no coordination. In production (ECS/K8s), auto-scale processing workers based on queue depth (`XLEN queue:a`). Ingestion and output are I/O-bound Rust services that handle thousands of jobs/sec on a single instance.

## Project Structure

```
invoice_parse/
├── services/
│   ├── ingestion/        Rust — IMAP poll, file ingest, job creation
│   ├── processing/       Python — OCR, table extraction, LLM, validation
│   ├── output/           Rust — Excel generation, delivery (Telegram/email planned)
│   └── dashboard/        Python/Streamlit — job monitoring
├── libs/
│   ├── shared-rs/        Rust — models, DB, blob store, queue adapters
│   └── shared-py/        Python — models, DB, blob store, queue adapters
├── config/               YAML config (local.yaml, docker.yaml, production.yaml)
├── infra/                Docker Compose (Postgres, Redis)
├── migrations/           SQL schema
├── invoices/             Sample invoice files for testing
└── docs/                 Architecture, service plans, devlog
```

**Mono-repo for POC, multi-repo in production.** Each service under `services/` has its own `Cargo.toml` or `pyproject.toml`, `Dockerfile`, and dependency closure — no Cargo workspace, no shared virtualenv. Shared libraries under `libs/` use path dependencies locally and would publish to a private crate registry / PyPI in production. This means any service can be extracted to its own repo and CI pipeline without restructuring.

## Key Technical Choices

| Component | Version | Rationale |
|-----------|---------|-----------|
| PaddleOCR | 3.4 | Layout-aware, multilingual, self-hosted |
| Gemini 3.1 Flash Lite | Preview | Structured JSON output, cost-efficient. Provider-swappable (Claude, OpenAI stubs ready). |
| rust_xlsxwriter | 0.94 | Native Rust, no external deps, standard .xlsx |
| Redis Streams | 8.x | Consumer groups, message reclaim on crash, local-cloud equivalent to SQS |
| Postgres | 18 | JSONB for flexible extraction data, enum for state machine |
| SQLAlchemy / sqlx | 2.0 / 0.8 | Mature ORMs matching each language |
