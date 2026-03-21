# Invoice Document Parsing System
## Architecture & Design Document

---

## 1. Problem Framing

**Given requirements:**
- Accept invoice documents via Telegram or email (any language)
- Extract 10 key fields + line items
- Validate extracted data
- Return structured output (Excel) via same channel

**What we're actually building:**
A multi-tenant, scalable document parsing pipeline that separates concerns by scaling needs, uses cost-efficient AI components, and maintains local-cloud equivalence for development.

---

## 2. Scope Refinement

The requirements are intentionally high-level. Senior engineering means making decisions on ambiguities.

| Ambiguity | Decision | Rationale |
|-----------|----------|-----------|
| Sync or async response? | **Async (queue-based)** | Enables batching, retry, scale |
| Single user or multiple? | **Multiple users** | Handle concurrency, tag jobs by source (chat_id/email) |
| OCR in LLM or separate? | **Separate OCR → text LLM** | Cost efficiency, batch capability, self-host option |
| Vision model for extraction? | **Text-only LLM** | Image tokens 10-50x more expensive than text |
| Confidence handling? | **Extract always, flag low confidence** | Don't block; surface for review |
| Schema fixed or configurable? | **Configurable per tenant** | "10 fields" today, more tomorrow |

---

## 3. Target Fields

Based on sample invoice analysis:

| Field | Example |
|-------|---------|
| Supplier Name | myAgency Ltd |
| Supplier Address | 145 Foley Street, Mountjoy, Dublin |
| Client Name | RedBeast Energy |
| Client Address | High Street 123, Dublin |
| Invoice/Order Number | R15 |
| Invoice Date | 12.7.2023 - 13.7.2023 |
| Location | Royal Albert Hall, Kensington Gore, London |
| Total excl. VAT | 2,305.00 CZK |
| VAT Amount (+ rate) | 461.00 CZK (20%) |
| Total incl. VAT | 2,766.00 CZK |

**Line items** (variable length):
- Job entries: Date, Item, Quantity, Start, Finish, Hours, Tariff, Total
- Misc entries: Item, Quantity, Tariff, Total

---

## 4. Architecture Overview

### 4.1 Service Separation by Scaling Needs

```
              Rust                  Python                 Rust
              ────                  ──────                 ────

┌───────────────┐  Queue A  ┌──────────────────┐  Queue B  ┌───────────────┐
│   INGESTION   │──────────▶│    PROCESSING    │─────────▶│    OUTPUT     │
│───────────────│           │──────────────────│          │───────────────│
│ • Webhook     │           │ • OCR            │          │ • Excel gen   │
│ • Auth/tenant │           │ • LLM Extraction │          │ • Send reply  │
│ • Enqueue job │           │ • Validation     │          │               │
└───────────────┘           └──────────────────┘          └───────────────┘
                                    │
       Scale: connections           │           Scale: light, handle backpressure
                                    │ writes
       Scale: throughput            ▼
       (OCR + LLM both         ┌─────────────────────────────────────┐
        slow, per-doc)         │              STORAGE                │
                               │─────────────────────────────────────│
                               │  Blob Storage    │    Postgres      │
                               │  • input.pdf     │    • jobs        │
                               │  • ocr.json      │    • extractions │
                               │  • output.xlsx   │    • tenants     │
                               └────────┬────────────────────────────┘
                                        │ reads
                                        ▼
                               ┌─────────────────┐
                               │    DASHBOARD    │
                               │   (Streamlit)   │
                               │─────────────────│
                               │ • Job status    │
                               │ • Metrics       │
                               │ • Error rates   │
                               └─────────────────┘
```

### 4.2 Why This Grouping

| Service | Language | Scaling Need | Characteristics |
|---------|----------|--------------|-----------------|
| **Ingestion** | Rust (Axum) | Burst connections | Fast, stateless, memory-light |
| **Processing** | Python | Throughput | OCR + LLM + Validation sequential per-doc, both slow |
| **Output** | Rust | Backpressure | Decoupled so slow delivery doesn't block workers |
| **Dashboard** | Python (Streamlit) | N/A | Read-only monitoring, fast to build |

**Language choice rationale — two axes:**

| Axis | Rust (Ingestion, Output) | Python (Processing, Dashboard) |
|------|--------------------------|-------------------------------|
| **Performance** | I/O-bound, high concurrency, low memory | CPU/API-bound, bottlenecked by external calls anyway |
| **Change velocity** | Stable plumbing — logic defined by external contracts (Telegram API, Excel spec, queue schema). Rarely changes once working. | Rapidly iterated — prompt tuning, OCR config, validation rules, schema evolution. Fast feedback loop critical. |

### 4.3 Why Queues Over Airflow

The pipeline is **linear** (OCR → LLM → Validation), **event-driven** (invoice arrives → process it), and **independent per document** (no cross-document dependencies). This is exactly what queues are built for.

**Airflow was considered and rejected** because:
- Airflow is designed for **scheduled batch ETL** with complex DAGs, not high-volume event-driven processing
- Hundreds of small DAG runs (one per document) is an Airflow anti-pattern — scheduler polls DB, lags under load, UI becomes unusable
- Scheduler + webserver + DB run always, even when idle — wasteful for bursty workloads
- Event-driven triggering is awkward (external API call to trigger DAG, not its native model)

**Queues win here because:**
- Workers pull independently — add more workers = linear scaling, no central bottleneck
- SQS handles thousands of messages/second; queue absorbs bursts, workers catch up
- Workers are stateless — spin up/down with demand (ECS auto-scaling), zero cost when idle
- Event-driven processing is the native model

**The tradeoff:** Retry logic, state tracking, and monitoring must be built manually (Airflow provides these for free). This is acceptable because building basic retry + status dashboard is cheaper than fighting Airflow's scaling model at volume.

### 4.4 Network/IO Bottleneck Analysis

| Step | Time | Bound by |
|------|------|----------|
| Ingestion (receive file) | ~10-50ms | Network I/O |
| OCR | ~1-5s | CPU |
| LLM extraction | ~1-5s | External API |
| Validation | ~5-10ms | CPU |
| Excel generation | ~20-50ms | CPU |
| Send response | ~50-100ms | Network I/O |
| Microservice hops | ~10-40ms | Network overhead |

**Conclusion:** Network overhead between services is <1% of total latency. OCR and LLM API calls dominate at 95%+. Service separation is justified by independent scaling, not latency optimization.

---

## 5. Component Design

### 5.1 OCR Layer

**Decision: Self-hosted layout-aware OCR, not vision LLM**

| Approach | Cost | Latency | Control |
|----------|------|---------|---------|
| Vision LLM per-doc | Image tokens expensive | Synchronous | Limited |
| Self-hosted OCR → Text LLM | Text tokens 10-50x cheaper | OCR fast, LLM batchable | Full |

**Tool options:**

| Tool | Strength |
|------|----------|
| PaddleOCR PP-Structure | Layout detection, table extraction, multilingual |
| Surya | Document-focused, line segmentation |
| DocTR | Hugging Face backed |

**Critical for invoices:** Layout-aware extraction returns table regions with preserved row/column structure. Prevents:
```
Bad:  "Item Qty Widget A 5 Hostess 3"  (lost structure)
Good: [{row: ["Widget A", "5"]}, {row: ["Hostess", "3"]}]  (preserved)
```

### 5.2 Extraction Layer

**Model choice: Gemini Flash 3.0**

| Consideration | Decision |
|---------------|----------|
| Model | Gemini Flash 3.0 (or Claude Haiku 4.5, GPT-4o-mini) |
| Input | Text only (from OCR), not images |
| Output | Structured JSON via `response_schema` or function calling |
| Batching | Use batch API for cost optimization at scale |

**Prompt design:**
- Schema-driven extraction (provide JSON schema, get structured output)
- Language-agnostic (no explicit language handling needed)
- Few-shot examples for edge cases

### 5.3 Validation (within Processing)

**Two levels:**

1. **Schema validation** (Pydantic)
   - Required fields present
   - Types correct
   - Enum values valid

2. **Business logic validation**
   - VAT math: `total_excl_vat × vat_rate ≈ vat_amount`
   - Date sanity
   - Currency consistency
   - Line items sum to totals

**Confidence scoring:**
- LLM provides confidence per field (if supported)
- Validation failures lower confidence
- Low confidence → flagged for review (future: human-in-loop)

### 5.4 Output Layer

- Generate Excel with openpyxl or Rust equivalent (rust_xlsxwriter)
- Route response to original channel (Telegram bot reply / email reply)
- Store output artifact to blob storage

---

## 6. Data Model

### 6.1 Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           STORAGE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Blob (S3/GCS/local)              Postgres                        │
│   ───────────────────              ────────                        │
│   /{tenant_id}/                    jobs                            │
│     /{job_id}/                     ├─ id                           │
│       input.pdf                    ├─ tenant_id                    │
│       ocr_output.json              ├─ status (enum)                │
│       extraction.json              ├─ source_channel               │
│       output.xlsx                  ├─ source_identifier            │
│                                    ├─ confidence_score             │
│                                    ├─ created_at                   │
│                                    ├─ updated_at                   │
│                                    ├─ blob_paths (jsonb)           │
│                                    └─ extraction_data (jsonb)      │
│                                                                     │
│                                    tenants                         │
│                                    ├─ id                           │
│                                    ├─ name                         │
│                                    ├─ config (jsonb)               │
│                                    └─ rate_limit                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Job Status Flow

```
queued → ocr_processing → ocr_done → extracting → extracted → validating → done
                │                         │              │
                ▼                         ▼              ▼
            ocr_failed            extraction_failed  needs_review
```

### 6.3 Queue Message Schemas

**Queue A (Ingestion → Processing):**
```json
{
  "job_id": "uuid",
  "tenant_id": "uuid",
  "blob_path": "/{tenant_id}/{job_id}/input.pdf",
  "source_channel": "telegram|email",
  "source_identifier": "chat_id|email_address",
  "created_at": "iso8601"
}
```

**Queue B (Processing → Output):**
```json
{
  "job_id": "uuid",
  "tenant_id": "uuid",
  "extraction": { ... },
  "confidence_score": 0.92,
  "output_blob_path": "/{tenant_id}/{job_id}/output.xlsx",
  "source_channel": "telegram|email",
  "source_identifier": "chat_id|email_address"
}
```

---

## 7. Local ↔ Cloud Equivalence

Design for local development with 1:1 cloud migration.

| Component | Local (POC) | Cloud (AWS) |
|-----------|-------------|-------------|
| Ingestion | Rust container | ECS Fargate |
| Processing | Python container | ECS Fargate |
| Output | Rust container | ECS Fargate |
| Queue | Redis | SQS |
| Blob storage | Local filesystem | S3 |
| Database | SQLite | RDS Postgres |
| Dashboard | Streamlit local | App Runner |

**Abstraction layer:**
```
config/
├── local.yaml
└── production.yaml

adapters/
├── queue/
│   ├── interface.py    # abstract base
│   ├── redis.py
│   └── sqs.py
├── storage/
│   ├── interface.py
│   ├── local_fs.py
│   └── s3.py
```

**Local run:**
```bash
docker-compose up  # All services + Redis (SQLite for POC, no Postgres container needed)
```

---

## 8. Scalability Considerations

### 8.1 Volume Scaling

| Invoices/day | Approach | Bottleneck |
|--------------|----------|------------|
| ~100 | Single worker | None |
| ~1k-10k | Multiple workers, queue | LLM rate limits |
| ~100k+ | Auto-scaling workers, batching | Cost optimization |

**Mitigations:**
- Queue absorbs bursts
- Workers scale horizontally based on queue depth
- LLM batch API reduces per-request overhead
- Multiple API keys if rate-limited

### 8.2 Multi-Tenant Scaling

| Concern | Solution |
|---------|----------|
| Isolation | Tenant ID on every record, scoped queries |
| Fair usage | Per-tenant rate limiting at ingestion |
| Noisy neighbor | Separate queues or priority queues per tenant tier |
| Cost tracking | Token usage logged per tenant |
| Data privacy | Tenant-specific blob storage paths |

### 8.3 Bottlenecks & Solutions

| Bottleneck | Solution |
|------------|----------|
| LLM rate limits | Multiple provider keys, batch API, priority queues |
| Bursty traffic | Queue absorbs spikes, auto-scaling workers |
| OCR CPU | Horizontal scaling of processing workers |
| Cold starts (serverless) | Provisioned concurrency or always-on pool |
| Database writes | Write-behind caching, batch inserts |

---

## 9. What We're NOT Building (POC Scope)

| Out of Scope | Reason |
|--------------|--------|
| Human review UI | POC; would be next phase |
| Custom model fine-tuning | Flash is good enough; premature optimization |
| Real-time streaming | Invoices aren't latency-critical |
| On-prem LLM | Cost/complexity tradeoff not justified yet |
| Multi-page document handling | Can add later; most invoices are single page |

---

## 10. POC Implementation Plan

**Target: 3-4 hours working prototype**

| Phase | Time | Deliverable |
|-------|------|-------------|
| 1. Ingestion stub | 30 min | Telegram webhook receives file, logs it |
| 2. OCR integration | 45 min | PaddleOCR container, returns structured text |
| 3. LLM extraction | 60 min | Gemini Flash prompt, structured output |
| 4. Validation | 30 min | Schema + business rules |
| 5. Output | 30 min | Excel generation + Telegram reply |
| 6. Glue + queue | 30 min | Redis queue, end-to-end flow |
| 7. Dashboard | 30 min | Streamlit showing job status |

**POC simplifications:**
- SQLite instead of Postgres
- Local filesystem instead of blob storage
- Single tenant
- Polling instead of webhooks (ngrok for demo)

---

## 11. Production Readiness Checklist

- [ ] Multi-tenant isolation
- [ ] Retry + dead-letter handling
- [ ] Per-tenant rate limits
- [ ] Confidence scoring + low-confidence flagging
- [ ] Observability (cost, latency, accuracy per field)
- [ ] Schema versioning
- [ ] Secrets management
- [ ] CI/CD pipeline
- [ ] Monitoring alerts

---

## 12. Summary

**Key architectural decisions:**

1. **Separate OCR from LLM** — cost efficiency, text tokens are 10-50x cheaper than image tokens
2. **Layout-aware OCR** — preserves table structure critical for invoices
3. **Service split by scaling needs** — not by function, not by language
4. **Rust for I/O-bound services** — low memory, high concurrency
5. **Python where required** — OCR libraries, dashboard (Streamlit)
6. **Queue-based async** — decouples services, enables retry, absorbs bursts
7. **Local-cloud equivalence** — same containers, different config

**The senior MLE signal:**
- Problem framing over blind execution
- Trade-off reasoning documented
- Production concerns addressed in POC design
- Clear "what we're not building" boundaries