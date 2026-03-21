# Infrastructure: Contracts & Local Dev Environment

## Purpose
Define the abstraction boundary between environment-agnostic services and environment-specific implementations. Provide everything needed to start coding and running services locally.

**Build and stabilize first** — all other components depend on these interfaces.

---

## Tech Stack (Local/POC)

| Component | Local Implementation |
|-----------|---------------------|
| Database | SQLite |
| Blob storage | Local filesystem |
| Queue | Redis |
| Orchestration | Docker Compose |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Adapter pattern for all infrastructure | Services code against abstract interfaces; swap implementations via config |
| ~~SQLite for POC~~ Postgres for both (via Docker) | SQLite concurrent write contention (FM-3.1) and JSON query semantic differences (FM-9.1) cause real bugs. Postgres via Docker is one extra container but eliminates an entire class of local↔prod behavioral mismatches |
| Config-driven environment switching | `local.yaml` / `production.yaml` determines which adapter implementations are loaded |
| Queue message schemas as JSON | Language-agnostic, human-readable, easy to debug |

---

## Adapter Interfaces

### BlobStore
```
put(path: str, data: bytes) -> None
get(path: str) -> bytes
exists(path: str) -> bool
delete(path: str) -> None
```
Implementations: `LocalFsBlobStore`, `GcsBlobStore` / `S3BlobStore`

**Path safety (FM-9.2):** `LocalFsBlobStore` must resolve the full path and assert it starts with the configured `base_path`. Reject any path containing `..`. Validate that `tenant_id` and `job_id` are valid UUIDs before constructing paths. (S3 keys are flat strings so this is not a concern in cloud.)

### Queue
```
publish(topic: str, message: dict) -> None
subscribe(topic: str, handler: Callable) -> None
ack(message_id: str) -> None
extend_visibility(message_id: str, seconds: int) -> None  # heartbeat for SQS (FM-2.2)
```
Implementations: `RedisStreamQueue`, `SqsQueue` / `PubSubQueue`

**Important (FM-2.1):** Use Redis Streams with consumer groups, **not** Redis lists (LPUSH/BRPOP). With Streams, messages are only removed from the pending list after explicit ACK. If a consumer crashes, `XPENDING` reveals unacknowledged messages and `XCLAIM` reassigns them to another worker. Do not ACK until the entire processing pipeline completes and the next queue message is published.

**Reaper process:** A periodic job (cron or background task) queries the DB for jobs stuck in intermediate states (`ocr_processing`, `extracting`) for longer than a timeout (e.g., 10 minutes) and re-enqueues them.

### Database
- Rust services: use `sqlx` with compile-time checked queries
- Python services: use `sqlalchemy` or raw `sqlite3` / `asyncpg`
- Shared migration files for schema consistency

---

## DB Schema

### `jobs` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| tenant_id | UUID | FK to tenants |
| status | ENUM | See state machine (expanded with delivery + review states) |
| source_channel | TEXT | `telegram` or `email` |
| source_identifier | TEXT | chat_id or email address |
| source_file_unique_id | TEXT | Telegram `file_unique_id` for dedup (FM-11.2) |
| confidence_score | FLOAT | Nullable, set after extraction |
| input_blob_path | TEXT | Denormalized from blob_paths for simple queries (FM-9.1) |
| output_blob_path | TEXT | Denormalized from blob_paths for simple queries (FM-9.1) |
| blob_paths | JSON | `{input, ocr, extraction, output}` |
| extraction_data | JSON | Nullable, full extraction result |
| error_message | TEXT | Nullable, set on failure |
| retry_count | INTEGER | Default 0, incremented on retry (FM-10.1) |
| delivery_attempts | INTEGER | Default 0, incremented per delivery try (FM-2.3) |
| last_delivery_error | TEXT | Nullable, most recent delivery failure reason (FM-2.3) |
| processed_by | TEXT | Worker ID for concurrent processing detection (FM-2.2) |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### `tenants` table
| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| name | TEXT | |
| config | JSON | Tenant-specific settings |
| rate_limit | INTEGER | Requests per minute |

### Job Status State Machine
```
queued → ocr_processing → ocr_done → extracting → extracted → validating → done → output_generated → delivered
              │                          │               │                                │
              ▼                          ▼               ▼                                ▼
          ocr_failed            extraction_failed   needs_review                   delivery_failed
              │                          │               │
              ▼ (retry)                  ▼ (retry)       ▼ (manual)
           queued                     ocr_done      reviewed / accepted / corrected
```

**Additions from failure mode analysis:**
- `done → output_generated → delivered`: Split output states so undelivered jobs are visible (FM-2.3)
- `delivery_failed`: Telegram/email delivery failure distinct from extraction failure
- `ocr_failed → queued`, `extraction_failed → ocr_done`: Retry transitions (FM-10.1)
- `needs_review → reviewed / accepted / corrected`: Resolution workflow (FM-10.2)

---

## Blob Storage Path Conventions
```
/{tenant_id}/{job_id}/input.pdf
/{tenant_id}/{job_id}/ocr_output.json
/{tenant_id}/{job_id}/extraction.json
/{tenant_id}/{job_id}/output.xlsx
```

---

## Queue Contracts

### Queue A: Ingestion → Processing
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

### Queue B: Processing → Output
```json
{
  "job_id": "uuid",
  "tenant_id": "uuid",
  "extraction": { "...structured extraction result..." },
  "confidence_score": 0.92,
  "output_blob_path": "/{tenant_id}/{job_id}/output.xlsx",
  "source_channel": "telegram|email",
  "source_identifier": "chat_id|email_address"
}
```

---

## Config Structure
```
config/
├── local.yaml       # SQLite, local FS, Redis localhost
└── production.yaml  # RDS, S3, SQS endpoints + credentials ref
```

Example `local.yaml`:
```yaml
database:
  type: sqlite
  path: ./data/invoices.db

blob_storage:
  type: local_fs
  base_path: ./data/blobs

queue:
  type: redis
  url: redis://localhost:6379
```

---

## Local Dev Setup

### Docker Compose services
- Redis (queue)
- Optional: Postgres (if skipping SQLite)

### Migration tooling
- SQL migration files in `migrations/` directory
- Applied via sqlx-cli (Rust) or Alembic (Python)

### Seed data
- Default tenant for single-tenant POC
- Sample invoice files for testing

---

## POC Scope
- [x] SQLite database with jobs + tenants tables
- [x] Local filesystem blob storage
- [x] Redis queue with JSON messages
- [x] Docker Compose for Redis
- [x] Config loading from `local.yaml`
- [ ] Adapter interfaces defined and implemented for local

## Production Considerations
- Postgres used in both environments — no migration needed
- Blob path conventions already match S3/GCS key structure
- Queue contracts are the same; `extend_visibility` is a no-op for Redis Streams but critical for SQS
- Config switching is the only change needed per environment
- **JSON Schema** (FM-3.2): Formally define Queue B message schema. Integration test: round-trip extraction JSON from Python through Rust deserialization. Store `vat_rate` as percentage integer (20, not 0.20) to avoid float representation issues
