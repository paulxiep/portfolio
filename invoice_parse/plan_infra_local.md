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
| SQLite for POC, Postgres for production | Simpler local setup; same SQL semantics for basic operations |
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

### Queue
```
publish(topic: str, message: dict) -> None
subscribe(topic: str, handler: Callable) -> None
ack(message_id: str) -> None
```
Implementations: `RedisQueue`, `SqsQueue` / `PubSubQueue`

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
| status | ENUM | See state machine below |
| source_channel | TEXT | `telegram` or `email` |
| source_identifier | TEXT | chat_id or email address |
| confidence_score | FLOAT | Nullable, set after extraction |
| blob_paths | JSON | `{input, ocr, extraction, output}` |
| extraction_data | JSON | Nullable, full extraction result |
| error_message | TEXT | Nullable, set on failure |
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
queued → ocr_processing → ocr_done → extracting → extracted → validating → done
              │                          │               │
              ▼                          ▼               ▼
          ocr_failed            extraction_failed   needs_review
```

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
- Migration from SQLite → Postgres (schema designed to be compatible)
- Blob path conventions already match S3/GCS key structure
- Queue message schemas are the contract — implementation swaps transparently
- Config switching is the only change needed per environment
