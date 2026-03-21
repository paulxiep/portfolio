# Ingestion Service

## Purpose
Receive invoice documents from external channels (Telegram, email), authenticate the source, store the input file, create a job record, and enqueue for processing.

**Rust (Axum)** — optimized for high-concurrency connection handling with minimal memory.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Rust |
| Web framework | Axum |
| Telegram client | teloxide or raw Bot API via reqwest |
| Database | sqlx (SQLite locally, Postgres in prod) |
| Queue | Redis via `redis-rs` |
| Blob storage | Local FS adapter (see infra contracts) |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Rust/Axum over Python/FastAPI | Handles burst connections with minimal memory; stateless I/O-bound work |
| Webhook mode over polling for Telegram | Lower latency, no polling overhead; ngrok for local dev |
| Tenant identification from source | Telegram: map chat_id → tenant; Email: map sender domain → tenant |
| Store raw file immediately | Decouple ingestion speed from processing speed |

---

## Interface Contracts

### Inputs
- **Telegram webhook**: POST from Telegram Bot API with document message
- **Email** (future): Inbound webhook from SendGrid/Mailgun, or IMAP polling

### Outputs
- **Queue A message**: JSON message to processing queue (see infra contracts)
- **Blob storage**: Raw input file at `/{tenant_id}/{job_id}/input.pdf`
- **Database**: New row in `jobs` table with status `queued`

---

## API / Endpoints

### `POST /webhook/telegram`
- Receives Telegram update
- Validates it contains a document
- Identifies tenant from chat_id
- Downloads file from Telegram servers
- Writes to blob storage
- Creates job record in DB
- Publishes to Queue A
- Responds 200 OK (async processing)

### `GET /health`
- Returns service health status

---

## Telegram Integration

### Bot setup
- BotFather: create bot, get token
- Set webhook URL: `https://<domain>/webhook/telegram`
- Local dev: ngrok tunnel → localhost

### File handling
- Telegram sends `file_id` in message
- Service calls `getFile` API → download URL
- Download file bytes, write to blob storage
- Support PDF, common image formats (jpg, png)

---

## Tenant Identification

### POC (single tenant)
- Hardcoded default tenant for all requests

### Production
- Lookup table: `source_identifier → tenant_id`
- Telegram: `chat_id` mapped to tenant
- Email: sender domain or specific address mapped to tenant
- Unknown source → reject with error message

---

## Rate Limiting
- Per-tenant rate limit from `tenants.rate_limit` column
- In-memory counter (POC) or Redis-based (production)
- Exceeded → respond with "rate limited" message, don't enqueue

---

## POC Scope
- [ ] Axum server with `/webhook/telegram` endpoint
- [ ] Telegram file download
- [ ] Local filesystem blob write
- [ ] SQLite job record creation
- [ ] Redis queue publish
- [ ] Hardcoded single tenant
- [ ] ngrok for local webhook testing

## Production Considerations
- Webhook signature validation for Telegram
- Email channel support (SendGrid inbound parse or IMAP)
- Multi-tenant lookup and rate limiting
- Request logging and tracing (correlation ID = job_id)
- Graceful shutdown (drain in-flight requests)
- Health check endpoint for ECS/ALB
