# Ingestion Service

## Purpose
Receive invoice documents from external channels (email, Telegram), authenticate the source, store the input file, create a job record, and enqueue for processing.

**Rust (Axum + tokio)** — Axum serves a health endpoint; a background tokio task polls an IMAP mailbox for incoming invoices.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Rust |
| Web framework | Axum (health endpoint only) |
| Email client | `async-imap` 0.11 + `async-native-tls` |
| Email parser | `mail-parser` 0.9 (MIME parsing, attachment extraction) |
| Database | sqlx (Postgres) |
| Queue | Redis Streams via `redis-rs` |
| Blob storage | Local FS adapter (see infra contracts) |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Rust/Axum over Python/FastAPI | Handles burst connections with minimal memory; stateless I/O-bound work |
| IMAP polling over webhook for POC | No external service signup (SendGrid/Mailgun), no ngrok, no DNS changes; works behind NAT. Gmail App Password takes 2 minutes to set up |
| `async-imap` over sync `imap` | Fits tokio runtime; avoids blocking the async executor |
| `mail-parser` for MIME | Zero-copy, RFC 5322/8621 compliant, `.attachments()` API for direct extraction |
| Tenant identification from source | Email: map sender domain/address → tenant |
| Store raw file immediately | Decouple ingestion speed from processing speed |
| Mark email as Seen after success | Failed processing leaves email UNSEEN for retry on next poll cycle |

---

## Interface Contracts

### Inputs
- **Email (IMAP)**: Poll Gmail INBOX for UNSEEN messages with PDF/image attachments
- **Telegram** (future): POST webhook from Telegram Bot API with document message

### Outputs
- **Queue A message**: JSON message to processing queue (see infra contracts)
- **Blob storage**: Raw input file at `/{tenant_id}/{job_id}/input.pdf`
- **Database**: New row in `jobs` table with status `queued`

---

## IMAP Polling Flow

### Poll loop (background tokio task)
```
loop {
    1. Connect to IMAP server (imap.gmail.com:993) with TLS
    2. Login with credentials (from env vars)
    3. SELECT INBOX
    4. SEARCH for UNSEEN messages
    5. For each UNSEEN message:
       a. FETCH full message (RFC822)
       b. Parse with mail-parser → extract attachments
       c. For each PDF/image attachment:
          - Dedup check: Message-ID + filename as unique key
          - Tenant ID: hardcoded POC tenant
          - Generate job_id (UUID v4)
          - Write attachment to blob storage
          - INSERT job record (status: queued)
          - Publish QueueAMessage to Redis Stream
       d. Mark message as \Seen via STORE command
    6. LOGOUT
    7. Sleep for poll_interval_secs (default: 30s)
}
```

### `GET /health`
- Returns service health status (Axum, runs alongside poll loop)

---

## Email Integration

### Gmail setup (POC)
- Enable 2-Step Verification on Google account
- Create App Password at https://myaccount.google.com/apppasswords
- Enable IMAP in Gmail Settings > Forwarding and POP/IMAP
- Store credentials as `IMAP_USER` and `IMAP_PASSWORD` env vars

### File handling
- `mail-parser` extracts attachments with filename + content bytes
- Support PDF, common image formats (jpg, png)
- Dedup key: `{Message-ID}:{attachment_filename}` — prevents reprocessing on next poll

### Configuration (`config/local.yaml`)
```yaml
imap:
  server: imap.gmail.com
  port: 993
  poll_interval_secs: 30
  mailbox: INBOX
```

---

## Tenant Identification

### POC (single tenant)
- Hardcoded default tenant for all requests

### Production
- Lookup table: `source_identifier → tenant_id`
- Email: sender domain or specific address mapped to tenant
- Telegram: `chat_id` mapped to tenant
- Unknown source → log unregistered attempts with sender address for operator visibility

---

## Rate Limiting
- Per-tenant rate limit from `tenants.rate_limit` column
- In-memory counter (POC) or Redis-based (production)
- Exceeded → skip processing, log warning

---

## POC Scope
- [ ] IMAP poll loop connecting to Gmail
- [ ] Email parsing and attachment extraction
- [ ] Dedup on Message-ID + filename
- [ ] Local filesystem blob write
- [ ] Postgres job record creation
- [ ] Redis Stream queue publish
- [ ] Hardcoded single tenant
- [ ] Axum health endpoint

## Production Considerations
- Telegram channel support (webhook mode with teloxide or raw reqwest)
- Email via SendGrid/Mailgun inbound parse (webhook, lower latency than IMAP)
- Multi-tenant lookup from sender domain/address
- Per-tenant rate limiting
- Request logging and tracing (correlation ID = job_id) — **structured JSON logging with job_id in every log line** (FM-CC.1)
- Graceful shutdown (drain in-flight poll cycle)
- Health check endpoint for ECS/ALB
- IMAP IDLE for push-based notification instead of polling (reduces latency + Gmail API calls)
