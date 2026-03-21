# Output Service

## Purpose
Consume validated extraction results from Queue B, generate Excel output, and deliver the result back to the user via the original channel (Telegram or email).

**Rust** — decoupled from processing so slow delivery doesn't block workers. Handles backpressure gracefully.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Rust |
| Excel generation | rust_xlsxwriter |
| Telegram client | teloxide or reqwest (Bot API) |
| Database | sqlx |
| Queue | redis-rs |
| Blob storage | Local FS adapter (see infra contracts) |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Rust over Python | Consistent with Ingestion; low memory, handles backpressure well |
| rust_xlsxwriter | Native Rust, no external deps, produces standard .xlsx |
| Separate from Processing | Slow delivery (network) shouldn't block OCR/LLM workers |
| Store output before sending | Artifact preserved even if delivery fails; retry delivery only |

---

## Interface Contracts

### Input
- **Queue B message**: JSON with extraction result, confidence, source info (see infra contracts)

### Output
- **Blob storage**: Write `output.xlsx` at `/{tenant_id}/{job_id}/output.xlsx`
- **Database**: Update job status to `done` or `needs_review`
- **Telegram/Email**: Send .xlsx file back to user

---

## Excel Generation

### Sheet structure
**Header section** (rows 1-12):
| Field | Value |
|-------|-------|
| Supplier Name | {extracted value} |
| Supplier Address | {extracted value} |
| Client Name | {extracted value} |
| ... | ... |
| Total incl. VAT | {extracted value} |

**Line items table** (starting after header):
| Date | Item | Qty | Start | Finish | Hours | Tariff | Total |
|------|------|-----|-------|--------|-------|--------|-------|
| ... | ... | ... | ... | ... | ... | ... | ... |

### Formatting
- Bold headers
- Currency formatting for monetary fields
- Date formatting
- Auto-width columns

---

## Response Routing

### Telegram
- Use `sendDocument` Bot API method
- Send .xlsx as document attachment to original `chat_id`
- Include summary text: "Invoice processed: {supplier} → {client}, Total: {total}"
- If `needs_review`: append warning "⚠ Low confidence — please verify"

### Email (future)
- Reply to original email with .xlsx attachment
- Include same summary in email body

---

## Processing Flow
1. Consume message from Queue B
2. Generate Excel from extraction data
3. Write .xlsx to blob storage
4. Send response to original channel
5. Update job status in DB (`done` or delivery failure status)

---

## POC Scope
- [ ] Redis queue consumer
- [ ] Excel generation with rust_xlsxwriter
- [ ] Telegram bot `sendDocument` reply
- [ ] Local filesystem blob write
- [ ] SQLite job status update

## Production Considerations
- Delivery retry with exponential backoff (Telegram rate limits, email bounces)
- Delivery failure tracking (separate from extraction failure)
- Template customization per tenant (future)
- Multiple output formats (CSV, PDF) per tenant config (future)
- Batch delivery (daily digest of all processed invoices)
