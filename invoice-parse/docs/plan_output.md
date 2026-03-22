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
- Currency formatting: apply `#,##0.00` number format to all monetary cells (FM-12.1)
- Date formatting
- Auto-width columns

### Mixed section handling (FM-12.2)
When line items have a `section` field, generate **separate tables per section** (e.g., "Job" and "Misc.") mirroring the original invoice structure. Each section table includes only the columns relevant to that section type:
- Job sections: Date, Item, Qty, Start, Finish, Hours, Total Hours, Tariff, Total
- Misc sections: Item, Qty, Unit, Tariff (with unit), Total

This avoids sparse tables with many empty cells and preserves fidelity to the original invoice.

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
2. **Idempotency check** (FM-2.2): Before processing, verify job isn't already in a terminal state (`delivered`, `delivery_failed`). If so, ACK and skip.
3. **Round all monetary values to 2 decimal places** before writing to Excel (FM-12.1)
4. Generate Excel from extraction data (see Excel Generation below)
5. Write .xlsx to blob storage
6. Update job status: `done → output_generated`
7. Send response to original channel
8. Update job status: `output_generated → delivered` (or `delivery_failed` with error details)
9. On delivery failure: increment `delivery_attempts`, store `last_delivery_error`, re-enqueue for retry with exponential backoff

---

## POC Scope
- [ ] Redis queue consumer
- [ ] Excel generation with rust_xlsxwriter
- [ ] Telegram bot `sendDocument` reply
- [ ] Local filesystem blob write
- [ ] SQLite job status update

## Production Considerations
- **Delivery retry is a core feature, not a stretch goal** (FM-2.3): Telegram rate-limits at 30 msg/sec per bot and caps files at 50MB via `sendDocument`. These will fail regularly.
- Delivery failure tracking: `delivery_attempts` counter + `last_delivery_error` in DB (separate from extraction failure)
- **Include `job_id` in Telegram reply** for user↔operator correlation (FM-CC.1)
- Template customization per tenant (future)
- Multiple output formats (CSV, PDF) per tenant config (future)
- Batch delivery (daily digest of all processed invoices)
- **JSON deserialization safety** (FM-3.2): Define a Rust struct matching the Queue B schema exactly. Handle `i64`-to-`f64` coercion explicitly in `serde`. Integration test: deserialize a known Python-serialized extraction JSON.
