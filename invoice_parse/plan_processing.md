# Processing Service

## Purpose
Consume jobs from Queue A, run the OCR → LLM extraction → validation pipeline, write results, and enqueue to Queue B for output.

**Python** — required for OCR libraries (PaddleOCR) and convenient for LLM API integration.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.12+ |
| OCR | PaddleOCR PP-Structure |
| LLM | Gemini Flash 3.0 (primary), Claude Haiku 4.5 / GPT-4o-mini (fallback) |
| Validation | Pydantic v2 |
| Queue | Redis via `redis-py` |
| Database | SQLAlchemy or raw sqlite3/asyncpg |
| PDF handling | pdf2image (poppler) or PyMuPDF |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| PaddleOCR PP-Structure | Layout-aware: detects tables, preserves row/column structure critical for invoices |
| Text-only LLM (no vision) | Image tokens 10-50x more expensive; OCR text is sufficient |
| Gemini Flash 3.0 as primary | Cost-efficient, structured output via `response_schema`, batch API available |
| Pydantic for validation | Schema validation + business rules in one framework |
| Sequential substeps with retry | OCR → LLM → Validation are tightly coupled; retry at each substep level |

---

## Interface Contracts

### Input
- **Queue A message**: JSON with job_id, tenant_id, blob_path (see infra contracts)
- **Blob storage**: Read input file from `/{tenant_id}/{job_id}/input.pdf`

### Output
- **Queue B message**: JSON with extraction result, confidence score (see infra contracts)
- **Blob storage**: Write `ocr_output.json` and `extraction.json`
- **Database**: Update job status through state machine, write extraction_data

---

## Pipeline: 3 Substeps

### Substep 1: OCR
1. Read PDF/image from blob storage
2. Convert PDF pages to images (pdf2image)
3. Run PaddleOCR PP-Structure
4. Output: structured text with layout regions (headers, tables, paragraphs)
5. Write `ocr_output.json` to blob storage
6. Update job status: `queued → ocr_processing → ocr_done`

**OCR output format:**
```json
{
  "pages": [
    {
      "regions": [
        {"type": "header", "text": "INVOICE"},
        {"type": "table", "rows": [["Item", "Qty", "Total"], ["Widget A", "5", "500"]]},
        {"type": "text", "text": "Thank you for your business"}
      ]
    }
  ]
}
```

### Substep 2: LLM Extraction
1. Build prompt from OCR output + extraction schema
2. Call Gemini Flash 3.0 with `response_schema` for structured JSON output
3. Parse response into Pydantic model
4. Write `extraction.json` to blob storage
5. Update job status: `ocr_done → extracting → extracted`

### Substep 3: Validation
1. **Schema validation** (Pydantic): required fields present, correct types
2. **Business logic validation**:
   - VAT math: `total_excl_vat × (1 + vat_rate) ≈ total_incl_vat` (tolerance ±0.01)
   - `total_excl_vat + vat_amount ≈ total_incl_vat`
   - Line items sum ≈ total_excl_vat
   - Date format parseable and sane (not in future, not ancient)
   - Currency consistency across all monetary fields
3. **Confidence scoring**:
   - Start with LLM self-reported confidence (if available)
   - Deduct for each validation failure
   - Below threshold → flag as `needs_review`
4. Update job status: `extracted → validating → done` or `needs_review`

---

## Extraction Schema (Pydantic)

```python
class LineItem(BaseModel):
    date: str | None
    item: str
    quantity: float | None
    start_time: str | None
    finish_time: str | None
    hours: float | None
    tariff: float | None
    total: float

class InvoiceExtraction(BaseModel):
    supplier_name: str
    supplier_address: str | None
    client_name: str
    client_address: str | None
    invoice_number: str
    invoice_date: str
    location: str | None
    total_excl_vat: float
    vat_amount: float
    vat_rate: float | None
    total_incl_vat: float
    currency: str
    line_items: list[LineItem]
```

---

## Prompt Engineering

- **System prompt**: "You are an invoice data extraction assistant..."
- **Schema**: Provide JSON schema derived from Pydantic model
- **Few-shot examples**: 2-3 diverse invoice formats (different languages, layouts)
- **Instructions**: Extract all fields, use null for missing, preserve original currency
- **Language handling**: No explicit language detection — LLM handles multilingual input natively

---

## Retry Logic

| Substep | Retry strategy |
|---------|---------------|
| OCR | Retry up to 2x on failure; mark `ocr_failed` if exhausted |
| LLM API call | Retry up to 3x with exponential backoff; fallback to alternate model |
| Validation | No retry (deterministic); flag issues, don't block |

---

## POC Scope
- [ ] Redis queue consumer (poll loop)
- [ ] PaddleOCR integration with layout detection
- [ ] Gemini Flash 3.0 API call with structured output
- [ ] Pydantic extraction model
- [ ] Basic validation (schema + VAT math)
- [ ] Simple confidence scoring
- [ ] Write results to local filesystem + SQLite
- [ ] Enqueue to Queue B

## Production Considerations
- Horizontal scaling: multiple worker instances consuming from same queue
- Batch API for Gemini to reduce per-request overhead at scale
- Model fallback chain: Gemini Flash → Claude Haiku → GPT-4o-mini
- Token usage logging per tenant for cost tracking
- Processing timeout per job (kill stuck OCR/LLM calls)
- Dead-letter queue for repeatedly failing jobs
- PaddleOCR container with GPU support for faster processing
