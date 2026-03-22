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
5. **Structural completeness check** (FM-1.1): If zero table regions detected but monetary patterns (regex: `\d+[.,]\d{2}`) found in text regions, flag OCR as potentially degraded. Fallback: send original image to vision model (Gemini Flash with vision) for this specific job. Cost penalty is per-failure, not per-invoice.
6. Log OCR confidence scores and region-type metadata for miss rate monitoring
7. Write `ocr_output.json` to blob storage
8. Update job status: `queued → ocr_processing → ocr_done`

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
1. Build prompt from OCR output + extraction schema (see Prompt Engineering for critical instructions)
2. Call LLM via `LLMExtractor` interface (see LLM Abstraction below)
3. Parse response into Pydantic model (schema + field validators enforce format)
4. **Number format cross-check** (FM-1.2): Independently regex-parse OCR text for monetary amounts and compare against LLM-extracted values. Flag discrepancies for review even if VAT math passes.
5. **Subtotal detection** (FM-5.3): Flag any line item whose `total` equals the sum of other items in the same section. Flag suspiciously few items when OCR text contains many apparent line entries.
6. Write `extraction.json` to blob storage
7. Update job status: `ocr_done → extracting → extracted`

### Substep 3: Validation
1. **Schema validation** (Pydantic): required fields present, correct types, format validators pass
2. **Business logic validation**:
   - VAT math: `total_excl_vat × (1 + vat_rate/100) ≈ total_incl_vat` (skip if `vat_rate` is null; derive rate from `vat_amount / total_excl_vat` and include as computed value)
   - `total_excl_vat + vat_amount ≈ total_incl_vat`
   - Line items sum ≈ total_excl_vat — **adaptive tolerance** (FM-5.2): `max(0.01, 0.005 × len(line_items))` or relative: `abs(diff) / total < 0.005`
   - Date sanity (not in future, not ancient) — already enforced by `invoice_date` format validator
   - Currency is valid ISO 4217 — already enforced by Pydantic field validator
   - Section consistency: a "Misc" line item with `start_time` filled is suspicious (FM-1.3)
3. **Confidence scoring** (FM-5.1, FM-8.1):
   - Do **not** use LLM self-reported confidence (unreliable, poorly calibrated)
   - Build score from validation signals:
     - Checks passed / checks possible (coverage-aware; skipped checks reduce ceiling)
     - PaddleOCR confidence scores per region (actual statistical output)
     - Field completeness (how many non-null fields)
     - Format consistency (dates parseable, currency recognized, numbers cross-checked)
   - Use LLM output logprobs if available (Gemini/OpenAI support this) — actual model probabilities, not generated text
   - Below threshold → flag as `needs_review`
   - Report per-check pass/fail/skip breakdown in extraction metadata
4. Update job status: `extracted → validating → done` or `needs_review`

---

## Extraction Schema (Pydantic)

```python
class LineItem(BaseModel):
    section: str | None        # e.g. "Job", "Misc." — preserves invoice structure
    date: str | None
    item: str
    quantity: float | None
    unit: str | None           # e.g. "pc", "km", "hours"
    start_time: str | None
    finish_time: str | None
    hours: float | None
    total_hours: float | None  # distinct from per-shift hours (e.g. 3 workers × 8h = 24h)
    tariff: float | None
    tariff_unit: str | None    # e.g. "CZK/pc", "CZK/km", "CZK/h"
    total: float

class InvoiceExtraction(BaseModel):
    supplier_name: str
    supplier_address: str | None
    client_name: str
    client_address: str | None
    invoice_number: str
    invoice_date: str          # normalized to YYYY-MM-DD
    invoice_date_end: str | None  # if date range, second date (also YYYY-MM-DD)
    invoice_date_raw: str      # original text as it appears on invoice
    location: str | None
    total_excl_vat: float
    vat_amount: float
    vat_rate: float | None     # as percentage integer, e.g. 20 (not 0.20)
    total_incl_vat: float
    currency: str              # ISO 4217 3-letter code (CZK, EUR, USD)
    line_items: list[LineItem]

    @field_validator('currency')
    @classmethod
    def normalize_currency(cls, v):
        """Normalize to ISO 4217. Map common variants: 'Kc' → 'CZK', '€' → 'EUR', etc."""
        ...

    @field_validator('invoice_date')
    @classmethod
    def validate_date_format(cls, v):
        """Enforce YYYY-MM-DD format."""
        ...
```

---

## Prompt Engineering

- **System prompt**: "You are an invoice data extraction assistant..."
- **Schema**: Provide JSON schema derived from Pydantic model
- **Few-shot examples**: 2-3 diverse invoice formats (different languages, layouts)
- **Language handling**: No explicit language detection — LLM handles multilingual input natively
- **Critical instructions** (address failure modes FM-1.2, FM-1.3, FM-5.3, FM-6.1, FM-6.2):
  - Number formats: "European invoices use comma as decimal separator and space/period as thousands separator (e.g., 2 305,00 means two thousand three hundred five). Parse numbers according to the currency context."
  - Subtotals: "Do NOT include subtotal, summary, or 'in total' rows as line items. Only extract individual transaction/charge rows."
  - Sections: "Invoices may have multiple sections (e.g., Job/Labor, Miscellaneous, Materials). Preserve section names in the `section` field."
  - Dates: "Convert all dates to YYYY-MM-DD format. If the invoice shows a date range, use the first date as `invoice_date` and the second as `invoice_date_end`. Preserve original text in `invoice_date_raw`."
  - Currency: "Extract currency as a 3-letter ISO 4217 code (e.g., CZK, EUR, USD). Do not include units like /pc or /km."
  - VAT rate: "Express VAT rate as a percentage integer (e.g., 20 for 20%), not a decimal (not 0.20)."

---

## LLM Abstraction (FM-7.1)

Each LLM provider has different structured output APIs. Abstract behind a common interface:

```python
class LLMExtractor(ABC):
    @abstractmethod
    def extract(self, ocr_text: str, schema: dict) -> InvoiceExtraction: ...

class GeminiExtractor(LLMExtractor):
    """Uses response_schema parameter for structured JSON output."""

class ClaudeExtractor(LLMExtractor):
    """Uses tool_use / function calling for structured output."""

class OpenAIExtractor(LLMExtractor):
    """Uses response_format: json_schema for structured output."""
```

- All implementations must produce an identical `InvoiceExtraction` Pydantic model
- Integration tests run the same sample invoice through all providers and compare outputs
- Shadow mode: periodically call secondary model alongside primary and diff results
- **Circuit breaker** (FM-7.2): After N consecutive failures to a provider, stop trying that provider for M minutes. Prevents burning retry budget on known-down provider. On total provider exhaustion, leave job in `ocr_done` and re-enqueue with delay (not `extraction_failed`)

---

## Retry Logic

| Substep | Retry strategy |
|---------|---------------|
| OCR | Retry up to 2x on failure; mark `ocr_failed` if exhausted |
| LLM API call | Retry up to 3x with exponential backoff per provider; then fallback to next provider in chain; circuit breaker stops retrying known-down providers |
| Validation | No retry (deterministic); flag issues, don't block |

---

## POC Scope
- [x] Redis queue consumer (poll loop)
- [x] PaddleOCR integration with layout detection
- [x] Gemini Flash API call with structured output
- [x] Pydantic extraction model (in shared lib)
- [x] Basic validation (schema + VAT math + line items sum + dates)
- [x] Simple confidence scoring (check pass rate + OCR confidence + field completeness)
- [x] Write results to blob storage + DB (optional DB via DI)
- [x] Enqueue to Queue B
- [x] CLI entry point for standalone testing without infra

## Production Considerations
- Horizontal scaling: multiple worker instances consuming from same queue
- Batch API for Gemini to reduce per-request overhead at scale
- Model fallback chain: Gemini Flash → Claude Haiku → GPT-4o-mini (via `LLMExtractor` interface)
- Token usage logging per tenant for cost tracking
- Processing timeout per job (kill stuck OCR/LLM calls)
- Dead-letter queue for repeatedly failing jobs
- PaddleOCR container with GPU support for faster processing
- **Golden test set** (FM-CC.2): Maintain 20-30 invoices with known-correct extractions. Run extraction pipeline against this set weekly. Alert if per-field accuracy drops below threshold. Pin model versions where possible.
- **Startup health check** (FM-7.2): Validate all LLM API keys with a trivial extraction test before accepting queue messages
- **Full LLM response logging**: Log raw LLM responses (not just parsed extractions) for replay and degradation investigation

---

## Implementation Design Decisions

Decisions made during MVP implementation that refine the original plan:

| Decision | Rationale |
|----------|-----------|
| PPStructureV3 API (not legacy PPStructure) | PaddleOCR 3.x provides `layout_blocks` with typed regions and HTML tables via `PPStructureV3` class. Replaces older `PPStructure` API. |
| PyMuPDF for PDF→image conversion | Pure Python wheel — no system `poppler` dependency (pdf2image requires it). Works on Windows without setup. |
| `google-genai` SDK with `response_json_schema` | New unified Google GenAI SDK (GA). Structured output via `response_json_schema=Model.model_json_schema()` + `response_mime_type="application/json"`. Temperature 0 for deterministic extraction. |
| `gemini-2.5-flash` default (configurable via `GEMINI_MODEL` env var) | Plan said "Gemini Flash 3.0" — actual model ID is `gemini-2.5-flash`. Made configurable for easy model upgrades. |
| Validation decoupled from OcrOutput | `validate_extraction()` takes `ocr_avg_confidence: float` instead of full `OcrOutput`. Keeps validation pure with no dependency on OCR data structures. |
| `run_pipeline()` with optional DB via DI | Single pipeline function serves both worker (with `db_session_factory`) and CLI (without). `_transition()` helper no-ops when factory is None. |
| CLI bypasses BlobStore for output | BlobStore enforces UUID path segments for multi-tenant safety. CLI writes to human-readable `--output-dir` for inspection. |
| HTML table parsing via stdlib `html.parser` | PPStructure returns tables as HTML. Used stdlib `HTMLParser` subclass — no BeautifulSoup dependency needed. |
| Adaptive line items tolerance | `max(0.01, 0.005 × n_items)` absolute OR `< 0.5%` relative. Handles both few-item and many-item invoices. |

### CLI Usage

```bash
# OCR only (no API key needed):
python -m invoice_processing.cli path/to/invoice.pdf --ocr-only -v

# Full pipeline (needs GEMINI_API_KEY):
GEMINI_API_KEY=xxx python -m invoice_processing.cli path/to/invoice.pdf -v

# Custom output directory and provider:
python -m invoice_processing.cli invoice.pdf --output-dir ./results --provider gemini
```

Outputs: `ocr_output.json`, `extraction.json`, `validation.json` + summary to stdout.
