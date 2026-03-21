# Dashboard

## Purpose
Read-only monitoring interface showing job status, metrics, and error rates. Quick to build, provides visibility into pipeline health.

**Streamlit** — minimal frontend code, Python-native, suitable for internal tools.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | Streamlit |
| Database | SQLite (POC) / Postgres (prod) via SQLAlchemy |
| Charts | Streamlit built-in charts + Plotly |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Streamlit over custom frontend | Fast to build, good enough for monitoring; not user-facing |
| Read-only DB access | No writes from dashboard; separate DB user with SELECT-only permissions in prod |
| Auto-refresh | Streamlit `st.rerun` on timer for near-real-time updates |

---

## Interface Contracts

### Input
- **Database**: Read from `jobs` and `tenants` tables
- **Blob storage**: Read OCR output and extraction JSON for detail views

### Output
- Web UI on `localhost:8501` (local) or Cloud Run / App Runner (prod)

---

## Views

### 1. Job Status Overview (main page)
- Status counts: cards showing # of jobs per status (queued, processing, done, output_generated, delivered, failed, needs_review, delivery_failed)
- Recent jobs table: last 50 jobs with status, tenant, created_at, confidence score, delivery_attempts
- Filterable by status, date range, tenant
- **Stuck job highlight** (FM-2.1): Jobs in `ocr_processing` or `extracting` for > 10 minutes highlighted in red

### 2. Per-Job Detail View
- Job metadata (ID, tenant, source, timestamps, `processed_by` worker ID)
- Input file preview (if image/PDF rendering available)
- OCR output text
- Extraction result (formatted JSON or table)
- Confidence score with per-check pass/fail/skip breakdown
- Validation warnings/errors
- Delivery status: attempts count, last error, delivery timestamps
- **Actions** (FM-10.1, FM-10.2):
  - Retry button: re-enqueue failed jobs (`ocr_failed → queued`, `extraction_failed → ocr_done`)
  - Review resolution: Accept / Correct / Reject for `needs_review` jobs
  - Re-deliver button: re-send Excel for `delivery_failed` jobs

### 3. Metrics
- Processing latency: histogram of `done - created_at` durations
- Throughput: jobs processed per hour/day
- Error rate: % of jobs in failed states over time
- Confidence distribution: histogram of confidence scores
- **Delivery success rate**: % of `output_generated` → `delivered` vs `delivery_failed` (FM-2.3)
- **OCR table detection rate**: % of jobs where OCR found table regions vs text-only fallback (FM-1.1)
- **LLM provider health**: Circuit breaker status per provider, success/failure rates (FM-7.2)

### 4. Per-Tenant View (future)
- Filter all above by tenant
- Per-tenant usage/cost tracking

---

## POC Scope
- [ ] Streamlit app with SQLite connection
- [ ] Job status overview with counts and recent jobs table
- [ ] Basic latency and error rate metrics
- [ ] Per-job detail view (extraction result display)

## Production Considerations
- Authentication (Streamlit has basic auth support; or put behind reverse proxy)
- Postgres connection with read-only credentials
- Auto-refresh interval tuning (avoid DB overload)
- Deploy as separate container on Cloud Run / App Runner
- Alerting integration (surface critical errors, not just display)
- **Write access for actions** (FM-10.1, FM-10.2): Dashboard needs a separate write-capable endpoint (API service or direct DB write with scoped credentials) for retry/review actions. Don't give the Streamlit app full write access to the jobs table — expose specific mutation endpoints only.
