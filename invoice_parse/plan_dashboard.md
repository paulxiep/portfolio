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
- Status counts: cards showing # of jobs per status (queued, processing, done, failed, needs_review)
- Recent jobs table: last 50 jobs with status, tenant, created_at, confidence score
- Filterable by status, date range

### 2. Per-Job Detail View
- Job metadata (ID, tenant, source, timestamps)
- Input file preview (if image/PDF rendering available)
- OCR output text
- Extraction result (formatted JSON or table)
- Confidence score with breakdown
- Validation warnings/errors

### 3. Metrics
- Processing latency: histogram of `done - created_at` durations
- Throughput: jobs processed per hour/day
- Error rate: % of jobs in failed states over time
- Confidence distribution: histogram of confidence scores

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
