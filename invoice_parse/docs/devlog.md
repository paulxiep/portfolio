# Development Log

## Session 1: Design + Failure Mode Analysis + Infra Scaffolding

### Design phase
- Wrote initial architecture doc with service separation rationale, component designs, data models, scalability plan
- Wrote detailed plans for each service (ingestion, processing, output, dashboard) and infrastructure (local, cloud)

### Failure mode analysis
- Stress-tested every major design decision for failure modes
- Identified 20 concrete failure modes across OCR fidelity (FM-1.x), queue semantics (FM-2.x), service separation (FM-3.x), multi-tenancy (FM-4.x), validation (FM-5.x), schema design (FM-6.x), LLM fallback (FM-7.x), confidence scoring (FM-8.x), local-cloud equivalence (FM-9.x), state machine (FM-10.x), Telegram integration (FM-11.x), Excel output (FM-12.x), and cross-cutting concerns (FM-CC.x)
- Updated all 6 plan documents with mitigations — schema changes, prompt engineering, state machine expansions, queue semantics, idempotency, delivery tracking, golden test set

### Structural decisions for production-readiness

| Decision | Rationale |
|----------|-----------|
| Independent crates, no Cargo workspace | Each service is a standalone repo in production. Path deps mimic registry deps. No root Cargo.toml. |
| Per-service .venv (conda) | Mimics production where each service has its own dependency environment |
| `libs/` for shared packages | Publishable to private crate registry / PyPI in production |
| `services/` with own Dockerfile each | Each service is independently deployable |
| Shared `migrations/`, `config/`, `seed.sql` | Single source of truth for DB schema and config structure |
| psycopg (v3) over psycopg2 | psycopg2-binary doesn't build on MSYS2/MinGW; psycopg3 works with `postgresql+psycopg` dialect |

### Infra local implementation
- Docker Compose: Postgres 18, Redis 8
- SQL migration with full job status enum (16 states), jobs + tenants tables, indexes, updated_at triggers
- Seed data: two test tenants for isolation testing
- Config: local.yaml (Postgres + local FS + Redis Streams) and production.yaml placeholder
- Python shared library (`libs/shared-py/`): config loader, Pydantic models (InvoiceExtraction, LineItem with field validators for ISO 4217 currency + YYYY-MM-DD dates), SQLAlchemy ORM with state machine transition validation, BlobStore ABC + LocalFsBlobStore (path traversal protection), MessageQueue ABC + RedisStreamQueue (Streams with consumer groups)
- Rust shared crate (`libs/shared-rs/`): mirror of Python types with serde/sqlx derives, config loader, PgPool factory, state machine transitions, BlobStore trait + LocalFsBlobStore, MessageQueue trait + RedisStreamQueue
- Service placeholders: ingestion (Rust), processing (Python), output (Rust), dashboard (Python/Streamlit) — each with Cargo.toml/pyproject.toml + Dockerfile
- Tests: 7 Python blob store tests pass, 3 Rust blob store tests pass, all Rust crates compile

### Version choices (adoption-driven, not latest-chasing)

| Component | Version | Signal |
|-----------|---------|--------|
| Postgres | 18 | Latest stable |
| Redis | 8 | Latest stable |
| pydantic | 2.12 | 710M downloads/mo |
| SQLAlchemy | 2.0 | 331M downloads/mo |
| redis-py | 7.3 | 176M downloads/mo |
| psycopg | 3.3 | Newer but works; psycopg2-binary incompatible with MSYS2 |
| sqlx (Rust) | 0.8 | 20M recent downloads, 0.9 is alpha |
| redis (Rust) | 1.0 | Stable, adopted |
| serde_yaml | 0.9 | 48M downloads; replacement serde_yml at 0.0.12 not adopted |
