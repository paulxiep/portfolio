-- Invoice Parse: Initial Schema
-- Shared between Rust (sqlx) and Python (SQLAlchemy) services

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Job status enum: full state machine from architecture doc
CREATE TYPE job_status AS ENUM (
    'queued',
    'ocr_processing',
    'ocr_done',
    'extracting',
    'extracted',
    'validating',
    'done',
    'output_generated',
    'delivered',
    'ocr_failed',
    'extraction_failed',
    'needs_review',
    'delivery_failed',
    'reviewed',
    'accepted',
    'corrected'
);

CREATE TABLE tenants (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    config      JSONB NOT NULL DEFAULT '{}',
    rate_limit  INTEGER NOT NULL DEFAULT 60,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id               UUID NOT NULL REFERENCES tenants(id),
    status                  job_status NOT NULL DEFAULT 'queued',
    source_channel          TEXT NOT NULL,                          -- 'telegram' or 'email'
    source_identifier       TEXT NOT NULL,                          -- chat_id or email address
    source_file_unique_id   TEXT,                                   -- Telegram file_unique_id for dedup (FM-11.2)
    confidence_score        DOUBLE PRECISION,
    input_blob_path         TEXT,                                   -- denormalized for simple queries (FM-9.1)
    output_blob_path        TEXT,                                   -- denormalized for simple queries (FM-9.1)
    blob_paths              JSONB NOT NULL DEFAULT '{}',            -- {input, ocr, extraction, output}
    extraction_data         JSONB,                                  -- full extraction result
    error_message           TEXT,
    retry_count             INTEGER NOT NULL DEFAULT 0,             -- FM-10.1
    delivery_attempts       INTEGER NOT NULL DEFAULT 0,             -- FM-2.3
    last_delivery_error     TEXT,                                   -- FM-2.3
    processed_by            TEXT,                                   -- worker ID for concurrency detection (FM-2.2)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX idx_jobs_tenant_id ON jobs(tenant_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_source_file_unique_id ON jobs(source_file_unique_id) WHERE source_file_unique_id IS NOT NULL;
CREATE INDEX idx_jobs_created_at ON jobs(created_at);

-- Updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_tenants_updated_at
    BEFORE UPDATE ON tenants
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
