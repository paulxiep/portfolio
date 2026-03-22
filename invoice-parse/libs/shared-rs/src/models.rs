use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Job status enum — mirrors the Postgres `job_status` enum and Python `JobStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "job_status", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    OcrProcessing,
    OcrDone,
    Extracting,
    Extracted,
    Validating,
    Done,
    OutputGenerated,
    Delivered,
    OcrFailed,
    ExtractionFailed,
    NeedsReview,
    DeliveryFailed,
    Reviewed,
    Accepted,
    Corrected,
}

/// Database row for `jobs` table.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Job {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub status: JobStatus,
    pub source_channel: String,
    pub source_identifier: String,
    pub source_file_unique_id: Option<String>,
    pub confidence_score: Option<f64>,
    pub input_blob_path: Option<String>,
    pub output_blob_path: Option<String>,
    pub blob_paths: serde_json::Value,
    pub extraction_data: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub retry_count: i32,
    pub delivery_attempts: i32,
    pub last_delivery_error: Option<String>,
    pub processed_by: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Database row for `tenants` table.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct Tenant {
    pub id: Uuid,
    pub name: String,
    pub config: serde_json::Value,
    pub rate_limit: i32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// --- Extraction Schema (for deserializing Python-produced JSON) ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineItem {
    pub section: Option<String>,
    pub date: Option<String>,
    pub item: String,
    pub quantity: Option<f64>,
    pub unit: Option<String>,
    pub start_time: Option<String>,
    pub finish_time: Option<String>,
    pub hours: Option<f64>,
    pub total_hours: Option<f64>,
    pub tariff: Option<f64>,
    pub tariff_unit: Option<String>,
    pub total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvoiceExtraction {
    pub supplier_name: String,
    pub supplier_address: Option<String>,
    pub client_name: String,
    pub client_address: Option<String>,
    pub invoice_number: String,
    pub invoice_date: String,
    pub invoice_date_end: Option<String>,
    pub invoice_date_raw: String,
    pub location: Option<String>,
    pub total_excl_vat: f64,
    pub vat_amount: f64,
    pub vat_rate: Option<f64>,
    pub total_incl_vat: f64,
    pub currency: String,
    pub line_items: Vec<LineItem>,
}

// --- Queue Message Contracts ---

/// Queue A: Ingestion → Processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueAMessage {
    pub job_id: Uuid,
    pub tenant_id: Uuid,
    pub blob_path: String,
    pub source_channel: String,
    pub source_identifier: String,
    pub created_at: DateTime<Utc>,
}

/// Queue B: Processing → Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueBMessage {
    pub job_id: Uuid,
    pub tenant_id: Uuid,
    pub extraction: InvoiceExtraction,
    pub confidence_score: f64,
    pub output_blob_path: String,
    pub source_channel: String,
    pub source_identifier: String,
}
