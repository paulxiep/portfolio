//! Core ingestion logic — receives a file, creates a job, stores the blob, enqueues for processing.
//!
//! Used by both the CLI (bulk enqueue) and the future IMAP/Telegram handlers.

use chrono::Utc;
use log::info;
use shared_rs::adapters::blob_store::BlobStore;
use shared_rs::adapters::queue::MessageQueue;
use shared_rs::models::{JobStatus, QueueAMessage};
use sqlx::PgPool;
use uuid::Uuid;

const QUEUE_A_TOPIC: &str = "queue:a";
const DEFAULT_TENANT: &str = "a0000000-0000-0000-0000-000000000001";

/// Ingest a single file: create job in DB, store blob, publish to Queue A.
pub async fn ingest_file(
    filename: &str,
    file_bytes: &[u8],
    pool: &PgPool,
    blob_store: &dyn BlobStore,
    queue: &dyn MessageQueue,
    source_channel: &str,
    source_identifier: &str,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let tenant_id: Uuid = DEFAULT_TENANT.parse()?;
    let job_id = Uuid::new_v4();
    let blob_path = format!("{}/{}/input.{}", tenant_id, job_id, file_extension(filename));

    // 1. Store file to blob storage
    blob_store.put(&blob_path, file_bytes)?;

    // 2. Create job record in DB
    sqlx::query(
        r#"INSERT INTO jobs (id, tenant_id, status, source_channel, source_identifier, input_blob_path, blob_paths)
           VALUES ($1, $2, $3, $4, $5, $6, $7)"#,
    )
    .bind(job_id)
    .bind(tenant_id)
    .bind(JobStatus::Queued)
    .bind(source_channel)
    .bind(source_identifier)
    .bind(&blob_path)
    .bind(serde_json::json!({"input": blob_path}))
    .execute(pool)
    .await?;

    // 3. Publish to Queue A
    let msg = QueueAMessage {
        job_id,
        tenant_id,
        blob_path: blob_path.clone(),
        source_channel: source_channel.to_string(),
        source_identifier: source_identifier.to_string(),
        created_at: Utc::now(),
    };
    queue.publish(QUEUE_A_TOPIC, &serde_json::to_value(&msg)?)?;

    info!("[{}] Ingested {} → {}", job_id, filename, blob_path);
    Ok(job_id)
}

fn file_extension(filename: &str) -> &str {
    filename
        .rsplit('.')
        .next()
        .unwrap_or("pdf")
}
