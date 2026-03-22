use sqlx::PgPool;
use uuid::Uuid;

use crate::config::AppConfig;
use crate::models::JobStatus;

/// Create a Postgres connection pool from config.
pub async fn create_pool(config: &AppConfig) -> Result<PgPool, sqlx::Error> {
    PgPool::connect(&config.database.url).await
}

/// Valid state machine transitions.
pub fn valid_transitions(from: JobStatus) -> &'static [JobStatus] {
    use JobStatus::*;
    match from {
        Queued => &[OcrProcessing],
        OcrProcessing => &[OcrDone, OcrFailed],
        OcrDone => &[Extracting],
        Extracting => &[Extracted, ExtractionFailed],
        Extracted => &[Validating],
        Validating => &[Done, NeedsReview],
        Done => &[OutputGenerated],
        OutputGenerated => &[Delivered, DeliveryFailed],
        // Retry transitions
        OcrFailed => &[Queued],
        ExtractionFailed => &[OcrDone],
        // Review resolution
        NeedsReview => &[Reviewed, Accepted, Corrected],
        // Delivery retry
        DeliveryFailed => &[OutputGenerated],
        // Terminal states
        Delivered | Reviewed | Accepted | Corrected => &[],
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransitionError {
    #[error("Invalid transition: {from:?} → {to:?}. Allowed: {allowed:?}")]
    InvalidTransition {
        from: JobStatus,
        to: JobStatus,
        allowed: Vec<JobStatus>,
    },
    #[error("Database error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

/// Transition a job to a new status, validating against the state machine.
pub async fn transition_job(
    pool: &PgPool,
    job_id: Uuid,
    to_status: JobStatus,
) -> Result<(), TransitionError> {
    let row: (JobStatus,) = sqlx::query_as("SELECT status FROM jobs WHERE id = $1 FOR UPDATE")
        .bind(job_id)
        .fetch_one(pool)
        .await?;

    let current = row.0;
    let allowed = valid_transitions(current);
    if !allowed.contains(&to_status) {
        return Err(TransitionError::InvalidTransition {
            from: current,
            to: to_status,
            allowed: allowed.to_vec(),
        });
    }

    sqlx::query("UPDATE jobs SET status = $1, updated_at = now() WHERE id = $2")
        .bind(to_status)
        .bind(job_id)
        .execute(pool)
        .await?;

    Ok(())
}
