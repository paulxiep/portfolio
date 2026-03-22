"""Database layer — SQLAlchemy ORM models and job status transitions."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import AppConfig
from .models import JobStatus


class Base(DeclarativeBase):
    pass


class TenantModel(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(Text, nullable=False)
    config = Column(JSONB, nullable=False, server_default="{}")
    rate_limit = Column(Integer, nullable=False, default=60)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    status = Column(Enum(JobStatus, name="job_status", create_type=False, values_callable=lambda e: [x.value for x in e]), nullable=False, default=JobStatus.QUEUED)
    source_channel = Column(Text, nullable=False)
    source_identifier = Column(Text, nullable=False)
    source_file_unique_id = Column(Text)
    confidence_score = Column(Float)
    input_blob_path = Column(Text)
    output_blob_path = Column(Text)
    blob_paths = Column(JSONB, nullable=False, server_default="{}")
    extraction_data = Column(JSONB)
    error_message = Column(Text)
    retry_count = Column(Integer, nullable=False, default=0)
    delivery_attempts = Column(Integer, nullable=False, default=0)
    last_delivery_error = Column(Text)
    processed_by = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


# State machine: valid transitions
VALID_TRANSITIONS: dict[JobStatus, list[JobStatus]] = {
    JobStatus.QUEUED: [JobStatus.OCR_PROCESSING],
    JobStatus.OCR_PROCESSING: [JobStatus.OCR_DONE, JobStatus.OCR_FAILED],
    JobStatus.OCR_DONE: [JobStatus.EXTRACTING],
    JobStatus.EXTRACTING: [JobStatus.EXTRACTED, JobStatus.EXTRACTION_FAILED],
    JobStatus.EXTRACTED: [JobStatus.VALIDATING],
    JobStatus.VALIDATING: [JobStatus.DONE, JobStatus.NEEDS_REVIEW],
    JobStatus.DONE: [JobStatus.OUTPUT_GENERATED],
    JobStatus.OUTPUT_GENERATED: [JobStatus.DELIVERED, JobStatus.DELIVERY_FAILED],
    # Retry transitions
    JobStatus.OCR_FAILED: [JobStatus.QUEUED],
    JobStatus.EXTRACTION_FAILED: [JobStatus.OCR_DONE],
    # Review resolution
    JobStatus.NEEDS_REVIEW: [JobStatus.REVIEWED, JobStatus.ACCEPTED, JobStatus.CORRECTED],
    # Delivery retry
    JobStatus.DELIVERY_FAILED: [JobStatus.OUTPUT_GENERATED],
    # Terminal states
    JobStatus.DELIVERED: [],
    JobStatus.REVIEWED: [],
    JobStatus.ACCEPTED: [],
    JobStatus.CORRECTED: [],
}


class InvalidTransitionError(Exception):
    pass


def engine_from_config(config: AppConfig):
    return create_engine(config.database.url)


def session_factory(config: AppConfig) -> sessionmaker[Session]:
    engine = engine_from_config(config)
    return sessionmaker(bind=engine)


@contextmanager
def get_session(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def transition_job(session: Session, job_id: str, to_status: JobStatus) -> JobModel:
    """Transition a job to a new status, validating against the state machine."""
    job = session.query(JobModel).filter(JobModel.id == job_id).with_for_update().one()
    current = JobStatus(job.status)
    allowed = VALID_TRANSITIONS.get(current, [])
    if to_status not in allowed:
        raise InvalidTransitionError(
            f"Cannot transition job {job_id} from '{current}' to '{to_status}'. "
            f"Allowed: {allowed}"
        )
    job.status = to_status
    job.updated_at = datetime.now(timezone.utc)
    return job
