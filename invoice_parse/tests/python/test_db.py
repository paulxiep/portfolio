"""Tests for DB layer — requires running Postgres on localhost:5432 with migration applied."""

import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from invoice_shared.db import (
    Base,
    InvalidTransitionError,
    JobModel,
    TenantModel,
    transition_job,
)
from invoice_shared.models import JobStatus

DB_URL = "postgresql://invoice:invoice@localhost:5432/invoice_parse"

TENANT_ID = uuid.UUID("a0000000-0000-0000-0000-000000000001")


@pytest.fixture
def session():
    """Create a transactional session that rolls back after each test."""
    engine = create_engine(DB_URL)
    conn = engine.connect()
    trans = conn.begin()
    factory = sessionmaker(bind=conn)
    sess = factory()
    yield sess
    sess.close()
    trans.rollback()
    conn.close()


@pytest.mark.integration
class TestJobTransitions:
    def _create_job(self, session, status=JobStatus.QUEUED):
        job = JobModel(
            tenant_id=TENANT_ID,
            status=status,
            source_channel="telegram",
            source_identifier="12345",
        )
        session.add(job)
        session.flush()
        return job

    def test_valid_transition_queued_to_ocr_processing(self, session):
        job = self._create_job(session)
        result = transition_job(session, str(job.id), JobStatus.OCR_PROCESSING)
        assert result.status == JobStatus.OCR_PROCESSING

    def test_invalid_transition_queued_to_done(self, session):
        job = self._create_job(session)
        with pytest.raises(InvalidTransitionError):
            transition_job(session, str(job.id), JobStatus.DONE)

    def test_retry_transition_ocr_failed_to_queued(self, session):
        job = self._create_job(session, status=JobStatus.OCR_FAILED)
        result = transition_job(session, str(job.id), JobStatus.QUEUED)
        assert result.status == JobStatus.QUEUED

    def test_review_resolution(self, session):
        job = self._create_job(session, status=JobStatus.NEEDS_REVIEW)
        result = transition_job(session, str(job.id), JobStatus.ACCEPTED)
        assert result.status == JobStatus.ACCEPTED

    def test_terminal_state_has_no_transitions(self, session):
        job = self._create_job(session, status=JobStatus.DELIVERED)
        with pytest.raises(InvalidTransitionError):
            transition_job(session, str(job.id), JobStatus.DONE)


@pytest.mark.integration
class TestTenantModel:
    def test_seed_tenants_exist(self, session):
        tenants = session.query(TenantModel).all()
        assert len(tenants) >= 2
        names = {t.name for t in tenants}
        assert "Test Tenant Alpha" in names
        assert "Test Tenant Beta" in names
