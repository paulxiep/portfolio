"""Invoice pipeline monitoring dashboard.

Run with: streamlit run dashboard/app.py
Requires: Postgres running (docker compose), shared-py installed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st
from sqlalchemy import text

from invoice_shared.config import load_config
from invoice_shared.db import engine_from_config

st.set_page_config(page_title="Invoice Pipeline", layout="wide")

REFRESH_INTERVAL = 10  # seconds


@st.cache_resource
def get_engine():
    config = load_config()
    return engine_from_config(config)


def query(sql: str, params: dict | None = None) -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return [dict(row._mapping) for row in result]


# --- Header ---
st.title("Invoice Pipeline Monitor")

# --- Status overview ---
st.subheader("Job Status")

status_counts = query(
    "SELECT status, COUNT(*) as count FROM jobs GROUP BY status ORDER BY count DESC"
)

if status_counts:
    cols = st.columns(min(len(status_counts), 6))
    for i, row in enumerate(status_counts):
        with cols[i % len(cols)]:
            st.metric(row["status"], row["count"])
else:
    st.info("No jobs yet.")

# --- Stuck jobs ---
stuck = query("""
    SELECT id, status, tenant_id, created_at, updated_at
    FROM jobs
    WHERE status IN ('ocr_processing', 'extracting')
      AND updated_at < NOW() - INTERVAL '10 minutes'
    ORDER BY updated_at
    LIMIT 10
""")

if stuck:
    st.warning(f"{len(stuck)} stuck job(s) detected (>10 min in processing state)")
    st.dataframe(stuck, use_container_width=True)

# --- Recent jobs ---
st.subheader("Recent Jobs")

status_filter = st.selectbox(
    "Filter by status",
    ["All"] + [r["status"] for r in status_counts],
    index=0,
)

if status_filter == "All":
    jobs = query("""
        SELECT id, tenant_id, status, source_channel, source_identifier,
               confidence_score, delivery_attempts, created_at, updated_at
        FROM jobs
        ORDER BY created_at DESC
        LIMIT 50
    """)
else:
    jobs = query(
        """
        SELECT id, tenant_id, status, source_channel, source_identifier,
               confidence_score, delivery_attempts, created_at, updated_at
        FROM jobs
        WHERE status = :status
        ORDER BY created_at DESC
        LIMIT 50
        """,
        {"status": status_filter},
    )

if jobs:
    st.dataframe(jobs, use_container_width=True)

    # --- Per-job detail ---
    job_ids = [str(j["id"]) for j in jobs]
    selected_id = st.selectbox("Inspect job", job_ids, index=0)

    if selected_id:
        detail = query(
            "SELECT * FROM jobs WHERE id = :id",
            {"id": selected_id},
        )
        if detail:
            job = detail[0]
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Job Details")
                st.json({
                    "id": str(job["id"]),
                    "tenant_id": str(job["tenant_id"]),
                    "status": job["status"],
                    "source_channel": job["source_channel"],
                    "source_identifier": job["source_identifier"],
                    "confidence_score": job.get("confidence_score"),
                    "retry_count": job.get("retry_count"),
                    "delivery_attempts": job.get("delivery_attempts"),
                    "error_message": job.get("error_message"),
                    "created_at": str(job["created_at"]),
                    "updated_at": str(job["updated_at"]),
                })

            with col2:
                st.subheader("Extraction Data")
                if job.get("extraction_data"):
                    st.json(job["extraction_data"])
                else:
                    st.info("No extraction data yet.")

                if job.get("blob_paths"):
                    st.subheader("Blob Paths")
                    st.json(job["blob_paths"])
else:
    st.info("No jobs match the filter.")

# --- Metrics ---
st.subheader("Pipeline Metrics")

metrics_cols = st.columns(4)

total = query("SELECT COUNT(*) as n FROM jobs")
with metrics_cols[0]:
    st.metric("Total Jobs", total[0]["n"] if total else 0)

delivered = query("SELECT COUNT(*) as n FROM jobs WHERE status = 'delivered'")
with metrics_cols[1]:
    st.metric("Delivered", delivered[0]["n"] if delivered else 0)

failed = query(
    "SELECT COUNT(*) as n FROM jobs WHERE status IN ('ocr_failed', 'extraction_failed', 'delivery_failed')"
)
with metrics_cols[2]:
    st.metric("Failed", failed[0]["n"] if failed else 0)

review = query("SELECT COUNT(*) as n FROM jobs WHERE status = 'needs_review'")
with metrics_cols[3]:
    st.metric("Needs Review", review[0]["n"] if review else 0)

# Confidence distribution
confidences = query(
    "SELECT confidence_score FROM jobs WHERE confidence_score IS NOT NULL"
)
if confidences:
    scores = [c["confidence_score"] for c in confidences]
    st.subheader("Confidence Distribution")
    st.bar_chart({"confidence": scores})

# --- Auto-refresh ---
st.divider()
if st.checkbox("Auto-refresh", value=False):
    import time
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
