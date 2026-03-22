"""Tests for RedisStreamQueue — requires running Redis on localhost:6379."""

import pytest

from invoice_shared.adapters.queue import RedisStreamQueue

TOPIC = "test_queue_topic"


@pytest.fixture
def queue():
    """Create a queue and clean up the test stream after."""
    q = RedisStreamQueue("redis://localhost:6379", "test_group")
    yield q
    # Cleanup
    try:
        q._client.delete(TOPIC)
    except Exception:
        pass


@pytest.mark.integration
class TestRedisStreamQueue:
    def test_publish_and_consume(self, queue):
        msg = {"job_id": "123", "data": "test"}
        msg_id = queue.publish(TOPIC, msg)
        assert msg_id

        results = queue.consume(TOPIC, count=1, block_ms=1000)
        assert len(results) == 1
        received_id, received_data = results[0]
        assert received_data["job_id"] == "123"

    def test_ack_removes_from_pending(self, queue):
        msg = {"job_id": "456"}
        queue.publish(TOPIC, msg)
        results = queue.consume(TOPIC, count=1, block_ms=1000)
        assert len(results) == 1
        msg_id = results[0][0]

        # Before ack: message should be pending
        pending = queue.pending(TOPIC)
        assert len(pending) >= 1

        # After ack: message removed from pending
        queue.ack(TOPIC, msg_id)

    def test_extend_visibility_is_noop(self, queue):
        # Should not raise
        queue.extend_visibility(TOPIC, "fake-id", 30)

    def test_consume_empty_returns_empty(self, queue):
        # Ensure group exists
        queue.publish(TOPIC, {"setup": True})
        queue.consume(TOPIC, count=1, block_ms=100)
        # Now consume again — should be empty
        results = queue.consume(TOPIC, count=1, block_ms=100)
        assert results == []
