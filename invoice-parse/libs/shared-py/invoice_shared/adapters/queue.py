"""Queue adapter — abstract interface + Redis Streams implementation."""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

import redis


class MessageQueue(ABC):
    @abstractmethod
    def publish(self, topic: str, message: dict) -> str:
        """Publish a message. Returns message ID."""
        ...

    @abstractmethod
    def consume(self, topic: str, count: int = 1, block_ms: int = 5000) -> list[tuple[str, dict]]:
        """Consume messages. Returns list of (message_id, message_dict)."""
        ...

    @abstractmethod
    def ack(self, topic: str, message_id: str) -> None: ...

    @abstractmethod
    def extend_visibility(self, topic: str, message_id: str, seconds: int) -> None: ...


class RedisStreamQueue(MessageQueue):
    """Redis Streams with consumer groups.

    Uses XADD for publish, XREADGROUP for consume, XACK for acknowledge.
    Consumer groups ensure messages are only delivered to one consumer
    and can be reclaimed (XCLAIM) if a consumer crashes.
    """

    def __init__(self, url: str, consumer_group: str) -> None:
        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._consumer_group = consumer_group
        self._consumer_name = f"worker-{uuid.uuid4().hex[:8]}"
        self._initialized_groups: set[str] = set()

    def _ensure_group(self, topic: str) -> None:
        if topic in self._initialized_groups:
            return
        try:
            self._client.xgroup_create(topic, self._consumer_group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._initialized_groups.add(topic)

    def publish(self, topic: str, message: dict) -> str:
        self._ensure_group(topic)
        payload = {"data": json.dumps(message)}
        msg_id: str = self._client.xadd(topic, payload)
        return msg_id

    def consume(self, topic: str, count: int = 1, block_ms: int = 5000) -> list[tuple[str, dict]]:
        self._ensure_group(topic)
        results = self._client.xreadgroup(
            groupname=self._consumer_group,
            consumername=self._consumer_name,
            streams={topic: ">"},
            count=count,
            block=block_ms,
        )
        if not results:
            return []
        messages = []
        for _stream, entries in results:
            for msg_id, fields in entries:
                data = json.loads(fields["data"])
                messages.append((msg_id, data))
        return messages

    def ack(self, topic: str, message_id: str) -> None:
        self._client.xack(topic, self._consumer_group, message_id)

    def extend_visibility(self, topic: str, message_id: str, seconds: int) -> None:
        # Redis Streams don't have visibility timeout — no-op.
        # SQS implementation would call ChangeMessageVisibility here.
        pass

    def pending(self, topic: str) -> list[dict[str, Any]]:
        """Get pending (unacknowledged) messages — useful for monitoring and reaper."""
        return self._client.xpending_range(
            topic, self._consumer_group, min="-", max="+", count=100
        )
