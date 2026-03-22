use redis::streams::{StreamReadOptions, StreamReadReply};
use redis::{Commands, RedisError};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QueueError {
    #[error("Redis error: {0}")]
    Redis(#[from] RedisError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Abstract message queue interface.
pub trait MessageQueue: Send + Sync {
    fn publish(&self, topic: &str, message: &Value) -> Result<String, QueueError>;
    fn consume(&self, topic: &str, count: usize, block_ms: usize) -> Result<Vec<(String, Value)>, QueueError>;
    fn ack(&self, topic: &str, message_id: &str) -> Result<(), QueueError>;
    fn extend_visibility(&self, topic: &str, message_id: &str, seconds: u64) -> Result<(), QueueError>;
}

/// Redis Streams implementation with consumer groups.
pub struct RedisStreamQueue {
    client: redis::Client,
    consumer_group: String,
    consumer_name: String,
}

impl RedisStreamQueue {
    pub fn new(url: &str, consumer_group: &str) -> Result<Self, QueueError> {
        let client = redis::Client::open(url)?;
        let consumer_name = format!("worker-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        Ok(Self {
            client,
            consumer_group: consumer_group.to_string(),
            consumer_name,
        })
    }

    fn ensure_group(&self, conn: &mut redis::Connection, topic: &str) -> Result<(), QueueError> {
        let result: Result<(), RedisError> =
            redis::cmd("XGROUP")
                .arg("CREATE")
                .arg(topic)
                .arg(&self.consumer_group)
                .arg("0")
                .arg("MKSTREAM")
                .query(conn);
        match result {
            Ok(()) => Ok(()),
            Err(e) if e.to_string().contains("BUSYGROUP") => Ok(()),
            Err(e) => Err(QueueError::Redis(e)),
        }
    }
}

impl MessageQueue for RedisStreamQueue {
    fn publish(&self, topic: &str, message: &Value) -> Result<String, QueueError> {
        let mut conn = self.client.get_connection()?;
        self.ensure_group(&mut conn, topic)?;
        let payload = serde_json::to_string(message)?;
        let id: String = conn.xadd(topic, "*", &[("data", &payload)])?;
        Ok(id)
    }

    fn consume(&self, topic: &str, count: usize, block_ms: usize) -> Result<Vec<(String, Value)>, QueueError> {
        let mut conn = self.client.get_connection()?;
        self.ensure_group(&mut conn, topic)?;

        let opts = StreamReadOptions::default()
            .group(&self.consumer_group, &self.consumer_name)
            .count(count)
            .block(block_ms);

        let reply: StreamReadReply = conn.xread_options(&[topic], &[">"], &opts)?;
        let mut messages = Vec::new();
        for key in reply.keys {
            for entry in key.ids {
                if let Some(redis::Value::BulkString(bytes)) = entry.map.get("data") {
                    let data_str = String::from_utf8_lossy(bytes);
                    let value: Value = serde_json::from_str(&data_str)?;
                    messages.push((entry.id, value));
                }
            }
        }
        Ok(messages)
    }

    fn ack(&self, topic: &str, message_id: &str) -> Result<(), QueueError> {
        let mut conn = self.client.get_connection()?;
        let _: () = conn.xack(topic, &self.consumer_group, &[message_id])?;
        Ok(())
    }

    fn extend_visibility(&self, _topic: &str, _message_id: &str, _seconds: u64) -> Result<(), QueueError> {
        // Redis Streams don't have visibility timeout — no-op.
        Ok(())
    }
}
