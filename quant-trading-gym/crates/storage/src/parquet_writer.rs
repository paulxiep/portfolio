//! Buffered Parquet writer for ML training data.
//!
//! V5.4: Dual-file output for market and agent features.
//! - `{name}_market.parquet`: Market features (1 row per tick per symbol)
//! - `{name}_agents.parquet`: Agent features (1 row per agent per tick)

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int8Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::comprehensive_features::{AgentFeatures, MarketFeatures};

/// Buffer size before flushing to Parquet (number of records).
const AGENT_BUFFER_SIZE: usize = 100_000;
const MARKET_BUFFER_SIZE: usize = 1_000;

// ─────────────────────────────────────────────────────────────────────────────
// Record Types
// ─────────────────────────────────────────────────────────────────────────────

/// Market-level features (1 row per tick per symbol).
#[derive(Debug, Clone)]
pub struct MarketRecord {
    pub tick: u64,
    pub symbol: String,
    /// 42 market features (price, indicators, news).
    pub features: Vec<f64>,
}

/// Agent-level features (1 row per agent per tick).
#[derive(Debug, Clone)]
pub struct AgentRecord {
    pub tick: u64,
    pub symbol: String,
    pub agent_id: u64,
    pub agent_name: String,
    /// 10 agent features (position, cash, pnl, risk).
    pub features: Vec<f64>,
    pub action: i8,
    pub action_quantity: f64,
    pub action_price: f64,
    pub fill_quantity: f64,
    pub fill_price: f64,
    pub reward: f64,
    pub reward_normalized: f64,
    pub next_position: f64,
    pub next_pnl: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Dual Parquet Writer
// ─────────────────────────────────────────────────────────────────────────────

/// Dual Parquet writer for market and agent features.
///
/// Creates two files:
/// - `{base}_market.parquet`: Market features (42)
/// - `{base}_agents.parquet`: Agent features (10) + action/fill/reward columns
pub struct DualParquetWriter {
    market_writer: ParquetTableWriter,
    agent_writer: ParquetTableWriter,
    market_feature_names: Vec<String>,
    agent_feature_names: Vec<String>,
}

impl DualParquetWriter {
    /// Create a new dual writer.
    ///
    /// Creates two files: `{base}_market.parquet` and `{base}_agents.parquet`
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self, ParquetWriterError> {
        let base = base_path.as_ref();
        let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("data");
        let parent = base.parent().unwrap_or(Path::new("."));

        let market_path = parent.join(format!("{}_market.parquet", stem));
        let agent_path = parent.join(format!("{}_agents.parquet", stem));

        // Create parent directories
        std::fs::create_dir_all(parent).map_err(|e| ParquetWriterError::Io(e.to_string()))?;

        let market_feature_names: Vec<String> = MarketFeatures::feature_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let agent_feature_names: Vec<String> = AgentFeatures::feature_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let market_schema = Self::build_market_schema(&market_feature_names);
        let agent_schema = Self::build_agent_schema(&agent_feature_names);

        Ok(Self {
            market_writer: ParquetTableWriter::new(
                &market_path,
                market_schema,
                MARKET_BUFFER_SIZE,
            )?,
            agent_writer: ParquetTableWriter::new(&agent_path, agent_schema, AGENT_BUFFER_SIZE)?,
            market_feature_names,
            agent_feature_names,
        })
    }

    fn build_market_schema(feature_names: &[String]) -> Schema {
        let mut fields = vec![
            Field::new("tick", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
        ];
        for name in feature_names {
            fields.push(Field::new(name, DataType::Float64, true));
        }
        Schema::new(fields)
    }

    fn build_agent_schema(feature_names: &[String]) -> Schema {
        let mut fields = vec![
            Field::new("tick", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("agent_id", DataType::UInt64, false),
            Field::new("agent_name", DataType::Utf8, false),
        ];
        for name in feature_names {
            fields.push(Field::new(name, DataType::Float64, true));
        }
        fields.extend([
            Field::new("action", DataType::Int8, false),
            Field::new("action_quantity", DataType::Float64, false),
            Field::new("action_price", DataType::Float64, false),
            Field::new("fill_quantity", DataType::Float64, false),
            Field::new("fill_price", DataType::Float64, false),
            Field::new("reward", DataType::Float64, false),
            Field::new("reward_normalized", DataType::Float64, false),
            Field::new("next_position", DataType::Float64, true),
            Field::new("next_pnl", DataType::Float64, true),
        ]);
        Schema::new(fields)
    }

    /// Write a market record (1 per tick per symbol).
    pub fn write_market(&mut self, record: MarketRecord) -> Result<(), ParquetWriterError> {
        self.market_writer
            .write_market_record(record, &self.market_feature_names)
    }

    /// Write an agent record (many per tick).
    pub fn write_agent(&mut self, record: AgentRecord) -> Result<(), ParquetWriterError> {
        self.agent_writer
            .write_agent_record(record, &self.agent_feature_names)
    }

    /// Finish writing and close files.
    pub fn finish(self) -> Result<(usize, usize), ParquetWriterError> {
        let market_count = self.market_writer.finish()?;
        let agent_count = self.agent_writer.finish()?;
        eprintln!(
            "[DualParquetWriter] Market: {} rows, Agents: {} rows",
            market_count, agent_count
        );
        Ok((market_count, agent_count))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal Table Writer
// ─────────────────────────────────────────────────────────────────────────────

/// Generic Parquet table writer with buffering.
struct ParquetTableWriter {
    schema: Arc<Schema>,
    writer: Option<ArrowWriter<File>>,
    records_written: usize,
    buffer_size: usize,
    current_batch: Option<BatchBuilder>,
}

struct BatchBuilder {
    builders: Vec<ColumnBuilder>,
    num_rows: usize,
}

enum ColumnBuilder {
    UInt64(UInt64Builder),
    String(StringBuilder),
    Float64(Float64Builder),
    Int8(Int8Builder),
}

impl ParquetTableWriter {
    fn new<P: AsRef<Path>>(
        path: P,
        schema: Schema,
        buffer_size: usize,
    ) -> Result<Self, ParquetWriterError> {
        let file =
            File::create(path.as_ref()).map_err(|e| ParquetWriterError::Io(e.to_string()))?;
        let schema_arc = Arc::new(schema);

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(file, schema_arc.clone(), Some(props))
            .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;

        Ok(Self {
            schema: schema_arc,
            writer: Some(writer),
            records_written: 0,
            buffer_size,
            current_batch: None,
        })
    }

    fn write_market_record(
        &mut self,
        record: MarketRecord,
        feature_names: &[String],
    ) -> Result<(), ParquetWriterError> {
        let batch = self
            .current_batch
            .get_or_insert_with(|| BatchBuilder::new_market(self.buffer_size, feature_names.len()));

        // tick
        if let ColumnBuilder::UInt64(b) = &mut batch.builders[0] {
            b.append_value(record.tick);
        }
        // symbol
        if let ColumnBuilder::String(b) = &mut batch.builders[1] {
            b.append_value(&record.symbol);
        }
        // features
        for (i, &val) in record.features.iter().enumerate() {
            if let ColumnBuilder::Float64(b) = &mut batch.builders[2 + i] {
                if val.is_nan() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
        }

        batch.num_rows += 1;
        if batch.num_rows >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    fn write_agent_record(
        &mut self,
        record: AgentRecord,
        feature_names: &[String],
    ) -> Result<(), ParquetWriterError> {
        let batch = self
            .current_batch
            .get_or_insert_with(|| BatchBuilder::new_agent(self.buffer_size, feature_names.len()));

        let mut idx = 0;
        // tick
        if let ColumnBuilder::UInt64(b) = &mut batch.builders[idx] {
            b.append_value(record.tick);
        }
        idx += 1;
        // symbol
        if let ColumnBuilder::String(b) = &mut batch.builders[idx] {
            b.append_value(&record.symbol);
        }
        idx += 1;
        // agent_id
        if let ColumnBuilder::UInt64(b) = &mut batch.builders[idx] {
            b.append_value(record.agent_id);
        }
        idx += 1;
        // agent_name
        if let ColumnBuilder::String(b) = &mut batch.builders[idx] {
            b.append_value(&record.agent_name);
        }
        idx += 1;
        // features
        for &val in &record.features {
            if let ColumnBuilder::Float64(b) = &mut batch.builders[idx] {
                if val.is_nan() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
            idx += 1;
        }
        // action
        if let ColumnBuilder::Int8(b) = &mut batch.builders[idx] {
            b.append_value(record.action);
        }
        idx += 1;
        // action_quantity, action_price, fill_quantity, fill_price, reward, reward_normalized
        for val in [
            record.action_quantity,
            record.action_price,
            record.fill_quantity,
            record.fill_price,
            record.reward,
            record.reward_normalized,
        ] {
            if let ColumnBuilder::Float64(b) = &mut batch.builders[idx] {
                b.append_value(val);
            }
            idx += 1;
        }
        // next_position, next_pnl (nullable)
        for val in [record.next_position, record.next_pnl] {
            if let ColumnBuilder::Float64(b) = &mut batch.builders[idx] {
                if val.is_nan() {
                    b.append_null();
                } else {
                    b.append_value(val);
                }
            }
            idx += 1;
        }

        batch.num_rows += 1;
        if batch.num_rows >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), ParquetWriterError> {
        let batch = match self.current_batch.take() {
            Some(b) if b.num_rows > 0 => b,
            _ => return Ok(()),
        };

        let columns: Vec<ArrayRef> = batch.builders.into_iter().map(|b| b.finish()).collect();
        let record_batch = RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| ParquetWriterError::Arrow(e.to_string()))?;

        if let Some(ref mut writer) = self.writer {
            writer
                .write(&record_batch)
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }

        self.records_written += batch.num_rows;
        Ok(())
    }

    fn finish(mut self) -> Result<usize, ParquetWriterError> {
        self.flush()?;
        if let Some(writer) = self.writer.take() {
            writer
                .close()
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }
        Ok(self.records_written)
    }
}

impl BatchBuilder {
    fn new_market(capacity: usize, num_features: usize) -> Self {
        let mut builders = Vec::with_capacity(2 + num_features);
        builders.push(ColumnBuilder::UInt64(UInt64Builder::with_capacity(
            capacity,
        )));
        builders.push(ColumnBuilder::String(StringBuilder::with_capacity(
            capacity,
            capacity * 16,
        )));
        for _ in 0..num_features {
            builders.push(ColumnBuilder::Float64(Float64Builder::with_capacity(
                capacity,
            )));
        }
        Self {
            builders,
            num_rows: 0,
        }
    }

    fn new_agent(capacity: usize, num_features: usize) -> Self {
        let mut builders = Vec::with_capacity(13 + num_features);
        builders.push(ColumnBuilder::UInt64(UInt64Builder::with_capacity(
            capacity,
        ))); // tick
        builders.push(ColumnBuilder::String(StringBuilder::with_capacity(
            capacity,
            capacity * 8,
        ))); // symbol
        builders.push(ColumnBuilder::UInt64(UInt64Builder::with_capacity(
            capacity,
        ))); // agent_id
        builders.push(ColumnBuilder::String(StringBuilder::with_capacity(
            capacity,
            capacity * 32,
        ))); // agent_name
        for _ in 0..num_features {
            builders.push(ColumnBuilder::Float64(Float64Builder::with_capacity(
                capacity,
            )));
        }
        builders.push(ColumnBuilder::Int8(Int8Builder::with_capacity(capacity))); // action
        for _ in 0..8 {
            // action_qty, action_price, fill_qty, fill_price, reward, reward_norm, next_pos, next_pnl
            builders.push(ColumnBuilder::Float64(Float64Builder::with_capacity(
                capacity,
            )));
        }
        Self {
            builders,
            num_rows: 0,
        }
    }
}

impl ColumnBuilder {
    fn finish(self) -> ArrayRef {
        match self {
            Self::UInt64(mut b) => Arc::new(b.finish()),
            Self::String(mut b) => Arc::new(b.finish()),
            Self::Float64(mut b) => Arc::new(b.finish()),
            Self::Int8(mut b) => Arc::new(b.finish()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during Parquet writing.
#[derive(Debug)]
pub enum ParquetWriterError {
    Io(String),
    Parquet(String),
    Arrow(String),
}

impl std::fmt::Display for ParquetWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "I/O error: {}", msg),
            Self::Parquet(msg) => write!(f, "Parquet error: {}", msg),
            Self::Arrow(msg) => write!(f, "Arrow error: {}", msg),
        }
    }
}

impl std::error::Error for ParquetWriterError {}
