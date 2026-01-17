//! Buffered Parquet writer for ML training data.
//!
//! V5.3: Writes feature records to Parquet files with Arrow schema.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Builder, Int8Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::features::FeatureExtractor;

/// Buffer size before flushing to Parquet (number of records).
const BUFFER_SIZE: usize = 10_000;

/// A single feature record for ML training.
#[derive(Debug, Clone)]
pub struct FeatureRecord {
    /// Simulation tick.
    pub tick: u64,
    /// Agent ID.
    pub agent_id: u64,
    /// Agent name.
    pub agent_name: String,
    /// Trading symbol.
    pub symbol: String,
    /// Pre-tick feature values.
    pub features: Vec<f64>,
    /// Action taken: -1 = sell, 0 = hold, 1 = buy.
    pub action: i8,
    /// Quantity attempted.
    pub action_quantity: f64,
    /// Limit price (if applicable).
    pub action_price: f64,
    /// Actual quantity filled.
    pub fill_quantity: f64,
    /// Average fill price.
    pub fill_price: f64,
    /// P&L change this tick.
    pub reward: f64,
    /// P&L change / initial_cash.
    pub reward_normalized: f64,
    /// Next tick mid price.
    pub next_mid_price: f64,
    /// Next tick position.
    pub next_position: f64,
    /// Next tick total PnL.
    pub next_pnl: f64,
}

impl Default for FeatureRecord {
    fn default() -> Self {
        Self {
            tick: 0,
            agent_id: 0,
            agent_name: String::new(),
            symbol: String::new(),
            features: Vec::new(),
            action: 0,
            action_quantity: 0.0,
            action_price: 0.0,
            fill_quantity: 0.0,
            fill_price: 0.0,
            reward: 0.0,
            reward_normalized: 0.0,
            next_mid_price: f64::NAN,
            next_position: f64::NAN,
            next_pnl: f64::NAN,
        }
    }
}

/// Buffered Parquet writer for feature records.
pub struct ParquetWriter {
    /// Arrow schema.
    schema: Arc<Schema>,
    /// Feature column names.
    feature_names: Vec<String>,
    /// Buffered records.
    buffer: Vec<FeatureRecord>,
    /// Arrow writer.
    writer: Option<ArrowWriter<File>>,
    /// Total records written (including buffer).
    records_written: usize,
}

impl ParquetWriter {
    /// Create a new Parquet writer.
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `extractor` - Feature extractor (for schema)
    pub fn new<P: AsRef<Path>>(
        path: P,
        extractor: &dyn FeatureExtractor,
    ) -> Result<Self, ParquetWriterError> {
        let feature_names: Vec<String> = extractor
            .feature_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let schema = Self::build_schema(&feature_names);
        let schema_arc = Arc::new(schema);

        // Create parent directories if needed
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| ParquetWriterError::Io(e.to_string()))?;
        }

        let file =
            File::create(path.as_ref()).map_err(|e| ParquetWriterError::Io(e.to_string()))?;

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(file, schema_arc.clone(), Some(props))
            .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;

        Ok(Self {
            schema: schema_arc,
            feature_names,
            buffer: Vec::with_capacity(BUFFER_SIZE),
            writer: Some(writer),
            records_written: 0,
        })
    }

    /// Build the Arrow schema for feature records.
    fn build_schema(feature_names: &[String]) -> Schema {
        let mut fields = vec![
            Field::new("tick", DataType::UInt64, false),
            Field::new("agent_id", DataType::UInt64, false),
            Field::new("agent_name", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
        ];

        // Add feature columns
        for name in feature_names {
            fields.push(Field::new(name, DataType::Float64, true));
        }

        // Add action columns
        fields.push(Field::new("action", DataType::Int8, false));
        fields.push(Field::new("action_quantity", DataType::Float64, false));
        fields.push(Field::new("action_price", DataType::Float64, false));

        // Add outcome columns
        fields.push(Field::new("fill_quantity", DataType::Float64, false));
        fields.push(Field::new("fill_price", DataType::Float64, false));
        fields.push(Field::new("reward", DataType::Float64, false));
        fields.push(Field::new("reward_normalized", DataType::Float64, false));

        // Add next-tick columns
        fields.push(Field::new("next_mid_price", DataType::Float64, true));
        fields.push(Field::new("next_position", DataType::Float64, true));
        fields.push(Field::new("next_pnl", DataType::Float64, true));

        Schema::new(fields)
    }

    /// Write a record to the buffer.
    ///
    /// Automatically flushes when buffer is full.
    pub fn write_record(&mut self, record: FeatureRecord) -> Result<(), ParquetWriterError> {
        self.buffer.push(record);

        if self.buffer.len() >= BUFFER_SIZE {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush buffered records to Parquet file.
    pub fn flush(&mut self) -> Result<(), ParquetWriterError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let batch = self.build_record_batch()?;
        if let Some(ref mut writer) = self.writer {
            writer
                .write(&batch)
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }

        self.records_written += self.buffer.len();
        self.buffer.clear();

        Ok(())
    }

    /// Build an Arrow RecordBatch from buffered records.
    fn build_record_batch(&self) -> Result<RecordBatch, ParquetWriterError> {
        let num_rows = self.buffer.len();

        // Build fixed columns
        let mut tick_builder = UInt64Builder::with_capacity(num_rows);
        let mut agent_id_builder = UInt64Builder::with_capacity(num_rows);
        let mut agent_name_builder = StringBuilder::with_capacity(num_rows, num_rows * 32);
        let mut symbol_builder = StringBuilder::with_capacity(num_rows, num_rows * 8);

        // Build feature columns
        let num_features = self.feature_names.len();
        let mut feature_builders: Vec<Float64Builder> = (0..num_features)
            .map(|_| Float64Builder::with_capacity(num_rows))
            .collect();

        // Build action columns
        let mut action_builder = Int8Builder::with_capacity(num_rows);
        let mut action_quantity_builder = Float64Builder::with_capacity(num_rows);
        let mut action_price_builder = Float64Builder::with_capacity(num_rows);

        // Build outcome columns
        let mut fill_quantity_builder = Float64Builder::with_capacity(num_rows);
        let mut fill_price_builder = Float64Builder::with_capacity(num_rows);
        let mut reward_builder = Float64Builder::with_capacity(num_rows);
        let mut reward_normalized_builder = Float64Builder::with_capacity(num_rows);

        // Build next-tick columns
        let mut next_mid_price_builder = Float64Builder::with_capacity(num_rows);
        let mut next_position_builder = Float64Builder::with_capacity(num_rows);
        let mut next_pnl_builder = Float64Builder::with_capacity(num_rows);

        // Populate builders
        for record in &self.buffer {
            tick_builder.append_value(record.tick);
            agent_id_builder.append_value(record.agent_id);
            agent_name_builder.append_value(&record.agent_name);
            symbol_builder.append_value(&record.symbol);

            // Features
            for (i, builder) in feature_builders.iter_mut().enumerate() {
                let value = record.features.get(i).copied().unwrap_or(f64::NAN);
                if value.is_nan() {
                    builder.append_null();
                } else {
                    builder.append_value(value);
                }
            }

            // Actions
            action_builder.append_value(record.action);
            action_quantity_builder.append_value(record.action_quantity);
            action_price_builder.append_value(record.action_price);

            // Outcomes
            fill_quantity_builder.append_value(record.fill_quantity);
            fill_price_builder.append_value(record.fill_price);
            reward_builder.append_value(record.reward);
            reward_normalized_builder.append_value(record.reward_normalized);

            // Next-tick
            if record.next_mid_price.is_nan() {
                next_mid_price_builder.append_null();
            } else {
                next_mid_price_builder.append_value(record.next_mid_price);
            }
            if record.next_position.is_nan() {
                next_position_builder.append_null();
            } else {
                next_position_builder.append_value(record.next_position);
            }
            if record.next_pnl.is_nan() {
                next_pnl_builder.append_null();
            } else {
                next_pnl_builder.append_value(record.next_pnl);
            }
        }

        // Build arrays
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(tick_builder.finish()),
            Arc::new(agent_id_builder.finish()),
            Arc::new(agent_name_builder.finish()),
            Arc::new(symbol_builder.finish()),
        ];

        for builder in feature_builders.iter_mut() {
            columns.push(Arc::new(builder.finish()));
        }

        columns.push(Arc::new(action_builder.finish()));
        columns.push(Arc::new(action_quantity_builder.finish()));
        columns.push(Arc::new(action_price_builder.finish()));
        columns.push(Arc::new(fill_quantity_builder.finish()));
        columns.push(Arc::new(fill_price_builder.finish()));
        columns.push(Arc::new(reward_builder.finish()));
        columns.push(Arc::new(reward_normalized_builder.finish()));
        columns.push(Arc::new(next_mid_price_builder.finish()));
        columns.push(Arc::new(next_position_builder.finish()));
        columns.push(Arc::new(next_pnl_builder.finish()));

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| ParquetWriterError::Arrow(e.to_string()))
    }

    /// Finish writing and close the file.
    ///
    /// Must be called to ensure all data is written.
    pub fn finish(mut self) -> Result<usize, ParquetWriterError> {
        self.flush()?;

        if let Some(writer) = self.writer.take() {
            writer
                .close()
                .map_err(|e| ParquetWriterError::Parquet(e.to_string()))?;
        }

        Ok(self.records_written)
    }

    /// Get the number of records written (including buffered).
    pub fn records_written(&self) -> usize {
        self.records_written + self.buffer.len()
    }

    /// Get the number of buffered records.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }
}

/// Errors that can occur during Parquet writing.
#[derive(Debug)]
pub enum ParquetWriterError {
    /// I/O error.
    Io(String),
    /// Parquet format error.
    Parquet(String),
    /// Arrow error.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comprehensive_features::ComprehensiveFeatures;
    use std::path::PathBuf;

    #[test]
    fn test_schema_field_count() {
        let extractor = ComprehensiveFeatures::new();
        let feature_names: Vec<String> = extractor
            .feature_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let schema = ParquetWriter::build_schema(&feature_names);

        // 4 fixed + 41 features + 3 action + 4 outcome + 3 next = 55
        let expected = 4 + feature_names.len() + 3 + 4 + 3;
        assert_eq!(schema.fields().len(), expected);
    }

    #[test]
    fn test_write_and_finish() {
        let extractor = ComprehensiveFeatures::new();
        let temp_path = PathBuf::from("target/test_parquet_writer.parquet");

        let mut writer = ParquetWriter::new(&temp_path, &extractor).unwrap();

        // Write a few records
        for i in 0..5 {
            let record = FeatureRecord {
                tick: i,
                agent_id: 1,
                agent_name: "TestAgent".to_string(),
                symbol: "AAPL".to_string(),
                features: vec![1.0; extractor.feature_names().len()],
                action: 1,
                action_quantity: 100.0,
                action_price: 150.0,
                fill_quantity: 100.0,
                fill_price: 150.0,
                reward: 10.0,
                reward_normalized: 0.01,
                next_mid_price: 151.0,
                next_position: 100.0,
                next_pnl: 10.0,
            };
            writer.write_record(record).unwrap();
        }

        let records = writer.finish().unwrap();
        assert_eq!(records, 5);

        // Clean up
        std::fs::remove_file(&temp_path).ok();
    }
}
