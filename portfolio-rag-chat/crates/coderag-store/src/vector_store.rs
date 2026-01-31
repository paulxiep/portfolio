use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
use futures::TryStreamExt;
use lancedb::{
    Connection, Table, connect,
    query::{ExecutableQuery, QueryBase},
};
use std::sync::Arc;
use thiserror::Error;

use coderag_types::{CodeChunk, CrateChunk, ModuleDocChunk, ReadmeChunk};

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("database error: {0}")]
    Database(#[from] lancedb::Error),

    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("table '{0}' not found")]
    TableNotFound(String),

    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
}

const CODE_TABLE: &str = "code_chunks";
const README_TABLE: &str = "readme_chunks";
const CRATE_TABLE: &str = "crate_chunks";
const MODULE_DOC_TABLE: &str = "module_doc_chunks";

/// LanceDB-backed vector store for code and readme chunks.
pub struct VectorStore {
    conn: Connection,
    dimension: usize,
}

impl VectorStore {
    /// Connect to LanceDB at the given path (creates if not exists).
    pub async fn new(db_path: &str, embedding_dimension: usize) -> Result<Self, StoreError> {
        // Ensure parent directory exists (important for Docker bind mounts)
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = connect(db_path).execute().await?;
        Ok(Self {
            conn,
            dimension: embedding_dimension,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    // ========================================================================
    // Write operations (used by code-raptor)
    // ========================================================================

    /// Insert code chunks with their embeddings. Creates table if needed.
    pub async fn upsert_code_chunks(
        &self,
        chunks: &[CodeChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let batch = code_chunks_to_batch(chunks, embeddings, self.dimension)?;
        let count = batch.num_rows();

        self.upsert_batch(CODE_TABLE, batch).await?;
        Ok(count)
    }

    /// Insert readme chunks with their embeddings. Creates table if needed.
    pub async fn upsert_readme_chunks(
        &self,
        chunks: &[ReadmeChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let batch = readme_chunks_to_batch(chunks, embeddings, self.dimension)?;
        let count = batch.num_rows();

        self.upsert_batch(README_TABLE, batch).await?;
        Ok(count)
    }

    /// Insert crate chunks with their embeddings. Creates table if needed.
    pub async fn upsert_crate_chunks(
        &self,
        chunks: &[CrateChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let batch = crate_chunks_to_batch(chunks, embeddings, self.dimension)?;
        let count = batch.num_rows();

        self.upsert_batch(CRATE_TABLE, batch).await?;
        Ok(count)
    }

    /// Insert module doc chunks with their embeddings. Creates table if needed.
    pub async fn upsert_module_doc_chunks(
        &self,
        chunks: &[ModuleDocChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let batch = module_doc_chunks_to_batch(chunks, embeddings, self.dimension)?;
        let count = batch.num_rows();

        self.upsert_batch(MODULE_DOC_TABLE, batch).await?;
        Ok(count)
    }

    // ========================================================================
    // Read operations (used by portfolio-rag-chat)
    // ========================================================================

    /// Search crate chunks by vector similarity.
    pub async fn search_crates(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<CrateChunk>, StoreError> {
        let table = self.get_table(CRATE_TABLE).await?;

        let results = table
            .vector_search(query_embedding.to_vec())?
            .limit(limit)
            .execute()
            .await?;

        batches_to_crate_chunks(results).await
    }

    /// Search module doc chunks by vector similarity.
    pub async fn search_module_docs(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<ModuleDocChunk>, StoreError> {
        let table = self.get_table(MODULE_DOC_TABLE).await?;

        let results = table
            .vector_search(query_embedding.to_vec())?
            .limit(limit)
            .execute()
            .await?;

        batches_to_module_doc_chunks(results).await
    }

    /// Search code chunks by vector similarity.
    pub async fn search_code(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<CodeChunk>, StoreError> {
        let table = self.get_table(CODE_TABLE).await?;

        let results = table
            .vector_search(query_embedding.to_vec())?
            .limit(limit)
            .execute()
            .await?;

        batches_to_code_chunks(results).await
    }

    /// Search readme chunks by vector similarity.
    pub async fn search_readme(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<ReadmeChunk>, StoreError> {
        let table = self.get_table(README_TABLE).await?;

        let results = table
            .vector_search(query_embedding.to_vec())?
            .limit(limit)
            .execute()
            .await?;

        batches_to_readme_chunks(results).await
    }

    /// Search both code and readme, return combined results.
    pub async fn search_all(
        &self,
        query_embedding: &[f32],
        code_limit: usize,
        readme_limit: usize,
        crate_limit: usize,
        module_doc_limit: usize,
    ) -> Result<
        (
            Vec<CodeChunk>,
            Vec<ReadmeChunk>,
            Vec<CrateChunk>,
            Vec<ModuleDocChunk>,
        ),
        StoreError,
    > {
        let code = self.search_code(query_embedding, code_limit).await?;
        let readme = self.search_readme(query_embedding, readme_limit).await?;
        let crates = self
            .search_crates(query_embedding, crate_limit)
            .await
            .unwrap_or_default();
        let module_docs = self
            .search_module_docs(query_embedding, module_doc_limit)
            .await
            .unwrap_or_default();
        Ok((code, readme, crates, module_docs))
    }

    pub async fn list_projects(&self) -> Result<Vec<String>, StoreError> {
        let table = match self.conn.open_table(CODE_TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()), // No data yet
        };

        // Query all project names
        let batches: Vec<RecordBatch> = table
            .query()
            .select(lancedb::query::Select::columns(&["project_name"]))
            .execute()
            .await?
            .try_collect()
            .await?;

        // Extract unique non-null project names
        let mut projects: Vec<String> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("project_name")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                    .map(|arr| {
                        (0..arr.len())
                            .filter(|&i| !arr.is_null(i))
                            .map(|i| arr.value(i).to_string())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect();

        // Deduplicate and sort
        projects.sort();
        projects.dedup();

        Ok(projects)
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    async fn upsert_batch(&self, table_name: &str, batch: RecordBatch) -> Result<(), StoreError> {
        let schema = batch.schema();

        // Try to open existing table, create if not exists
        match self.conn.open_table(table_name).execute().await {
            Ok(table) => {
                let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
                table.add(batches).execute().await?;
            }
            Err(_) => {
                let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
                self.conn
                    .create_table(table_name, batches)
                    .execute()
                    .await?;
            }
        }

        Ok(())
    }

    async fn get_table(&self, name: &str) -> Result<Table, StoreError> {
        self.conn
            .open_table(name)
            .execute()
            .await
            .map_err(|_| StoreError::TableNotFound(name.to_string()))
    }
}

// ============================================================================
// Arrow conversion functions (pure, no side effects)
// ============================================================================

fn code_chunks_to_batch(
    chunks: &[CodeChunk],
    embeddings: Vec<Vec<f32>>,
    dim: usize,
) -> Result<RecordBatch, StoreError> {
    use arrow_array::builder::FixedSizeListBuilder;
    use std::sync::Arc;

    let file_paths: StringArray = chunks.iter().map(|c| Some(c.file_path.as_str())).collect();
    let languages: StringArray = chunks.iter().map(|c| Some(c.language.as_str())).collect();
    let identifiers: StringArray = chunks.iter().map(|c| Some(c.identifier.as_str())).collect();
    let node_types: StringArray = chunks.iter().map(|c| Some(c.node_type.as_str())).collect();
    let code_contents: StringArray = chunks
        .iter()
        .map(|c| Some(c.code_content.as_str()))
        .collect();
    let start_lines: UInt64Array = chunks.iter().map(|c| Some(c.start_line as u64)).collect();
    let project_names: StringArray = chunks.iter().map(|c| c.project_name.as_deref()).collect();
    let docstrings: StringArray = chunks.iter().map(|c| c.docstring.as_deref()).collect();

    // Build fixed-size vector column
    let mut vector_builder =
        FixedSizeListBuilder::new(arrow_array::builder::Float32Builder::new(), dim as i32);

    for emb in &embeddings {
        vector_builder.values().append_slice(emb);
        vector_builder.append(true);
    }

    let vectors = vector_builder.finish();

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("language", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("identifier", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("node_type", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("code_content", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("start_line", arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new("project_name", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("docstring", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new(
            "vector",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                dim as i32,
            ),
            false,
        ),
    ]));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(file_paths),
            Arc::new(languages),
            Arc::new(identifiers),
            Arc::new(node_types),
            Arc::new(code_contents),
            Arc::new(start_lines),
            Arc::new(project_names),
            Arc::new(docstrings),
            Arc::new(vectors),
        ],
    )?)
}

fn readme_chunks_to_batch(
    chunks: &[ReadmeChunk],
    embeddings: Vec<Vec<f32>>,
    dim: usize,
) -> Result<RecordBatch, StoreError> {
    use arrow_array::builder::FixedSizeListBuilder;

    let file_paths: StringArray = chunks.iter().map(|c| Some(c.file_path.as_str())).collect();
    let project_names: StringArray = chunks
        .iter()
        .map(|c| Some(c.project_name.as_str()))
        .collect();
    let contents: StringArray = chunks.iter().map(|c| Some(c.content.as_str())).collect();

    let mut vector_builder =
        FixedSizeListBuilder::new(arrow_array::builder::Float32Builder::new(), dim as i32);

    for emb in &embeddings {
        vector_builder.values().append_slice(emb);
        vector_builder.append(true);
    }

    let vectors = vector_builder.finish();

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("project_name", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("content", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new(
            "vector",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                dim as i32,
            ),
            false,
        ),
    ]));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(file_paths),
            Arc::new(project_names),
            Arc::new(contents),
            Arc::new(vectors),
        ],
    )?)
}

fn crate_chunks_to_batch(
    chunks: &[CrateChunk],
    embeddings: Vec<Vec<f32>>,
    dim: usize,
) -> Result<RecordBatch, StoreError> {
    use arrow_array::builder::FixedSizeListBuilder;

    let crate_names: StringArray = chunks.iter().map(|c| Some(c.crate_name.as_str())).collect();
    let crate_paths: StringArray = chunks.iter().map(|c| Some(c.crate_path.as_str())).collect();
    let descriptions: StringArray = chunks.iter().map(|c| c.description.as_deref()).collect();
    // Store dependencies as comma-separated string
    let dependencies: StringArray = chunks
        .iter()
        .map(|c| Some(c.dependencies.join(",")).filter(|s: &String| !s.is_empty()))
        .collect();
    let project_names: StringArray = chunks.iter().map(|c| c.project_name.as_deref()).collect();

    let mut vector_builder =
        FixedSizeListBuilder::new(arrow_array::builder::Float32Builder::new(), dim as i32);

    for emb in &embeddings {
        vector_builder.values().append_slice(emb);
        vector_builder.append(true);
    }

    let vectors = vector_builder.finish();

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("crate_name", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("crate_path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("description", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("dependencies", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new("project_name", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new(
            "vector",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                dim as i32,
            ),
            false,
        ),
    ]));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(crate_names),
            Arc::new(crate_paths),
            Arc::new(descriptions),
            Arc::new(dependencies),
            Arc::new(project_names),
            Arc::new(vectors),
        ],
    )?)
}

fn module_doc_chunks_to_batch(
    chunks: &[ModuleDocChunk],
    embeddings: Vec<Vec<f32>>,
    dim: usize,
) -> Result<RecordBatch, StoreError> {
    use arrow_array::builder::FixedSizeListBuilder;

    let file_paths: StringArray = chunks.iter().map(|c| Some(c.file_path.as_str())).collect();
    let module_names: StringArray = chunks
        .iter()
        .map(|c| Some(c.module_name.as_str()))
        .collect();
    let doc_contents: StringArray = chunks
        .iter()
        .map(|c| Some(c.doc_content.as_str()))
        .collect();
    let project_names: StringArray = chunks.iter().map(|c| c.project_name.as_deref()).collect();

    let mut vector_builder =
        FixedSizeListBuilder::new(arrow_array::builder::Float32Builder::new(), dim as i32);

    for emb in &embeddings {
        vector_builder.values().append_slice(emb);
        vector_builder.append(true);
    }

    let vectors = vector_builder.finish();

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("module_name", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("doc_content", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("project_name", arrow_schema::DataType::Utf8, true),
        arrow_schema::Field::new(
            "vector",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                dim as i32,
            ),
            false,
        ),
    ]));

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(file_paths),
            Arc::new(module_names),
            Arc::new(doc_contents),
            Arc::new(project_names),
            Arc::new(vectors),
        ],
    )?)
}

async fn batches_to_code_chunks(
    stream: impl futures::Stream<Item = Result<RecordBatch, lancedb::Error>> + Unpin,
) -> Result<Vec<CodeChunk>, StoreError> {
    use futures::TryStreamExt;

    stream
        .map_err(StoreError::from)
        .try_fold(Vec::new(), |mut acc, batch| async move {
            acc.extend(extract_code_chunks_from_batch(&batch)?);
            Ok(acc)
        })
        .await
}

fn extract_code_chunks_from_batch(batch: &RecordBatch) -> Result<Vec<CodeChunk>, StoreError> {
    let col = |name: &str| -> Result<&StringArray, StoreError> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| StoreError::SchemaMismatch(name.into()))
    };

    let file_paths = col("file_path")?;
    let languages = col("language")?;
    let identifiers = col("identifier")?;
    let node_types = col("node_type")?;
    let code_contents = col("code_content")?;

    let start_lines = batch
        .column_by_name("start_line")
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| StoreError::SchemaMismatch("start_line".into()))?;

    // Optional columns
    let project_names = batch
        .column_by_name("project_name")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let docstrings = batch
        .column_by_name("docstring")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let nullable_string = |arr: Option<&StringArray>, i: usize| -> Option<String> {
        arr.filter(|a| !a.is_null(i))
            .map(|a| a.value(i).to_string())
    };

    let chunks = (0..batch.num_rows())
        .map(|i| CodeChunk {
            file_path: file_paths.value(i).to_string(),
            language: languages.value(i).to_string(),
            identifier: identifiers.value(i).to_string(),
            node_type: node_types.value(i).to_string(),
            code_content: code_contents.value(i).to_string(),
            start_line: start_lines.value(i) as usize,
            project_name: nullable_string(project_names, i),
            docstring: nullable_string(docstrings, i),
        })
        .collect();

    Ok(chunks)
}

async fn batches_to_readme_chunks(
    stream: impl futures::Stream<Item = Result<RecordBatch, lancedb::Error>> + Unpin,
) -> Result<Vec<ReadmeChunk>, StoreError> {
    use futures::TryStreamExt;

    stream
        .map_err(StoreError::from)
        .try_fold(Vec::new(), |mut acc, batch| async move {
            acc.extend(extract_readme_chunks_from_batch(&batch)?);
            Ok(acc)
        })
        .await
}

fn extract_readme_chunks_from_batch(batch: &RecordBatch) -> Result<Vec<ReadmeChunk>, StoreError> {
    let col = |name: &str| -> Result<&StringArray, StoreError> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| StoreError::SchemaMismatch(name.into()))
    };

    let file_paths = col("file_path")?;
    let project_names = col("project_name")?;
    let contents = col("content")?;

    let chunks = (0..batch.num_rows())
        .map(|i| ReadmeChunk {
            file_path: file_paths.value(i).to_string(),
            project_name: project_names.value(i).to_string(),
            content: contents.value(i).to_string(),
        })
        .collect();

    Ok(chunks)
}

async fn batches_to_crate_chunks(
    stream: impl futures::Stream<Item = Result<RecordBatch, lancedb::Error>> + Unpin,
) -> Result<Vec<CrateChunk>, StoreError> {
    use futures::TryStreamExt;

    stream
        .map_err(StoreError::from)
        .try_fold(Vec::new(), |mut acc, batch| async move {
            acc.extend(extract_crate_chunks_from_batch(&batch)?);
            Ok(acc)
        })
        .await
}

fn extract_crate_chunks_from_batch(batch: &RecordBatch) -> Result<Vec<CrateChunk>, StoreError> {
    let col = |name: &str| -> Result<&StringArray, StoreError> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| StoreError::SchemaMismatch(name.into()))
    };

    let crate_names = col("crate_name")?;
    let crate_paths = col("crate_path")?;

    let descriptions = batch
        .column_by_name("description")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let dependencies = batch
        .column_by_name("dependencies")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let project_names = batch
        .column_by_name("project_name")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let nullable_string = |arr: Option<&StringArray>, i: usize| -> Option<String> {
        arr.filter(|a| !a.is_null(i))
            .map(|a| a.value(i).to_string())
    };

    let chunks = (0..batch.num_rows())
        .map(|i| CrateChunk {
            crate_name: crate_names.value(i).to_string(),
            crate_path: crate_paths.value(i).to_string(),
            description: nullable_string(descriptions, i),
            dependencies: nullable_string(dependencies, i)
                .map(|s| s.split(',').map(String::from).collect())
                .unwrap_or_default(),
            project_name: nullable_string(project_names, i),
        })
        .collect();

    Ok(chunks)
}

async fn batches_to_module_doc_chunks(
    stream: impl futures::Stream<Item = Result<RecordBatch, lancedb::Error>> + Unpin,
) -> Result<Vec<ModuleDocChunk>, StoreError> {
    use futures::TryStreamExt;

    stream
        .map_err(StoreError::from)
        .try_fold(Vec::new(), |mut acc, batch| async move {
            acc.extend(extract_module_doc_chunks_from_batch(&batch)?);
            Ok(acc)
        })
        .await
}

fn extract_module_doc_chunks_from_batch(
    batch: &RecordBatch,
) -> Result<Vec<ModuleDocChunk>, StoreError> {
    let col = |name: &str| -> Result<&StringArray, StoreError> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| StoreError::SchemaMismatch(name.into()))
    };

    let file_paths = col("file_path")?;
    let module_names = col("module_name")?;
    let doc_contents = col("doc_content")?;

    let project_names = batch
        .column_by_name("project_name")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    let nullable_string = |arr: Option<&StringArray>, i: usize| -> Option<String> {
        arr.filter(|a| !a.is_null(i))
            .map(|a| a.value(i).to_string())
    };

    let chunks = (0..batch.num_rows())
        .map(|i| ModuleDocChunk {
            file_path: file_paths.value(i).to_string(),
            module_name: module_names.value(i).to_string(),
            doc_content: doc_contents.value(i).to_string(),
            project_name: nullable_string(project_names, i),
        })
        .collect();

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_code_chunk() -> CodeChunk {
        CodeChunk {
            file_path: "/test/main.rs".into(),
            language: "rust".into(),
            identifier: "test_func".into(),
            node_type: "function_item".into(),
            code_content: "fn test_func() {}".into(),
            start_line: 1,
            project_name: Some("test_project".into()),
            docstring: Some("A test function".into()),
        }
    }

    fn sample_readme_chunk() -> ReadmeChunk {
        ReadmeChunk {
            file_path: "/test/README.md".into(),
            project_name: "test_project".into(),
            content: "# Test Project\nThis is a test.".into(),
        }
    }

    fn fake_embedding(dim: usize) -> Vec<f32> {
        vec![0.1; dim]
    }

    #[test]
    fn test_code_chunks_to_batch() {
        let chunks = vec![sample_code_chunk()];
        let embeddings = vec![fake_embedding(384)];

        let batch = code_chunks_to_batch(&chunks, embeddings, 384).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 9); // 8 fields + vector
    }

    #[test]
    fn test_readme_chunks_to_batch() {
        let chunks = vec![sample_readme_chunk()];
        let embeddings = vec![fake_embedding(384)];

        let batch = readme_chunks_to_batch(&chunks, embeddings, 384).unwrap();

        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 4); // 3 fields + vector
    }

    #[test]
    fn test_code_batch_preserves_data() {
        let chunk = sample_code_chunk();
        let chunks = vec![chunk.clone()];
        let embeddings = vec![fake_embedding(384)];

        let batch = code_chunks_to_batch(&chunks, embeddings, 384).unwrap();

        let identifiers = batch
            .column_by_name("identifier")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(identifiers.value(0), chunk.identifier);
    }

    #[tokio::test]
    #[ignore = "requires filesystem, run with --ignored"]
    async fn test_vector_store_roundtrip() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.lance");

        let store = VectorStore::new(db_path.to_str().unwrap(), 384)
            .await
            .unwrap();

        let chunks = vec![sample_code_chunk()];
        let embeddings = vec![fake_embedding(384)];

        let count = store.upsert_code_chunks(&chunks, embeddings).await.unwrap();
        assert_eq!(count, 1);

        let results = store.search_code(&fake_embedding(384), 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].identifier, "test_func");
    }
}
