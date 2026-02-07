# CodeRAG: Architecture & Technical Design

A formal architecture document for the CodeRAG system, enforcing **Declarative**, **Modular**, and **Separation of Concerns** principles.

---

## 1. Overview

### 1.1 Document Purpose

This document serves as the authoritative technical reference for the CodeRAG system architecture. It is intended for:

- **Contributors**: Understanding system structure before implementing features
- **Reviewers**: Evaluating architectural decisions and code quality
- **Portfolio Reviewers**: Assessing technical design capabilities

This is a living document that evolves with the codebase. For feature roadmap, see [development_plan.md](../development_plan.md). For vision and differentiation, see [project-vision.md](../project-vision.md).

### 1.2 System Summary

**CodeRAG** is a Rust code understanding system that answers questions about code repositories using Retrieval-Augmented Generation (RAG).

**Value Proposition**: *Decouple knowledge from reasoning.*

Rich, structured retrieval amplifies *any* model—cheap or frontier. By offloading "what context is relevant" to the retrieval layer, the model's complexity budget can be spent on reasoning, multi-step workflows, or tool orchestration. This scales independently of model choice: better retrieval benefits Haiku and Opus alike.

### 1.3 Scope

**What CodeRAG does:**
- Semantic search over function-level code chunks
- Answer questions about unfamiliar/third-party codebases
- Multi-repo ingestion and querying
- Architecture-level understanding (folder/module/function hierarchy)

**Anti-goals (not competing here):**
- Real-time code completion (Copilot is better)
- Code generation from scratch
- IDE integration or inline quick fixes

---

## 2. Architectural Principles

### 2.1 Core Tenets

| Principle | Meaning | Enforcement |
|-----------|---------|-------------|
| **Declarative** | Describe *what*, not *how*. Config over code. Data-driven behavior. | Configuration schemas for retrieval limits, intent routing, ignored directories. Rules are data, not if-else chains. |
| **Modular** | Components are self-contained, swappable, and independently testable. | Trait-based interfaces. No cross-crate coupling except via shared types. Each crate has a single update frequency. |
| **SoC** (Separation of Concerns) | Each module has ONE job. No god objects. Clear boundaries. | code-raptor = indexing. portfolio-rag-chat = querying. coderag-types = type definitions only (no logic). coderag-store = embedding + persistence. |

### 2.2 Before Writing Code, Ask:

1. Am I describing behavior or implementing mechanics? (Declarative)
2. Can this be swapped out without ripple effects? (Modular)
3. Does this component have exactly one responsibility? (SoC)

### 2.3 Design Constraints

- **LanceDB as sole coupling point**: Producer (code-raptor) and consumer (portfolio-rag-chat) communicate via LanceDB schema only, not code imports
- **No shared runtime state**: Each crate can run independently; no in-memory coupling
- **Types-only shared crate**: `coderag-types` contains only struct definitions with serde, no business logic

---

## 3. C4 Model Diagrams

### 3.1 Level 1: Context Diagram

```
                    ┌─────────────────┐
                    │    Developer    │
                    │  (queries code) │
                    └────────┬────────┘
                             │ HTTP POST /api/chat
                             ▼
┌──────────────┐    ┌─────────────────────────┐    ┌──────────────┐
│     Code     │───▶│      CodeRAG System     │───▶│  Gemini LLM  │
│ Repositories │    │                         │    │   (Google)   │
│   (input)    │    │  code-raptor + chat     │    │              │
└──────────────┘    └─────────────────────────┘    └──────────────┘
                             │
                             │ Answers with sources
                             ▼
                    ┌─────────────────┐
                    │   Web Browser   │
                    │   (htmx UI)     │
                    └─────────────────┘
```

**External Actors:**
- Developer: Submits natural language queries about code
- CI/CD: Triggers ingestion on code changes (future)

**External Systems:**
- Gemini LLM: Google's language model for response generation
- Filesystem: Source code repositories to be indexed

### 3.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       Cargo Workspace                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐       ┌─────────────────────┐          │
│  │     code-raptor     │       │  portfolio-rag-chat │          │
│  │    (CLI Binary)     │       │   (Web Server)      │          │
│  │                     │       │                     │          │
│  │  - tree-sitter      │       │  - Axum 0.8         │          │
│  │  - walkdir          │       │  - rig-core (LLM)   │          │
│  │  - clap             │       │  - htmx + Askama    │          │
│  └──────────┬──────────┘       └──────────┬──────────┘          │
│             │ writes                       │ reads               │
│             ▼                              ▼                     │
│  ┌───────────────────────────────────────────────────┐          │
│  │                   coderag-store                    │          │
│  │  - Embedder (FastEmbed BGE-small-en-v1.5)         │          │
│  │  - VectorStore (LanceDB)                          │          │
│  └───────────────────────┬───────────────────────────┘          │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────┐          │
│  │                   coderag-types                    │          │
│  │  - CodeChunk, ReadmeChunk                         │          │
│  │  - CrateChunk, ModuleDocChunk                     │          │
│  │  - (serde only, no logic)                         │          │
│  └───────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   LanceDB Database    │
              │   (4 vector tables)   │
              │                       │
              │  - code_chunks        │
              │  - readme_chunks      │
              │  - crate_chunks       │
              │  - module_doc_chunks  │
              └───────────────────────┘
```

**Crate Responsibilities:**

| Crate | Responsibility | Update Frequency |
|-------|----------------|------------------|
| `code-raptor` | Parse code, extract chunks, embed, store | On code changes |
| `portfolio-rag-chat` | Query API, retrieval, LLM generation | On user queries |
| `coderag-store` | Embedding model, vector database ops | Shared infrastructure |
| `coderag-types` | Type definitions (data contracts) | On schema changes |

### 3.3 Level 3: Component Diagram

#### code-raptor Components

```
┌─────────────────────────────────────────────────────────────┐
│                        code-raptor                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │   main.rs    │  CLI entry point                          │
│  │              │  - ingest <repo_path> --db-path <path>    │
│  │              │  - status                                  │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              ingestion/mod.rs                     │       │
│  │  run_ingestion() - orchestrates full pipeline     │       │
│  │  - walkdir traversal                              │       │
│  │  - chunk extraction (code, readme, crate, docs)   │       │
│  │  - batch embedding (size: 25)                     │       │
│  │  - store upsert                                   │       │
│  └──────┬───────────────────────────────────────────┘       │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              ingestion/parser.rs                  │       │
│  │  CodeAnalyzer - tree-sitter wrapper               │       │
│  │  - analyze_content(source, lang) -> Vec<CodeChunk>│       │
│  │  - extract_module_docs(source) -> Option<String>  │       │
│  │  SupportedLanguage enum (Rust, Python)            │       │
│  │  parse_cargo_toml() - crate metadata              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### portfolio-rag-chat Components

```
┌─────────────────────────────────────────────────────────────┐
│                    portfolio-rag-chat                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │   main.rs    │  Server startup, environment loading       │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌───────────────────────────────────────────────────┐      │
│  │                    api/                            │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │      │
│  │  │ handlers.rs│  │  state.rs  │  │   dto.rs   │  │      │
│  │  │ POST /chat │  │ AppState   │  │ ChatRequest│  │      │
│  │  │ GET /proj  │  │ Mutex<Emb> │  │ ChatResp   │  │      │
│  │  └────────────┘  └────────────┘  └────────────┘  │      │
│  │  ┌────────────┐  ┌────────────┐                   │      │
│  │  │  error.rs  │  │   web.rs   │                   │      │
│  │  │  ApiError  │  │ Askama tpl │                   │      │
│  │  └────────────┘  └────────────┘                   │      │
│  └───────────────────────────────────────────────────┘      │
│         │                                                    │
│         ▼                                                    │
│  ┌───────────────────────────────────────────────────┐      │
│  │                   engine/                          │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │      │
│  │  │retriever.rs│  │ context.rs │  │generator.rs│  │      │
│  │  │ retrieve() │  │build_ctx() │  │ generate() │  │      │
│  │  │ embed+srch │  │format chunks│  │ Gemini API │  │      │
│  │  └────────────┘  └────────────┘  └────────────┘  │      │
│  │  ┌────────────┐                                   │      │
│  │  │  config.rs │  RetrievalConfig (limits)        │      │
│  │  └────────────┘                                   │      │
│  └───────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Interface Contracts

### 4.1 coderag-types — Domain Types

**Location:** [crates/coderag-types/src/lib.rs](../crates/coderag-types/src/lib.rs)

These types define the contract between producer (code-raptor) and consumer (portfolio-rag-chat). They contain **no business logic**, only serde serialization.

```rust
/// A function-level code snippet extracted from source code.
///
/// This is the primary unit of semantic search. Each function, class,
/// struct, trait, or other named code element becomes one CodeChunk.
///
/// # Example
/// ```
/// CodeChunk {
///     file_path: "/project/src/lib.rs".into(),
///     language: "rust".into(),
///     identifier: "process_data".into(),
///     node_type: "function_item".into(),
///     code_content: "fn process_data(input: &str) -> Result<()> { ... }".into(),
///     start_line: 42,
///     project_name: Some("my_project".into()),
///     docstring: Some("Processes input data and returns results.".into()),
/// }
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CodeChunk {
    /// Absolute or relative path to the source file
    pub file_path: String,

    /// Programming language: "rust" or "python"
    pub language: String,

    /// Name of the function, class, struct, trait, etc.
    pub identifier: String,

    /// Tree-sitter node type: "function_item", "class_definition", etc.
    pub node_type: String,

    /// The complete source code of the element
    pub code_content: String,

    /// 1-indexed line number where the element starts
    pub start_line: usize,

    /// Parent project name (e.g., "7_wonders", "quant-trading-gym")
    /// Used for filtering and attribution in responses
    pub project_name: Option<String>,

    /// Extracted documentation (/// for Rust, docstrings for Python)
    /// NOTE: Currently always None (extraction not yet implemented - V1.3)
    pub docstring: Option<String>,
}

/// A README.md file from a project.
///
/// Provides project-level context for overview queries.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReadmeChunk {
    /// Path to the README.md file
    pub file_path: String,

    /// Project name (required, not optional)
    pub project_name: String,

    /// Full content of the README
    pub content: String,
}

/// Rust crate metadata extracted from Cargo.toml.
///
/// Provides architectural context: what crates exist, their purposes,
/// and their local dependencies (workspace relationships).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrateChunk {
    /// Crate name from [package].name
    pub crate_name: String,

    /// Path to the crate directory containing Cargo.toml
    pub crate_path: String,

    /// Description from [package].description (optional)
    pub description: Option<String>,

    /// Local/workspace dependencies only (path dependencies)
    /// External crates.io dependencies are excluded
    pub dependencies: Vec<String>,

    /// Parent project context
    pub project_name: Option<String>,
}

/// Module-level documentation from //! comments.
///
/// Captures crate-level documentation typically found at the top of lib.rs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModuleDocChunk {
    /// Path to the lib.rs or module file
    pub file_path: String,

    /// Module/crate name
    pub module_name: String,

    /// Concatenated //! doc comments
    pub doc_content: String,

    /// Parent project context
    pub project_name: Option<String>,
}
```

### 4.2 coderag-store — Embedding & Storage

**Location:** [crates/coderag-store/src/](../crates/coderag-store/src/)

#### Embedder (embedder.rs)

```rust
/// Error types for embedding operations.
#[derive(Error, Debug)]
pub enum EmbedError {
    /// Model initialization failed (e.g., download error, OOM)
    #[error("failed to initialize embedding model: {0}")]
    Init(#[from] anyhow::Error),

    /// Embedding generation failed for input text
    #[error("embedding generation failed: {0}")]
    Embed(String),
}

/// FastEmbed wrapper for text embeddings.
///
/// Holds loaded model weights in memory. Must be protected by Mutex
/// for concurrent access in the web server.
///
/// # Model
/// Default: BGE-small-en-v1.5 (384 dimensions)
/// - Good balance of quality and speed for code
/// - ~33M parameters, runs locally without GPU
///
/// # Thread Safety
/// The underlying model is NOT thread-safe. Wrap in `Mutex<Embedder>`
/// for concurrent HTTP handlers.
pub struct Embedder {
    model: TextEmbedding,
    dimension: usize,
}

impl Embedder {
    /// Initialize with default model (BGE-small-en-v1.5, 384 dimensions).
    ///
    /// Downloads model on first run (~50MB), cached in `.fastembed_cache/`.
    ///
    /// # Errors
    /// Returns `EmbedError::Init` if model download or loading fails.
    pub fn new() -> Result<Self, EmbedError>;

    /// Initialize with a specific FastEmbed model.
    ///
    /// # Supported Models
    /// - `BGESmallENV15` (384 dim) - default, recommended
    /// - `BGEBaseENV15` (768 dim) - higher quality, slower
    /// - `BGELargeENV15` (1024 dim) - highest quality, slowest
    /// - `AllMiniLML6V2` (384 dim) - alternative lightweight
    pub fn with_model(model_name: EmbeddingModel) -> Result<Self, EmbedError>;

    /// Embed a single text string.
    ///
    /// Convenience wrapper around `embed_batch` for single inputs.
    ///
    /// # Returns
    /// 384-dimensional vector (f32) for default model.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed multiple texts in a single call (more efficient).
    ///
    /// Use batch size of 25 for memory efficiency during ingestion.
    ///
    /// # Returns
    /// Vector of embeddings, one per input text.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Get embedding dimension (384 for default model).
    pub fn dimension(&self) -> usize;
}
```

#### Text Formatting Functions (embedder.rs)

```rust
/// Formats a code chunk for embedding.
///
/// Concatenates identifier + language + docstring + code for richer semantic signal.
///
/// # Output Format
/// ```text
/// identifier (language)
/// docstring (if present)
/// code_content
/// ```
///
/// # Example
/// ```text
/// process_data (rust)
/// Processes input data and returns results
/// fn process_data(input: &str) -> Result<Output, Error> { ... }
/// ```
pub fn format_code_for_embedding(
    identifier: &str,
    language: &str,
    docstring: Option<&str>,
    code: &str,
) -> String;

/// Formats a README for embedding.
///
/// # Output Format
/// ```text
/// Project: project_name
/// content
/// ```
pub fn format_readme_for_embedding(project_name: &str, content: &str) -> String;

/// Formats a crate for embedding.
///
/// # Output Format
/// ```text
/// Crate: crate_name
/// description (if present)
/// Dependencies: dep1, dep2, dep3
/// ```
pub fn format_crate_for_embedding(
    crate_name: &str,
    description: Option<&str>,
    deps: &[String],
) -> String;

/// Formats module documentation for embedding.
///
/// # Output Format
/// ```text
/// Module: module_name
/// doc_content
/// ```
pub fn format_module_doc_for_embedding(module_name: &str, doc_content: &str) -> String;
```

#### VectorStore (vector_store.rs)

```rust
/// Error types for vector store operations.
#[derive(Error, Debug)]
pub enum StoreError {
    /// LanceDB operation failed
    #[error("database error: {0}")]
    Database(#[from] lancedb::Error),

    /// Arrow format conversion failed
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    /// Requested table doesn't exist (no data ingested yet)
    #[error("table '{0}' not found")]
    TableNotFound(String),

    /// Column type mismatch during extraction
    #[error("schema mismatch: {0}")]
    SchemaMismatch(String),
}

/// LanceDB-backed vector store for code chunks.
///
/// Manages 4 tables, each with its own schema:
/// - `code_chunks` - function-level code
/// - `readme_chunks` - README files
/// - `crate_chunks` - Cargo.toml metadata
/// - `module_doc_chunks` - //! documentation
///
/// # Connection
/// Uses local file-based LanceDB. Creates database directory if not exists.
///
/// # Thread Safety
/// `Connection` is internally reference-counted and safe for concurrent use.
pub struct VectorStore {
    conn: Connection,
    dimension: usize,
}

impl VectorStore {
    /// Connect to LanceDB at the given path.
    ///
    /// Creates the database directory if it doesn't exist.
    /// Does NOT create tables - tables are created on first upsert.
    ///
    /// # Arguments
    /// * `db_path` - Path to LanceDB directory (e.g., "data/portfolio.lance")
    /// * `embedding_dimension` - Must match Embedder dimension (384 for BGE-small)
    pub async fn new(db_path: &str, embedding_dimension: usize) -> Result<Self, StoreError>;

    // ═══════════════════════════════════════════════════════════════
    // Write Operations (used by code-raptor)
    // ═══════════════════════════════════════════════════════════════

    /// Insert code chunks with their embeddings.
    ///
    /// Creates `code_chunks` table on first call. Subsequent calls append data.
    ///
    /// # Returns
    /// Number of chunks inserted.
    pub async fn upsert_code_chunks(
        &self,
        chunks: &[CodeChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError>;

    /// Insert README chunks with their embeddings.
    pub async fn upsert_readme_chunks(
        &self,
        chunks: &[ReadmeChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError>;

    /// Insert crate chunks with their embeddings.
    pub async fn upsert_crate_chunks(
        &self,
        chunks: &[CrateChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError>;

    /// Insert module doc chunks with their embeddings.
    pub async fn upsert_module_doc_chunks(
        &self,
        chunks: &[ModuleDocChunk],
        embeddings: Vec<Vec<f32>>,
    ) -> Result<usize, StoreError>;

    // ═══════════════════════════════════════════════════════════════
    // Read Operations (used by portfolio-rag-chat)
    // ═══════════════════════════════════════════════════════════════

    /// Search code chunks by vector similarity.
    ///
    /// Uses cosine similarity (LanceDB default for normalized vectors).
    ///
    /// # Arguments
    /// * `query_embedding` - 384-dim query vector
    /// * `limit` - Maximum number of results (typically 5)
    pub async fn search_code(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<CodeChunk>, StoreError>;

    /// Search README chunks by vector similarity.
    pub async fn search_readme(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<ReadmeChunk>, StoreError>;

    /// Search crate chunks by vector similarity.
    pub async fn search_crates(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<CrateChunk>, StoreError>;

    /// Search module doc chunks by vector similarity.
    pub async fn search_module_docs(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<ModuleDocChunk>, StoreError>;

    /// Search all 4 tables and return combined results.
    ///
    /// This is the primary search method used by the retriever.
    ///
    /// # Returns
    /// Tuple of (code_chunks, readme_chunks, crate_chunks, module_doc_chunks)
    pub async fn search_all(
        &self,
        query_embedding: &[f32],
        code_limit: usize,
        readme_limit: usize,
        crate_limit: usize,
        module_doc_limit: usize,
    ) -> Result<(Vec<CodeChunk>, Vec<ReadmeChunk>, Vec<CrateChunk>, Vec<ModuleDocChunk>), StoreError>;

    /// List all unique project names in the database.
    ///
    /// Used by the `/projects` API endpoint.
    pub async fn list_projects(&self) -> Result<Vec<String>, StoreError>;
}
```

### 4.3 LanceDB Schema Contract

The LanceDB schema is the **sole coupling point** between code-raptor and portfolio-rag-chat.

#### code_chunks Table

| Column | Type | Nullable | Purpose |
|--------|------|----------|---------|
| `file_path` | UTF8 | NO | Full path to source file |
| `language` | UTF8 | NO | "rust" or "python" |
| `identifier` | UTF8 | NO | Function/class name |
| `node_type` | UTF8 | NO | Tree-sitter node type |
| `code_content` | UTF8 | NO | Source code snippet |
| `start_line` | UInt64 | NO | 1-indexed line number |
| `project_name` | UTF8 | YES | Parent project name |
| `docstring` | UTF8 | YES | Extracted documentation |
| `vector` | FixedSizeList(Float32, 384) | NO | Embedding vector |

#### readme_chunks Table

| Column | Type | Nullable |
|--------|------|----------|
| `file_path` | UTF8 | NO |
| `project_name` | UTF8 | NO |
| `content` | UTF8 | NO |
| `vector` | FixedSizeList(Float32, 384) | NO |

#### crate_chunks Table

| Column | Type | Nullable |
|--------|------|----------|
| `crate_name` | UTF8 | NO |
| `crate_path` | UTF8 | NO |
| `description` | UTF8 | YES |
| `dependencies` | UTF8 | YES | Comma-separated string |
| `project_name` | UTF8 | YES |
| `vector` | FixedSizeList(Float32, 384) | NO |

#### module_doc_chunks Table

| Column | Type | Nullable |
|--------|------|----------|
| `file_path` | UTF8 | NO |
| `module_name` | UTF8 | NO |
| `doc_content` | UTF8 | NO |
| `project_name` | UTF8 | YES |
| `vector` | FixedSizeList(Float32, 384) | NO |

### 4.4 code-raptor — Ingestion Pipeline

**Location:** [crates/code-raptor/src/](../crates/code-raptor/src/)

```rust
/// Supported programming languages for parsing.
///
/// Add new languages by:
/// 1. Add variant to this enum
/// 2. Implement `get_grammar()` with tree-sitter grammar
/// 3. Implement `query_string()` with S-expression patterns
#[derive(Debug, Clone, Copy)]
pub enum SupportedLanguage {
    Rust,
    Python,
}

impl SupportedLanguage {
    /// Detect language from file extension.
    ///
    /// # Supported Extensions
    /// - `.rs` → Rust
    /// - `.py` → Python
    /// - Others → None
    pub fn from_path(path: &Path) -> Option<Self>;

    /// Get tree-sitter grammar for this language.
    pub fn get_grammar(&self) -> tree_sitter::Language;

    /// Get language name string ("rust" or "python").
    pub fn name(&self) -> &'static str;

    /// Get tree-sitter S-expression query for code elements.
    ///
    /// # Rust Query Patterns
    /// - `function_item` - functions
    /// - `struct_item` - structs
    /// - `enum_item` - enums
    /// - `trait_item` - traits
    /// - `impl_item` - impl blocks
    /// - `type_item` - type aliases
    /// - `macro_definition` - macros
    ///
    /// # Python Query Patterns
    /// - `function_definition` - functions
    /// - `class_definition` - classes
    pub fn query_string(&self) -> &'static str;
}

/// Tree-sitter based code analyzer.
///
/// Extracts semantic code units (functions, classes, etc.) from source files.
pub struct CodeAnalyzer {
    parser: Parser,
}

impl CodeAnalyzer {
    /// Create a new analyzer with initialized parser.
    pub fn new() -> Self;

    /// Analyze source code and extract code chunks.
    ///
    /// # Arguments
    /// * `source` - Source code as string
    /// * `lang` - Programming language for grammar selection
    ///
    /// # Returns
    /// Vector of CodeChunk with `file_path` set to "<set_by_caller>".
    /// Caller must set `file_path` and `project_name` after extraction.
    ///
    /// # Deduplication
    /// Results are deduplicated by (identifier, start_line) to handle
    /// impl block methods being captured multiple times.
    pub fn analyze_content(&mut self, source: &str, lang: SupportedLanguage) -> Vec<CodeChunk>;

    /// Extract module-level //! documentation from Rust source.
    ///
    /// Only processes Rust files. Returns None for Python.
    ///
    /// # Example
    /// ```rust
    /// //! This is module documentation.
    /// //! It can span multiple lines.
    /// ```
    pub fn extract_module_docs(&mut self, source: &str) -> Option<String>;
}

/// Parse Cargo.toml and extract crate metadata.
///
/// # Returns
/// Tuple of (crate_name, description, local_dependencies)
/// where local_dependencies are path/workspace dependencies only.
///
/// # Example
/// ```toml
/// [package]
/// name = "my-crate"
/// description = "A utility crate"
///
/// [dependencies]
/// types = { path = "../types" }  # Included
/// serde = "1.0"                   # Excluded (external)
/// ```
pub fn parse_cargo_toml(content: &str) -> Option<(String, Option<String>, Vec<String>)>;
```

### 4.5 portfolio-rag-chat — Query Interface

**Location:** [src/](../src/)

#### HTTP API Contract

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/api/chat` | POST | `{"query": string}` | `{"answer": string, "sources": [...]}` |
| `/chat` | POST | Form: `query=...` | HTML fragment (htmx) |
| `/api/projects` | GET | - | `{"projects": [string], "count": number}` |
| `/health` | GET | - | `{"status": "ok"}` |

#### Engine Components

```rust
/// RAG pipeline configuration.
#[derive(Clone, Debug, Default)]
pub struct EngineConfig {
    pub retrieval: RetrievalConfig,
}

/// Retrieval limits per chunk type.
///
/// Default values are tuned for typical queries.
/// Future: Load from config file for declarative control.
#[derive(Clone, Debug)]
pub struct RetrievalConfig {
    /// Code chunks to retrieve (default: 5)
    pub code_limit: usize,

    /// README chunks to retrieve (default: 2)
    pub readme_limit: usize,

    /// Crate chunks to retrieve (default: 3)
    pub crate_limit: usize,

    /// Module doc chunks to retrieve (default: 3)
    pub module_doc_limit: usize,
}

/// Retrieved context from vector search.
#[derive(Debug)]
pub struct RetrievalResult {
    pub code_chunks: Vec<CodeChunk>,
    pub readme_chunks: Vec<ReadmeChunk>,
    pub crate_chunks: Vec<CrateChunk>,
    pub module_doc_chunks: Vec<ModuleDocChunk>,
}

/// Execute retrieval pipeline.
///
/// 1. Embed the query text
/// 2. Search all 4 LanceDB tables
/// 3. Return structured results
///
/// # Thread Safety
/// Requires mutable reference to Embedder (held in Mutex<Embedder>).
pub async fn retrieve(
    query: &str,
    embedder: &mut Embedder,
    store: &VectorStore,
    config: &RetrievalConfig,
) -> Result<RetrievalResult, EngineError>;

/// System prompt for LLM.
///
/// Instructs the model to:
/// - Use provided code snippets for accurate answers
/// - Reference project names and file paths
/// - Admit when context is insufficient
/// - Be concise but thorough
pub const SYSTEM_PROMPT: &str;

/// Format retrieved chunks into LLM context.
///
/// # Section Order
/// 1. Crate Structure (architectural overview)
/// 2. Module Documentation (high-level design)
/// 3. Relevant Code (implementation details)
/// 4. Project Documentation (README content)
///
/// # Truncation
/// README content truncated to 800 chars, module docs to 600 chars.
pub fn build_context(result: &RetrievalResult) -> String;

/// Build complete prompt for LLM.
///
/// # Format
/// ```text
/// {SYSTEM_PROMPT}
///
/// ---
///
/// {context}
///
/// ---
///
/// **Question:** {query}
/// ```
pub fn build_prompt(query: &str, context: &str) -> String;
```

---

## 5. Configuration Schema

### 5.1 Current State (Hardcoded)

Currently, configuration values are hardcoded in [src/engine/config.rs](../src/engine/config.rs):

```rust
impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            code_limit: 5,      // Hardcoded
            readme_limit: 2,    // Hardcoded
            crate_limit: 3,     // Hardcoded
            module_doc_limit: 3, // Hardcoded
        }
    }
}
```

Other hardcoded values:
- Embedding model: `BGESmallENV15` in `coderag-store/src/embedder.rs`
- Batch size: `25` in `code-raptor/src/main.rs`
- Ignored directories: `target`, `.git`, `node_modules` in ingestion
- LLM model: `gemini-2.0-flash` in environment variable

### 5.2 Proposed Declarative Configuration

To enforce the **Declarative** principle, migrate to TOML configuration:

```toml
# config/coderag.toml (proposed)

[embedding]
model = "BGESmallENV15"       # Options: BGESmallENV15, BGEBaseENV15, BGELargeENV15
dimension = 384                # Auto-set based on model
batch_size = 25                # Chunks per embedding batch

[retrieval]
code_limit = 5
readme_limit = 2
crate_limit = 3
module_doc_limit = 3

# Future: V2.3 intent routing
[retrieval.intent_routing]
enabled = false
[retrieval.intent_routing.overview]
readme = 3
crate = 5
code = 1
module_doc = 2
[retrieval.intent_routing.implementation]
code = 7
readme = 1
crate = 1
module_doc = 2

[ingestion]
ignored_dirs = [".git", "target", "node_modules", "__pycache__", ".venv"]
supported_languages = ["rust", "python"]  # Future: add "typescript"

[llm]
model = "gemini-2.0-flash"
system_prompt_file = "prompts/system.txt"  # Externalize prompt

[database]
path = "data/portfolio.lance"
```

### 5.3 Environment Variables

Current environment configuration (`.env`):

| Variable | Purpose | Required |
|----------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API authentication | Yes |
| `DATABASE_PATH` | LanceDB storage location | No (default: `data/portfolio.lance`) |
| `PORT` | HTTP server port | No (default: `3000`) |

---

## 6. Extension Points

### 6.1 Adding New Languages (V1.2)

**Location:** [crates/code-raptor/src/ingestion/parser.rs](../crates/code-raptor/src/ingestion/parser.rs)

```rust
// Step 1: Add variant to SupportedLanguage
pub enum SupportedLanguage {
    Rust,
    Python,
    TypeScript,  // Add new variant
}

// Step 2: Implement from_path detection
impl SupportedLanguage {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "ts" | "tsx" => Some(Self::TypeScript),  // Add extensions
            _ => None,
        }
    }
}

// Step 3: Add tree-sitter grammar
impl SupportedLanguage {
    pub fn get_grammar(&self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TSX.into(),  // Add grammar
        }
    }
}

// Step 4: Add query patterns
impl SupportedLanguage {
    pub fn query_string(&self) -> &'static str {
        match self {
            Self::TypeScript => {
                r#"(function_declaration name: (identifier) @name) @body
(arrow_function) @body
(method_definition name: (property_identifier) @name) @body
(class_declaration name: (type_identifier) @name) @body"#
            }
            // ...
        }
    }
}

// Step 5: Add to Cargo.toml
// [dependencies]
// tree-sitter-typescript = "0.23"
```

### 6.2 Adding New Chunk Types (A1: Hierarchy)

**Pattern for adding FolderChunk, FileChunk:**

1. **Define type** in `coderag-types/src/lib.rs`:
   ```rust
   pub struct FolderChunk {
       pub folder_path: String,
       pub summary: String,  // Auto-generated description
       pub file_count: usize,
       pub project_name: Option<String>,
   }
   ```

2. **Add formatting** in `coderag-store/src/embedder.rs`:
   ```rust
   pub fn format_folder_for_embedding(path: &str, summary: &str) -> String {
       format!("Folder: {}\n{}", path, summary)
   }
   ```

3. **Add schema** in `coderag-store/src/vector_store.rs`:
   - Add `FOLDER_TABLE` constant
   - Implement `folder_chunks_to_batch()`
   - Implement `extract_folder_chunks_from_batch()`
   - Add `upsert_folder_chunks()` and `search_folders()`

4. **Add extraction** in `code-raptor/src/ingestion/`:
   - Add folder summarization logic
   - Update `run_ingestion()` to process folders

5. **Update retrieval** in `portfolio-rag-chat/src/engine/`:
   - Add to `RetrievalResult`
   - Add to `search_all()` call
   - Add `format_folder_section()` in `context.rs`

### 6.3 Intent Classification (V2.2)

**Proposed location:** `src/engine/intent.rs`

```rust
/// Query intent categories for routing.
#[derive(Debug, Clone, Copy)]
pub enum QueryIntent {
    /// "What does this project do?" - README, CrateChunks
    Overview,

    /// "How does this function work?" - CodeChunks
    Implementation,

    /// "What calls this?" - Call graph (future)
    Relationship,

    /// "How does A compare to B?" - Cross-project
    Comparison,
}

/// Classify query intent using keyword heuristics.
///
/// # Declarative Approach
/// Rules should be loaded from config, not hardcoded.
pub fn classify_intent(query: &str, config: &IntentConfig) -> QueryIntent {
    // Data-driven: rules from config/intent_rules.toml
}

/// Adjust retrieval limits based on intent.
pub fn route_by_intent(intent: QueryIntent, base_config: &RetrievalConfig) -> RetrievalConfig {
    match intent {
        QueryIntent::Overview => RetrievalConfig {
            readme_limit: 3,
            crate_limit: 5,
            code_limit: 1,
            module_doc_limit: 2,
        },
        QueryIntent::Implementation => RetrievalConfig {
            code_limit: 7,
            readme_limit: 1,
            crate_limit: 1,
            module_doc_limit: 2,
        },
        // ...
    }
}
```

### 6.4 Hybrid Search (B1)

**Extension point:** `coderag-store/src/vector_store.rs`

```rust
/// Search with combined BM25 + vector similarity.
///
/// Uses Reciprocal Rank Fusion (RRF) to merge results.
///
/// # Future Implementation
/// LanceDB supports full-text search. Add BM25 index to tables,
/// then fuse with vector results.
pub async fn search_hybrid(
    &self,
    query_text: &str,
    query_embedding: &[f32],
    limit: usize,
    bm25_weight: f32,  // 0.0-1.0, higher = more lexical
) -> Result<Vec<CodeChunk>, StoreError>;
```

### 6.5 Call Graph (C1-C3)

**Future table schema for `call_edges`:**

| Column | Type | Purpose |
|--------|------|---------|
| `caller_id` | UTF8 | Composite key: `file_path::identifier` |
| `callee_id` | UTF8 | Composite key: `file_path::identifier` |
| `edge_type` | UTF8 | `same_file`, `cross_file`, `module_import` |
| `project_name` | UTF8 | For scoping |

**Query interface:**
```rust
/// Find all callers of a function.
pub async fn find_callers(
    &self,
    callee_id: &str,
) -> Result<Vec<CallEdge>, StoreError>;

/// Find call chain from entry point to target.
pub async fn trace_call_path(
    &self,
    from_id: &str,
    to_id: &str,
) -> Result<Vec<CallEdge>, StoreError>;
```

---

## 7. Architectural Decision Records (ADRs)

### ADR-001: Workspace Structure with 4 Crates

**Status:** Accepted

**Context:**
The system has two distinct use cases with different execution patterns:
- Ingestion: Batch processing on code changes
- Query serving: Real-time HTTP requests

These need to share type definitions and storage infrastructure without coupling their runtimes.

**Decision:**
Organize as a Cargo workspace with 4 crates:
- `code-raptor` - Ingestion CLI (independent binary)
- `portfolio-rag-chat` - Query API (independent binary)
- `coderag-store` - Shared embedding + storage
- `coderag-types` - Shared type definitions (no logic)

**Consequences:**
- (+) Clear SoC: each crate has one responsibility
- (+) Independent update frequencies
- (+) Can compile and test crates independently
- (+) `code-raptor` can be published to crates.io separately
- (-) More boilerplate (4 Cargo.toml files)
- (-) Must be careful not to leak implementation details through types crate

---

### ADR-002: LanceDB as Sole Coupling Point

**Status:** Accepted

**Context:**
Producer (code-raptor) and consumer (portfolio-rag-chat) need to exchange data without code-level coupling.

**Decision:**
All data exchange happens via LanceDB schema. The schema (4 tables with defined columns) is the contract. Neither crate imports the other.

**Consequences:**
- (+) No shared runtime state
- (+) Producer and consumer can run on different machines
- (+) Schema changes are explicit and versioned
- (+) Enables future multi-producer scenarios (CI/CD workers)
- (-) Schema is a critical interface - changes require coordination
- (-) No compile-time checks for schema compatibility

---

### ADR-003: Function-Level Chunking Granularity

**Status:** Accepted

**Context:**
Need to choose the semantic unit for embedding. Options:
- File-level: Coarse, loses precision
- Function-level: Balanced
- Line/block-level: Too fine, loses context

**Decision:**
1 function/class/struct/trait = 1 vector. Each named code element is a separate chunk.

**Consequences:**
- (+) Precise retrieval for "how does X work?" queries
- (+) Natural semantic boundary
- (+) Matches how developers think about code
- (-) Large functions may exceed context limits (planned: B3 chunking)
- (-) Cannot search within a function

---

### ADR-004: FastEmbed BGE-small-en-v1.5 for Embeddings

**Status:** Accepted

**Context:**
Need embedding model that:
- Runs locally (no API costs)
- Works well for code semantics
- Has reasonable memory footprint

**Decision:**
Use BGE-small-en-v1.5 via FastEmbed:
- 384 dimensions
- ~33M parameters
- Good code semantic capture
- ~50MB download, cached locally

**Consequences:**
- (+) No API costs or rate limits
- (+) Works offline
- (+) Consistent embeddings across runs
- (-) Quality below larger models (BGE-large, OpenAI)
- (-) First run requires model download

---

### ADR-005: Mutex on Embedder Only

**Status:** Accepted

**Context:**
Concurrent HTTP requests need shared access to resources:
- Embedder (model weights)
- VectorStore (database connection)
- LlmClient (API client)

**Decision:**
Only `Embedder` requires `Mutex`. VectorStore's `Connection` is internally reference-counted. LlmClient is stateless.

```rust
pub struct AppState {
    pub embedder: Mutex<Embedder>,  // Requires locking
    pub store: VectorStore,          // Internally Arc'd
    pub llm: LlmClient,              // Stateless
    pub config: EngineConfig,        // Immutable
}
```

**Consequences:**
- (+) Minimal lock contention
- (+) Most operations don't require locking
- (-) Embedder becomes bottleneck under high concurrency
- (Future) Could pool multiple Embedder instances

---

### ADR-006: Tree-sitter for Language Parsing

**Status:** Accepted

**Context:**
Need robust, multi-language AST parsing for code extraction. Options:
- Regex: Fragile, no semantic understanding
- Language-specific parsers: High maintenance
- Tree-sitter: Universal, grammar-based

**Decision:**
Use tree-sitter with per-language grammars:
- Declarative S-expression queries
- Consistent API across languages
- Easy to add new languages

**Consequences:**
- (+) Robust parsing (handles partial/invalid syntax)
- (+) Declarative query patterns
- (+) Large ecosystem of grammars
- (-) Dependency on external grammar crates
- (-) S-expression syntax has learning curve

---

## 8. Data Flow Diagrams

### 8.1 Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                            │
└──────────────────────────────────────────────────────────────────┘

[Repository Path]
       │
       ▼
┌──────────────────┐
│  walkdir + filter │  Ignore: .git, target, node_modules
└────────┬─────────┘
         │
         ├────────────────────────────────────────────────────────┐
         │                                                         │
         ▼                                                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  is_readme?         │  │  is_cargo_toml?     │  │  SupportedLanguage? │
│  README.md          │  │  Cargo.toml         │  │  .rs, .py           │
└────────┬────────────┘  └────────┬────────────┘  └────────┬────────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  ReadmeChunk        │  │  parse_cargo_toml() │  │  CodeAnalyzer       │
│  - file_path        │  │  -> CrateChunk      │  │  .analyze_content() │
│  - project_name     │  │                     │  │  -> Vec<CodeChunk>  │
│  - content          │  │  + extract_module   │  │                     │
└────────┬────────────┘  │  _docs() for lib.rs │  └────────┬────────────┘
         │               └────────┬────────────┘           │
         │                        │                         │
         └────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  format_*_for_embedding  │
                    │  (per chunk type)        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  Embedder.embed_batch    │
                    │  (batch_size: 25)        │
                    │  -> Vec<Vec<f32>>        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  VectorStore.upsert_*    │
                    │  (chunks + embeddings)   │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       LanceDB            │
                    │  - code_chunks           │
                    │  - readme_chunks         │
                    │  - crate_chunks          │
                    │  - module_doc_chunks     │
                    └──────────────────────────┘
```

### 8.2 Query Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                              │
└──────────────────────────────────────────────────────────────────┘

[User Query]  "How does the retriever work?"
       │
       ▼
┌──────────────────────────────────┐
│  POST /api/chat                  │
│  handlers::chat(query)           │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  Acquire Mutex<Embedder>         │
│  embedder.embed_one(query)       │
│  -> query_embedding [384]        │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  VectorStore.search_all(         │
│    query_embedding,              │
│    code_limit: 5,                │
│    readme_limit: 2,              │
│    crate_limit: 3,               │
│    module_doc_limit: 3           │
│  )                               │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  RetrievalResult {               │
│    code_chunks: [...],           │
│    readme_chunks: [...],         │
│    crate_chunks: [...],          │
│    module_doc_chunks: [...]      │
│  }                               │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  context::build_context(result)  │
│                                  │
│  ## Crate Structure              │
│  ## Module Documentation         │
│  ## Relevant Code                │  <- Markdown formatted
│  ## Project Documentation        │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  context::build_prompt(          │
│    query,                        │
│    context                       │
│  )                               │
│                                  │
│  {SYSTEM_PROMPT}                 │
│  ---                             │
│  {context}                       │
│  ---                             │
│  **Question:** {query}           │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  generator::generate(prompt)     │
│  -> Gemini API call              │
│  -> ~2-5s latency                │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│  ChatResponse {                  │
│    answer: "The retriever...",   │
│    sources: [                    │
│      { file, line, identifier }  │
│    ]                             │
│  }                               │
└──────────────────────────────────┘
```

---

## 9. Quality Attributes

### 9.1 Testability Strategy

| Level | What | How | Location |
|-------|------|-----|----------|
| **Unit tests** | Pure functions, parsers | Standard `#[test]` | Each crate's `mod tests` |
| **Integration tests** | End-to-end with fixtures | `#[ignore]` for I/O tests | `tests/` directories |
| **Hero queries** | Manual validation | 5-10 key queries per milestone | V1, V2 checkpoints |
| **Quantitative harness** | Automated recall@K | JSON test dataset + script | V3 infrastructure |

### 9.2 Performance Characteristics

| Operation | Typical Latency | Bottleneck |
|-----------|-----------------|------------|
| Embedding (single query) | ~50ms | CPU inference |
| Embedding (batch of 25) | ~200ms | CPU inference |
| Vector search (per table) | ~10ms | LanceDB I/O |
| Context building | <1ms | String formatting |
| LLM generation | 2-5s | Gemini API |
| **Full query (end-to-end)** | **3-6s** | **Dominated by LLM** |

### 9.3 Error Handling Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR HANDLING LAYERS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Crate-specific errors (thiserror)                        │   │
│  │  - EmbedError (embedding failures)                        │   │
│  │  - StoreError (database failures)                         │   │
│  │  - EngineError (pipeline failures)                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Application errors (anyhow)                              │   │
│  │  - Used in main.rs for startup failures                   │   │
│  │  - Wraps crate errors with context                        │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  HTTP errors (ApiError)                                   │   │
│  │  - Maps to HTTP status codes                              │   │
│  │  - 400: Bad request (invalid query)                       │   │
│  │  - 500: Internal error (embed/store/LLM failures)         │   │
│  │  - 503: Service unavailable (model not loaded)            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Appendices

### A. Glossary

| Term | Definition |
|------|------------|
| **Chunk** | Semantic unit of code/documentation for embedding (function, README, etc.) |
| **Embedding** | 384-dimensional vector representation of text |
| **Retrieval** | Vector similarity search to find relevant chunks |
| **Context** | Formatted chunks provided to LLM as reference material |
| **Intent** | Classified query type for routing (overview, implementation, etc.) |
| **RAPTOR** | Recursive Abstractive Processing for Tree-Organized Retrieval |

### B. References

- [development_plan.md](../development_plan.md) - V1-V3 roadmap + Tracks A/B/C
- [project-vision.md](../project-vision.md) - Full improvement ideas and differentiation
- [development_log.md](../development_log.md) - Version history (V0.1-V0.3)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [FastEmbed](https://github.com/qdrant/fastembed)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)

### C. Document Change Log

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-02-01 | Initial architecture document |
