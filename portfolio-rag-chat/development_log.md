# Development Log

## 2026-01-31: V0.3 Workspace Restructuring

### Summary
Restructured monolithic crate into a Cargo workspace with 3 subcrates. Separates concerns between indexing (code-raptor), storage (coderag-store), and shared types (coderag-types). Root crate becomes pure query interface consumer.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cargo Workspace                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐   ┌─────────────────┐                  │
│  │   code-raptor   │   │ portfolio-rag-  │                  │
│  │   (Indexing)    │   │     chat        │                  │
│  │                 │   │  (Query API)    │                  │
│  │  - CLI          │   │                 │                  │
│  │  - tree-sitter  │   │  - Axum server  │                  │
│  │  - walkdir      │   │  - LLM client   │                  │
│  └────────┬────────┘   └────────┬────────┘                  │
│           │                     │                           │
│           ▼                     ▼                           │
│  ┌─────────────────────────────────────────┐                │
│  │           coderag-store                  │                │
│  │  - Embedder (FastEmbed)                 │                │
│  │  - VectorStore (LanceDB)                │                │
│  └─────────────────┬───────────────────────┘                │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │           coderag-types                  │                │
│  │  - CodeChunk, ReadmeChunk               │                │
│  │  - CrateChunk, ModuleDocChunk           │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### New Crates

| Crate | Purpose |
|-------|---------|
| `crates/code-raptor/` | Ingestion CLI - parses repositories, extracts chunks, stores in LanceDB |
| `crates/coderag-store/` | Storage layer - Embedder (FastEmbed) + VectorStore (LanceDB) |
| `crates/coderag-types/` | Shared domain types - CodeChunk, ReadmeChunk, CrateChunk, ModuleDocChunk |

### Files

| File | Purpose |
|------|---------|
| `crates/code-raptor/src/main.rs` | CLI entry point with `ingest` and `status` commands |
| `crates/code-raptor/src/lib.rs` | Library exports for ingestion module |
| `crates/code-raptor/src/ingestion/mod.rs` | Directory walker, chunk extraction pipeline |
| `crates/code-raptor/src/ingestion/parser.rs` | CodeAnalyzer with tree-sitter AST queries |
| `crates/coderag-store/src/lib.rs` | Library exports |
| `crates/coderag-store/src/embedder.rs` | FastEmbed wrapper (BGE-small-en-v1.5, 384-dim) |
| `crates/coderag-store/src/vector_store.rs` | LanceDB 4-table schema, upsert/search operations |
| `crates/coderag-types/src/lib.rs` | CodeChunk, ReadmeChunk, CrateChunk, ModuleDocChunk structs |

### Key Design Decisions

1. **Workspace structure**: Enables independent compilation and clearer ownership boundaries
2. **code-raptor as standalone CLI**: Can run ingestion separately from query server
3. **Shared types crate**: Single source of truth for domain models across crates
4. **Store abstraction**: Both code-raptor and portfolio-rag-chat consume coderag-store

---

## 2026-01-01: V0.2 Docker Deployment

### Summary
Added Docker containerization for deployment. Two-stage workflow: first run ingestion container to populate LanceDB, then run query server container.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Ingestion                                         │
│  ┌─────────────────────────────────────────┐                │
│  │  docker-compose-ingest.yaml             │                │
│  │  - Mounts source repos                  │                │
│  │  - Runs code-raptor ingest              │                │
│  │  - Outputs to shared LanceDB volume     │                │
│  └─────────────────────────────────────────┘                │
│                         │                                   │
│                         ▼                                   │
│               ┌─────────────────┐                           │
│               │  LanceDB Volume │                           │
│               └─────────────────┘                           │
│                         │                                   │
│                         ▼                                   │
│  Stage 2: Query Server                                      │
│  ┌─────────────────────────────────────────┐                │
│  │  docker-compose.yaml                    │                │
│  │  - Mounts LanceDB volume (read)         │                │
│  │  - Runs portfolio-rag-chat server       │                │
│  │  - Exposes port 3000                    │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build for Rust binary |
| `docker-compose.yaml` | Query server orchestration |
| `docker-compose-ingest.yaml` | Ingestion pipeline orchestration |
| `clean_docker.sh` | Cleanup script for containers/volumes |

### Key Design Decisions

1. **Two-stage workflow**: Separates expensive ingestion from lightweight query serving
2. **Shared volume**: LanceDB data persisted between containers
3. **Multi-stage Dockerfile**: Smaller final image, build dependencies not included

---

## 2025-12-23: V0.1 MVP - Core Engine Functional

### Summary
Implemented complete RAG chatbot MVP for code repositories. Parses Rust/Python codebases with tree-sitter, generates embeddings with FastEmbed, stores in LanceDB, and answers questions via Google Gemini. Web UI built with htmx + Askama.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Repository Files                                           │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │  CodeAnalyzer   │  tree-sitter AST parsing               │
│  │  (parser.rs)    │  Rust: function_item, struct_item, ... │
│  └────────┬────────┘  Python: function_definition, class_...│
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Chunk Extraction│  CodeChunk, ReadmeChunk,               │
│  │  (ingestion/)   │  CrateChunk, ModuleDocChunk            │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Embedder     │  FastEmbed BGE-small-en-v1.5           │
│  │  (embedder.rs)  │  384-dimensional vectors               │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   VectorStore   │  LanceDB with 4 tables:                │
│  │ (vector_store)  │  code_chunks, readme_chunks,           │
│  └─────────────────┘  crate_chunks, module_doc_chunks       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Query Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query                                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────┐                                        │
│  │   Axum Router   │  GET /, POST /api/chat, POST /chat     │
│  │    (api/)       │  GET /projects, GET /health            │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Retriever    │  Embeds query → searches 4 tables      │
│  │  (retriever.rs) │  Returns RetrievalResult               │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Context Builder │  Formats chunks into markdown          │
│  │  (context.rs)   │  Builds system + user prompt           │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Generator    │  Google Gemini via rig-core            │
│  │  (generator.rs) │  Returns answer + sources              │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   Web Response  │  htmx partial (HTML) or JSON           │
│  │    (web.rs)     │  Askama templates                      │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Files

**API Layer**
| File | Purpose |
|------|---------|
| `src/api/mod.rs` | Router setup with all endpoints |
| `src/api/handlers.rs` | Request handlers for chat, projects, health |
| `src/api/state.rs` | AppState with Embedder, VectorStore, LlmClient |
| `src/api/dto.rs` | ChatRequest, ChatResponse DTOs |
| `src/api/error.rs` | Error types and responses |
| `src/api/web.rs` | Askama templates, htmx integration |

**Engine Layer**
| File | Purpose |
|------|---------|
| `src/engine/mod.rs` | Engine module exports |
| `src/engine/retriever.rs` | Vector search across 4 tables |
| `src/engine/context.rs` | Prompt building, chunk formatting |
| `src/engine/generator.rs` | LLM response generation |
| `src/engine/config.rs` | RetrievalConfig, EngineConfig |

**Entry Point**
| File | Purpose |
|------|---------|
| `src/main.rs` | Server startup, environment loading |

### Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Axum 0.8 |
| LLM | Google Gemini (rig-core 0.27) |
| Vector Database | LanceDB |
| Embeddings | FastEmbed (BGE-small-en-v1.5, 384-dim) |
| Code Parsing | tree-sitter (Rust, Python) |
| Frontend | htmx + Askama templates |
| Async Runtime | Tokio 1.48 |

### Retrieval Configuration

| Chunk Type | Default Limit |
|------------|---------------|
| Code | 5 |
| README | 2 |
| Crate | 3 |
| Module Docs | 3 |

### Key Design Decisions

1. **Function-level chunking**: 1 function/class → 1 vector for precise retrieval
2. **4-table schema**: Separate tables for different content types with specialized formatting
3. **htmx frontend**: Server-rendered HTML with async updates, minimal JS
4. **Mutex on Embedder**: Only resource needing synchronization (model weights)
5. **rig-core for LLM**: Clean abstraction over Gemini API

### Known Limitations (documented for future work)

- `docstring` field exists but always `None` (extraction not implemented)
- No call graph or cross-function relationships
- No incremental ingestion (full re-scan each time)
- No hybrid search (semantic only, no BM25)
