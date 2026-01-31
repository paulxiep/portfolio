# Portfolio RAG Chat — Technical Summary

## Architecture

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

## Crate Responsibilities

| Crate | Purpose | Key Files |
|-------|---------|-----------|
| `code-raptor` | Ingestion CLI - tree-sitter parsing, chunk extraction | `ingestion/parser.rs`, `main.rs` |
| `coderag-store` | Embedder (FastEmbed) + VectorStore (LanceDB) | `embedder.rs`, `vector_store.rs` |
| `coderag-types` | Shared types (CodeChunk, ReadmeChunk, etc.) | `lib.rs` |
| `portfolio-rag-chat` | Query API - Axum server, LLM client, retrieval | `api/`, `engine/` |

## Query Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│   Axum Router   │  POST /api/chat
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retriever    │  Embeds query → searches 4 tables
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Builder │  Formats chunks into markdown
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Generator    │  Google Gemini via rig-core
└────────┬────────┘
         │
         ▼
    JSON/HTML Response with Sources
```

## Vector Schema (4 Tables)

| Table | Content | Embedding Input |
|-------|---------|-----------------|
| `code_chunks` | Functions, classes, structs | `identifier (language) + docstring + code` |
| `readme_chunks` | README.md files | `Project: name + content` |
| `crate_chunks` | Cargo.toml metadata | `Crate: name + description + dependencies` |
| `module_doc_chunks` | Module-level docs (`//!`) | `Module: name + doc_content` |

## Key Design Decisions

1. **Function-level chunking**: 1 function/class → 1 vector for precise retrieval
2. **4-table schema**: Separate tables for different content types with specialized formatting
3. **Mutex on Embedder**: Only resource needing synchronization (model weights)
4. **htmx frontend**: Server-rendered HTML with async updates, minimal JS
5. **Two-stage Docker**: Separate ingestion from query serving

## Retrieval Configuration

| Chunk Type | Default Limit |
|------------|---------------|
| Code | 5 |
| README | 2 |
| Crate | 3 |
| Module Docs | 3 |

## Build & Run

```bash
# Ingest repositories
docker-compose -f docker-compose-ingest.yaml up

# Run query server
docker-compose up

# Clean up
sh clean_docker.sh
```
