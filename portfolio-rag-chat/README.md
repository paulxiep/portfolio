# Portfolio RAG Chat

### Keywords
*Extracted by GitHub Copilot*

- **Language:** `Rust`
- **Architecture & Patterns:** `Layered Architecture (API/Store/Ingestion)` · `Trait-Based Abstraction (LanguageHandler)` · `Registry Pattern (OnceLock)` · `Three-Layer Pipeline (Parse→Reconcile→Orchestrate)` · `Router Pattern` · `Handler Pattern` · `Shared State (Arc)` · `Repository Pattern` · `DTO Pattern` · `Modular Design` · `Pipeline Pattern (Ingest→Embed→Store)` · `Visitor Pattern (WalkDir)` · `Error Propagation (thiserror)`
- **LLM & RAG:** `RAG (Retrieval-Augmented Generation)` · `LLM Integration` · `Google Gemini API` · `rig-core` · `Semantic Search` · `Chatbot`
- **Vector Database:** `LanceDB` · `FastEmbed` · `BGE Embeddings`
- **Code Analysis:** `Tree-sitter` · `AST Parsing` · `Code Chunking` · `Docstring Extraction` · `JSDoc Parsing` · `Multi-Language (Rust, Python, TypeScript)` · `Incremental Ingestion (SHA256)`
- **Web Framework:** `Axum` · `htmx` · `Askama Templates` · `Tower HTTP` · `CORS`
- **Async & Runtime:** `Tokio Runtime` · `Async Programming`
- **DevOps:** `Docker` · `Docker Compose`
- **Rust Ecosystem:** `tracing` · `Error Handling (anyhow/thiserror)` · `Serde` · `Let-Chaining`

---

A RAG chatbot that answers questions about code repositories. Parses Rust, Python, and TypeScript codebases with tree-sitter, extracts docstrings, generates embeddings, and responds via Google Gemini. Incremental ingestion skips unchanged files for fast iteration.

- [Executive Summary](docs/executive_summary.md)
- [Technical Summary](docs/technical_summary.md)

## Usage

1. `docker-compose -f docker-compose-ingest.yaml up`
2. `docker-compose up`

To clean, run `sh clean_docker.sh`.

## Development Roadmap

- [Development Log](development_log.md)
- [Development Plan](development_plan.md)
- [Project Vision](project-vision.md)

| Version | Date | Focus |
|---------|------|-------|
| **V0.1** | 2025-12-23 | MVP - Core engine |
| **V0.2** | 2026-01-01 | Docker deployment |
| **V0.3** | 2026-01-31 | Workspace restructuring |
| **V1.1** | 2026-02-04 | Schema foundation (UUID, content_hash, delete API) |
| **V1.2** | 2026-02-06 | LanguageHandler trait refactor |
| **V1.3** | 2026-02-06 | Incremental ingestion (SHA256, three-layer architecture) |
| **V1.4** | 2026-02-07 | TypeScript support (TSX grammar, JSDoc) |
| **V1.5** | 2026-02-07 | Docstring extraction (Rust, Python, TypeScript) |

## Purpose

- To practice and demonstrate familiarity with
  - Rust
  - RAG
  - Chatbot Application
- To function as meta-project
  - To answer questions about my portfolio

## Tech Stack

- **Backend**: Rust + Axum
- **LLM**: Google Gemini (via rig-core)
- **Vector DB**: LanceDB
- **Embeddings**: FastEmbed
- **Frontend**: htmx + Askama templates

---

## Guiding Principles

> **"Vertical slices, retrieval quality, code understanding"**

| Principle | Meaning |
|-----------|---------|
| **Vertical** | Build working end-to-end first, then deepen |
| **Retrieval** | Quality of retrieved context determines answer quality |
| **Understanding** | Goal is semantic code understanding, not just text search |

## Architecture

| Crate | Single Responsibility |
|-------|----------------------|
| `code-raptor` | Ingestion CLI — parsing, chunk extraction |
| `coderag-store` | Storage — embeddings, vector search |
| `coderag-types` | Shared types — no logic |
| `portfolio-rag-chat` | Query API — retrieval, LLM, web UI |

## Current State (V1 Complete)

- Function-level chunking: 1 function/class → 1 vector (BGE-small, 384 dim)
- Supports Rust, Python, and TypeScript via tree-sitter AST parsing
- Docstrings extracted: `///` (Rust), `"""` (Python), `/** */` (TypeScript JSDoc)
- Incremental ingestion: SHA256 file hashing, skips unchanged files
- 97 tests, 0 warnings

## Known Limitations

- **Granularity**: Cannot search within functions or at file/module level
- **Relationships**: No call graph — cannot answer "What calls X?"
- **Intent**: No query classification — all queries use the same retrieval strategy

## Planned Features

See [project-vision.md](project-vision.md) and [development_plan.md](development_plan.md) for roadmap.
