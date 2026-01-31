# Portfolio RAG Chat

### Keywords
*Extracted by GitHub Copilot*

- **Language:** `Rust`
- **Architecture & Patterns:** `Layered Architecture (API/Store/Ingestion)` · `Router Pattern` · `Handler Pattern` · `Shared State (Arc)` · `Repository Pattern` · `DTO Pattern` · `Modular Design` · `Pipeline Pattern (Ingest→Embed→Store)` · `Visitor Pattern (WalkDir)` · `Error Propagation (thiserror)`
- **LLM & RAG:** `RAG (Retrieval-Augmented Generation)` · `LLM Integration` · `Google Gemini` · `rig-core` · `Semantic Search` · `Chatbot`
- **Vector Database:** `LanceDB` · `FastEmbed` · `BGE Embeddings`
- **Code Analysis:** `Tree-sitter` · `AST Parsing` · `Code Parsing` · `Code Chunking` · `Multi-Language Support`
- **Web Framework:** `Axum` · `htmx` · `Askama Templates` · `Tower HTTP` · `CORS`
- **Async & Runtime:** `Tokio Runtime` · `Async Programming`
- **DevOps:** `Docker` · `Docker Compose`
- **Rust Ecosystem:** `tracing` · `Error Handling (anyhow/thiserror)` · `Serde`

---

This is my first LLM-era portfolio project. The AI coding assistant has changed the portfolio game.

This project's functional core engine was completed within 1 day. The expansion plan is underway on code embeddings side, called `code-raptor`.

**There'll be updates to improve ingestion** but the core engine is now functional.

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

## Current State

- Function-level chunking: 1 function/class → 1 vector (BGE-small, 384 dim)
- Supports Rust and Python via tree-sitter AST parsing
- `docstring` field exists but extraction not yet implemented

## Known Limitations

- **Granularity**: Cannot search within functions or at file/module level
- **Relationships**: No call graph — cannot answer "What calls X?"
- **Docstrings**: Undocumented functions have weaker semantic signal

## Planned Features

See [project-vision.md](project-vision.md) and [development_plan.md](development_plan.md) for roadmap.
