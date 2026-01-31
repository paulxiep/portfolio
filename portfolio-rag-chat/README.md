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

This project's functional core engine was completed within 1 day. The expansion plan in underway on code embeddings side, called `code-raptor`.

**There'll be updates to improve ingestion** but the core engine are now functional.

## Log

- <2025-12-23>: Core engine functional.
- <2026-01-01>: Docker deployment added.
- <2026-01-31>: Split out `code-raptor` crate.

## Usage

1. `docker-compose -f docker-compose-ingest.yaml up`
2. `docker-compose up`

To clean, run `sh clean_docker.sh`.

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

## Architecture Notes

### Current Implementation

#### Embedding Approach
- Uses **text embeddings** (BGE-small via fastembed, 384 dimensions)
- 1 function/class → 1 vector
- Embedding input formatted as: identifier + language + docstring + code

#### Chunking Strategy
- Function-level granularity via tree-sitter parsing
- Supports Rust (`function_item`) and Python (`function_definition`, `class_definition`)
- Each `CodeChunk` stored with metadata: file path, language, identifier, line number

#### Docstrings
- `docstring` field exists in `CodeChunk` but is always `None`
- Formatter handles gracefully (skips if empty)

---

### Known Limitations

#### Granularity
- Cannot search for specific lines or code blocks within functions
- Cannot search at file/module level
- Large functions may exceed embedding model token limits

#### Composition & Relationships
- Functions are isolated vectors with no relationship data
- No call graph (caller/callee)
- Cannot answer: "What calls X?", "Show me the auth flow", "How do these connect?"

#### Docstrings
- Undocumented functions have weaker semantic signal
- No extraction of existing docstrings from source

---

### Potential Improvements

This project is being expanded in code embedding and ingestion under `code-raptor` crate. [The vision is here.](project-vision.md)
