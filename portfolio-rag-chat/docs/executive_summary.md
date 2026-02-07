# Portfolio RAG Chat — Executive Summary

## What It Is

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about code repositories. Parses Rust, Python, and TypeScript codebases with tree-sitter, extracts docstrings, generates embeddings with FastEmbed, stores in LanceDB, and responds via Google Gemini.

## Why It Matters

- **Portfolio showcase**: Demonstrates Rust, RAG architecture, and chatbot development skills
- **Meta-project**: Can answer questions about itself and other portfolio projects
- **Code understanding**: Semantic search over function-level code chunks

## Key Features

- **Multi-language parsing**: Rust, Python, and TypeScript via tree-sitter AST queries
- **Docstring extraction**: `///` (Rust), `"""` (Python), `/** */` (TypeScript JSDoc) — enriches embeddings and LLM context
- **Incremental ingestion**: SHA256 file hashing skips unchanged files for fast re-indexing
- **4 chunk types**: Code functions, README files, Crate metadata, Module docs
- **Trait-based language abstraction**: Add new languages by implementing `LanguageHandler` trait
- **Vector search**: LanceDB with FastEmbed (BGE-small-en-v1.5, 384 dimensions)
- **LLM integration**: Google Gemini via rig-core
- **Web UI**: htmx + Askama templates for server-rendered chat interface

## Quick Start

```bash
# 1. Ingest repositories
docker-compose -f docker-compose-ingest.yaml up

# 2. Run query server
docker-compose up
```

Open http://localhost:3000 for the chat interface.

## Current State

**V1.5** — Indexing Foundation complete (97 tests, 0 warnings):
- `code-raptor`: Ingestion CLI — trait-based language handlers, incremental ingestion, docstring extraction
- `coderag-store`: Embedder + VectorStore — delete API, embedding model versioning
- `coderag-types`: Shared types — UUID chunk IDs, content hashes, nullable docstrings
- `portfolio-rag-chat`: Query API — docstrings displayed in LLM context

## Technology

- **Language**: Rust
- **Web Framework**: Axum 0.8
- **LLM**: Google Gemini (rig-core 0.27)
- **Vector Database**: LanceDB
- **Embeddings**: FastEmbed (BGE-small-en-v1.5)
- **Code Parsing**: tree-sitter (Rust, Python, TypeScript/TSX)
- **Frontend**: htmx + Askama templates
