# Portfolio RAG Chat — Executive Summary

## What It Is

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about code repositories. Parses Rust and Python codebases with tree-sitter, generates embeddings with FastEmbed, stores in LanceDB, and responds via Google Gemini.

## Why It Matters

- **Portfolio showcase**: Demonstrates Rust, RAG architecture, and chatbot development skills
- **Meta-project**: Can answer questions about itself and other portfolio projects
- **Code understanding**: Semantic search over function-level code chunks

## Key Features

- **Multi-language parsing**: Rust and Python via tree-sitter AST queries
- **4 chunk types**: Code functions, README files, Crate metadata, Module docs
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

**V0.3** — Workspace restructured into 3 crates:
- `code-raptor`: Ingestion CLI
- `coderag-store`: Embedder + VectorStore
- `coderag-types`: Shared type definitions

## Technology

- **Language**: Rust
- **Web Framework**: Axum 0.8
- **LLM**: Google Gemini (rig-core 0.27)
- **Vector Database**: LanceDB
- **Embeddings**: FastEmbed (BGE-small-en-v1.5)
- **Code Parsing**: tree-sitter
- **Frontend**: htmx + Askama templates
