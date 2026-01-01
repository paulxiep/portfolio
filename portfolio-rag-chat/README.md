# Portfolio RAG Chat

This is my first LLM-era portfolio project. The AI coding assistant has changed the portfolio game.

This project was (with planned Shuttle deployment removed) completed within 1 day. I'm starting to question what kind of problems I should solve to both be useful, add new skills and have some level of challenge.

**An alternative deployment to Shuttle will be figured out ASAP.**

**There'll be updates to improve ingestion** but the core engine are now functional.

## Log

- <2025-12-23>: Core engine functional.
- <2026-01-01>: Docker deployment added. To use, run `docker-compose up`. To clean, run `sh clean_docker.sh`.

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

For future improvements, summarised by Opus 4.5 thinking after I noted potential improvements.

In addition, the **code-topology-construction** mentioned in main readme will be used to enrich this chatbot's functionalities if successful.

---

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

#### Hierarchical Embedding
- Store multiple granularities: file → function → block
- Link via `parent_id`
- Query at appropriate level based on user intent

#### Call Graph Extraction
- Use tree-sitter to extract function calls
- Store as edges: `{ caller, callee, file }`
- Enables relationship queries and flow traversal
- Alternative: bake call context into embedding text ("Called by: X, Y")

#### Docstring Generation Module
- Decouple from ingestion pipeline
- LLM-generate on demand (Haiku recommended for cost/speed)
- Selective targeting: all, missing-only, single file, single function
- Change detection via code content hashing
- Persist separately so docstrings survive re-ingestion
- Re-embed after generation to update vector
