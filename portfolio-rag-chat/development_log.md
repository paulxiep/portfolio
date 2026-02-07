# Development Log

## 2026-02-07: V1.5 Docstring Extraction (V1 Milestone Complete)

### Summary
Wired `extract_docstring()` into the parser pipeline and implemented it for all three language handlers (Rust, Python, TypeScript). The `docstring` field is now populated with real documentation instead of `None`. This completes the V1 (Indexing Foundation) milestone.

### Three Concerns (SoC)

1. **Parser wiring** — restructured `analyze_with_handler()` fold closure to call `handler.extract_docstring()` while the tree-sitter Node is still alive
2. **Handler implementations** — implemented for RustHandler (`///`, `#[doc]`) and PythonHandler (triple-quoted string in body via AST traversal)
3. **TypeScript verification** — V1.4's JSDoc extraction was dead code; V1.5 activated it via parser wiring and verified with pipeline tests

### Changes

| File | Change |
|------|--------|
| `crates/code-raptor/src/ingestion/parser.rs` | Extended fold tuple to `(String, String, String, usize, Option<String>)`, call `handler.extract_docstring()` inside fold, added 4 cross-language pipeline tests |
| `crates/code-raptor/src/ingestion/languages/rust.rs` | Implemented `extract_docstring` for `///` and `#[doc = "..."]`, added 7 unit tests |
| `crates/code-raptor/src/ingestion/languages/python.rs` | Implemented `extract_docstring` with AST traversal + `dedent_docstring()`, added 6 unit tests |
| `crates/code-raptor/src/ingestion/languages/typescript.rs` | Added 5 pipeline tests verifying JSDoc through `analyze_with_handler` |
| `crates/code-raptor/src/ingestion/language.rs` | Updated stale doc comments (V1.4 references to V1.5) |
| `src/engine/context.rs` (portfolio-rag-chat) | Added docstring display to `format_code_section()`, added context test |

### Extraction Strategies by Language

| Language | Strategy | Patterns |
|----------|----------|----------|
| Rust | Scan backwards from `node.start_position().row` | `///` outer doc, `#[doc = "..."]` attribute form. Skips `#[derive]`/`#[cfg]` |
| Python | AST traversal into function/class body | `"""..."""` and `'''...'''` triple-quoted strings. First `expression_statement` → `string` node. Dedented via `dedent_docstring()` |
| TypeScript | Scan backwards for JSDoc blocks (V1.4) | `/** ... */` multi/single-line. Filters out `@param`, `@returns` |

### Key Design Decisions

1. **Docstring extracted inside `fold` closure** — Node lifetime constraint: tree-sitter Nodes are only valid during fold iteration. Must extract before the tuple is created.
2. **`//!` (inner doc) excluded from RustHandler** — Already handled by `extract_module_docs()` in parser.rs. SoC: per-item docs vs module-level docs.
3. **Python uses AST, not line scanning** — Unlike Rust/TypeScript which scan backwards from the node, Python docstrings live inside the body. Tree-sitter AST traversal (`node → body → first expression_statement → string`) is the correct approach.
4. **Downstream already ready** — `format_code_for_embedding()`, Arrow schema, VectorStore roundtrip, and retriever all handled `Option<String>` docstrings since V1.1. Only context display needed a small addition.

### Gotchas Found During Implementation

1. **Node lifetime in `fold` closure** — Only primitives survived into the tuple. Must call `extract_docstring()` inside fold where Node is alive.
2. **Clippy: `if_same_then_else`** — Python's `parse_python_docstring()` had identical blocks for `"""` vs `'''` and `"` vs `'`. Consolidated with `||` conditions.
3. **Clippy: `collapsible_if`** — Rust's `#[doc]` parsing had nested `if let` chains. Collapsed with `let`-chaining.
4. **TypeScript arrow function `@body` offset** — `@body` captures `arrow_function` node, not `lexical_declaration`. Works for single-line declarations; accepted limitation for rare multi-line splits.

### Test Results

All 97 tests pass (0 warnings):
- code-raptor: 64 unit tests (7 new Rust + 6 new Python + 5 new TypeScript pipeline + 4 new cross-language pipeline)
- code-raptor: 9 integration tests
- coderag-store: 6 tests
- coderag-types: 9 tests
- portfolio-rag-chat: 9 tests (1 new docstring context test)
- `cargo fmt --all` clean
- `cargo clippy --workspace` clean (0 warnings)

### V1 Milestone Status

V1 (Indexing Foundation) is now complete:
- V1.1: Schema Foundation (chunk_id, content_hash, embedding_model_version)
- V1.2: LanguageHandler Trait (trait-based language abstraction)
- V1.3: Incremental Ingestion (file-level change detection, reconcile)
- V1.4: TypeScript Support (TypeScriptHandler + JSDoc extraction)
- V1.5: Docstring Extraction (wired pipeline + Rust/Python/TypeScript handlers)

**Crate:** code-raptor, portfolio-rag-chat

---

## 2026-02-07: V1.4 TypeScript Support

### Summary
Added TypeScript as a supported language using the V1.2 LanguageHandler trait. TypeScriptHandler uses the TSX grammar (superset of TS/JS/JSX/TSX) and captures 8 node types: functions, arrow functions (const/let/var), classes, methods, interfaces, type aliases, and enums. JSDoc extraction is implemented on the handler but remains unwired in parser.rs until V1.5 (SoC: handler declares capability, parser wires on its own schedule).

### Changes

| File | Change |
|------|--------|
| `crates/code-raptor/Cargo.toml` | Added `tree-sitter-typescript = "0.23"` |
| `crates/code-raptor/src/ingestion/languages/typescript.rs` | **NEW** — TypeScriptHandler + 15 unit tests |
| `crates/code-raptor/src/ingestion/languages/mod.rs` | Registered TypeScriptHandler in handler vec |
| `crates/code-raptor/src/ingestion/parser.rs` | Fixed `.js` test assertion (`is_none()` → `is_some()`), added `.go` for `is_none()` |
| `crates/code-raptor/src/ingestion/mod.rs` | Added `test_run_ingestion_typescript` integration test |
| `portfolio-rag-chat/development_plan.md` | Fixed V1.4/V1.5 ordering (was swapped) |

### TypeScript Query Patterns

| Pattern | Captures | Example |
|---------|----------|---------|
| `function_declaration` | Named functions | `function foo() {}` |
| `arrow_function` in `lexical_declaration` | Arrow functions (const/let) | `const foo = () => {}` |
| `arrow_function` in `variable_declaration` | Arrow functions (var) | `var foo = () => {}` |
| `class_declaration` | Classes | `class Foo {}` |
| `method_definition` | Class methods | `class { foo() {} }` |
| `interface_declaration` | Interfaces | `interface Foo {}` |
| `type_alias_declaration` | Type aliases | `type Foo = ...` |
| `enum_declaration` | Enums | `enum Foo { A, B }` |

Exported items (`export function foo()`, `export class Foo`) are captured by the base patterns — no separate export patterns needed.

### Key Design Decisions

1. **TSX grammar for all JS/TS**: `LANGUAGE_TSX` is a superset that handles `.ts`, `.tsx`, `.js`, `.jsx` — avoids maintaining separate grammars
2. **`language` field always "typescript"**: Accepted for V1.4. Not worth per-file language detection complexity
3. **`extract_docstring` implemented but dead**: SoC — handler declares JSDoc extraction capability, parser.rs hardcodes `docstring: None` until V1.5 wires it
4. **No redundant export patterns**: Tree-sitter queries match nested nodes, so `function_declaration` already matches inside `export_statement`. Dedup via `(identifier, start_line)` handles any duplicates

### Gotchas Found During Implementation

1. **`extract_docstring` is dead code until V1.5** — parser.rs line 96 hardcodes `docstring: None`. JSDoc tests must call `handler.extract_docstring()` directly, not expect docstrings in `CodeChunk` output from the pipeline
2. **`tree-sitter-typescript` version** — v0.23.2 uses `tree-sitter-language = "0.1"` as bridge crate, compatible with `tree-sitter = "0.26"` (same pattern as rust 0.24 and python 0.25)
3. **Existing test broke** — `parser.rs` had `assert!(handler_for_path(Path::new("test.js")).is_none())`, fixed to `is_some()` and added `test.go` for `is_none()`
4. **Missing `enum_declaration`** — original plan omitted TypeScript enums, added to query patterns
5. **Export patterns were redundant** — removed export-wrapped patterns, verified with `test_parse_exported_function`
6. **Clippy: identical `if` branches** — consolidated `line.starts_with("//")` branch into general break condition in `extract_docstring`

### Test Results

All 51 tests pass (0 warnings):
- code-raptor: 42 unit tests (15 new TypeScript + 27 existing)
- code-raptor: 9 integration tests (1 new TypeScript)
- `cargo fmt` clean
- `cargo clippy` clean

### Unit Tests (15 in `typescript.rs`)

| Test | Validates |
|------|-----------|
| `test_extensions` | All 4 extensions: `.ts`, `.tsx`, `.js`, `.jsx` |
| `test_parse_function_declaration` | `function foo()` → identifier "foo", node_type "function_declaration" |
| `test_parse_arrow_function` | `const add = () => ...` → identifier "add" |
| `test_parse_arrow_function_var` | `var legacy = () => {}` → identifier "legacy" |
| `test_parse_class_with_methods` | Class + methods captured separately |
| `test_parse_interface` | `interface User {}` → node_type "interface_declaration" |
| `test_parse_type_alias` | `type Result<T> = ...` → node_type "type_alias_declaration" |
| `test_parse_enum` | `enum Direction {}` → node_type "enum_declaration" |
| `test_parse_exported_function` | `export function` captured by base pattern |
| `test_parse_react_component` | TSX function component captured |
| `test_parse_arrow_react_component` | TSX arrow component captured |
| `test_jsdoc_single_line` | `/** text */` → extracts description (calls handler directly) |
| `test_jsdoc_multiline` | Multi-line JSDoc → description only, `@param`/`@returns` excluded |
| `test_jsdoc_no_doc` | No JSDoc → `None` |
| `test_jsdoc_with_export` | JSDoc before `export function` → validates no panic |

### Integration Test

`test_run_ingestion_typescript`: Creates temp directory with `.ts`, `.tsx`, `.js` files, runs `run_ingestion()`, verifies all three files produce chunks with `language: "typescript"`, correct identifiers, and normalized paths.

### Unblocks

- V1.5: Docstring Extraction (wire `handler.extract_docstring()` into parser pipeline for Rust, Python, TypeScript)

**Crate:** code-raptor

---

## 2026-02-06: V1.3 Incremental Ingestion

### Summary
Implemented file-level incremental ingestion with three-layer architecture (parse → reconcile → orchestrate). Changed files are detected by SHA256 hash, unchanged files are skipped entirely. Includes schema tightening: `project_name` became non-optional, paths normalized to relative forward-slash format, CrateChunk hash fixed to include description. Chunk IDs switched from random UUID v4 to deterministic `hash(file_path, content)` for Track C edge stability. Content hashing normalizes CRLF → LF for cross-OS consistency.

### Architecture: Three-Layer Separation

```
Layer 1 (sync):  run_ingestion()         → IngestionResult (parse code, no DB)
Layer 2 (sync):  reconcile()             → ReconcileResult (data comparison, no DB)
Layer 3 (async): main.rs orchestration   → apply diff (DB reads/writes)
```

### Changes by Crate

| Crate | Changes |
|-------|---------|
| coderag-types | `project_name: Option<String>` → `String` on all types; `deterministic_chunk_id()` replaces random UUID; `content_hash()` normalizes CRLF |
| coderag-store | Arrow schemas nullable → non-nullable for project_name; added `get_file_index()`, `delete_chunks_by_ids()`, `get_embedding_model_version()` |
| code-raptor | New `reconcile` module; `resolve_project_name()` + `normalize_path()` helpers; orchestration in main.rs with `--full`, `--dry-run`, `--project-name` flags |
| portfolio-rag-chat | Updated context.rs, dto.rs, handlers, templates for non-optional project_name |

### New Module: `ingestion/reconcile.rs`

Pure data comparison — no I/O, no DB handle, fully unit-testable.

| Type | Purpose |
|------|---------|
| `ExistingFileIndex` | Per-table file → (hash, chunk_ids) mapping from DB |
| `ReconcileResult` | What to insert + what to delete + stats |
| `DeletionsByTable` | Deletions partitioned by LanceDB table |
| `IngestionStats` | Counts: unchanged, changed, new, deleted files + chunks |

| Function | Purpose |
|----------|---------|
| `reconcile()` | Entry point: compares current ingestion against existing index |
| `reconcile_by_file()` | Generic: many chunks per file (CodeChunk, ReadmeChunk) |
| `reconcile_single_per_file()` | Generic: 1:1 file mapping (ModuleDocChunk) |
| `reconcile_crates()` | By `crate_name` instead of file path |

### New VectorStore Methods

| Method | Purpose |
|--------|---------|
| `get_file_index(table, project, path_col)` | Returns file → (hash, chunk_ids) for change detection |
| `delete_chunks_by_ids(table, chunk_ids)` | Batch delete with `IN (...)` predicate, batched in groups of 100 |
| `get_embedding_model_version(project)` | Query one chunk's model version for mismatch detection |

### CLI Flags

| Flag | Behavior |
|------|----------|
| `--full` | Force full re-index: delete all project chunks → re-embed → re-insert |
| `--dry-run` | Run reconcile, print stats, no DB changes (conflicts with `--full`) |
| `--project-name <name>` | Override project name for all chunks (defaults to directory inference) |

### Incremental Flow

1. Parse code into chunks (sync, no DB)
2. Initialize embedder + store
3. Check embedding model version (mismatch → bail with `--full` suggestion)
4. Build existing index from DB (async)
5. Reconcile: pure data comparison (sync)
6. Insert new chunks first (safer on crash: duplicates > missing data)
7. Delete old chunks

### Schema Tightening

| Change | Before | After |
|--------|--------|-------|
| `project_name` | `Option<String>` | `String` (non-optional) |
| Path storage | Absolute, OS-specific | Relative to repo root, forward slashes |
| CrateChunk hash | Omitted description | `crate_name:description:deps` |
| CodeChunk hash | Per-chunk content hash | File-level SHA256 (all chunks from same file share hash) |
| `chunk_id` | Random UUID v4 | Deterministic `hash(file_path, content)` — stable across re-indexing |
| `content_hash()` | Raw bytes | CRLF-normalized before hashing (cross-OS consistency) |
| `resolve_project_name()` | N/A | CLI override > subdir name > repo dir name > "unknown" |

### Test Results

All 58 tests pass (0 warnings):
- coderag-types: 9 tests (deterministic ID + CRLF normalization tests added)
- coderag-store: 6 tests
- code-raptor: 26 unit + 9 integration tests (deterministic ID stability test added)
- portfolio-rag-chat: 8 tests

### Integration Tests (`tests/incremental_ingestion.rs`)

| Test | Verifies |
|------|----------|
| `roundtrip_no_changes` | Re-ingest same files → 0 inserts/deletes |
| `detects_modified_file` | Modified file → correct replacement, untouched files skipped |
| `detects_deleted_file` | Deleted file → chunks removed by ID |
| `detects_new_file` | New file → chunks inserted |
| `mixed_changes` | Changed + deleted + new + unchanged simultaneously |
| `project_name_override_stable_reconcile` | `--project-name` override produces stable reconcile |
| `paths_normalized` | All paths relative, forward slashes |
| `file_level_content_hash` | All chunks from same file share content hash |
| `deterministic_ids_stable_across_runs` | Same input produces identical chunk_ids across runs |

### Migration

Existing databases incompatible (schema change: nullable → non-nullable). Requires full re-ingestion:
```bash
rm -rf data/portfolio.lance
cargo run --bin code-raptor -- ingest /path/to/projects --db-path data/portfolio.lance
```

Subsequent ingestions are incremental by default:
```bash
cargo run --bin code-raptor -- ingest /path/to/projects --db-path data/portfolio.lance
cargo run --bin code-raptor -- ingest /path/to/projects --db-path data/portfolio.lance --dry-run
cargo run --bin code-raptor -- ingest /path/to/projects --db-path data/portfolio.lance --full
```

---

## 2026-02-06: V1.2 LanguageHandler Refactor

### Summary
Replaced monolithic `SupportedLanguage` enum with trait-based `LanguageHandler` abstraction. Adding a new language is now "implement one trait + register" instead of modifying 4+ match statements. Pure refactor — ingestion output identical before and after.

### Changes

| Change | Detail |
|--------|--------|
| New trait | `LanguageHandler` with `name()`, `extensions()`, `grammar()`, `query_string()`, `extract_docstring()` (default None) |
| Implementations | `RustHandler`, `PythonHandler` |
| Registry | `handler_for_path()`, `handler_by_name()`, `supported_extensions()` via `OnceLock<Vec<Box<dyn LanguageHandler>>>` |
| CodeAnalyzer | `analyze_content(src, lang)` → `analyze_file(path, src)` + `analyze_with_handler(src, handler)` |
| Module docs | `extract_module_docs()` uses `RustHandler` directly (Rust-specific `//!` syntax) |
| Deleted | `SupportedLanguage` enum entirely removed |

### New File Structure

```
crates/code-raptor/src/ingestion/
├── mod.rs              # Re-exports, orchestration
├── parser.rs           # CodeAnalyzer (updated)
├── reconcile.rs        # Reconcile module (V1.3)
├── language.rs         # LanguageHandler trait (new)
└── languages/
    ├── mod.rs          # Registry + handler_for_path (new)
    ├── rust.rs         # RustHandler (new)
    └── python.rs       # PythonHandler (new)
```

### Key Design Decisions

1. **Trait with default `extract_docstring`**: Returns `None` now, V1.4 implements per-handler
2. **`OnceLock` registry**: Zero-cost after first access, thread-safe
3. **`analyze_file()` as primary API**: Auto-detects language from path, cleaner call sites
4. **Rust-specific module docs**: `extract_module_docs()` uses `RustHandler` directly rather than generalizing

### Unblocks

- V1.4: TypeScript Support (implement `TypeScriptHandler` + register)
- V1.5: Docstring Extraction (wire `extract_docstring()` into parser pipeline)

**Crate:** code-raptor

---

## 2026-02-04: V1.1 Schema Foundation

### Summary
Added foundational schema fields and APIs required for incremental ingestion (V1.3) and call graph (Track C). All 4 chunk types now have `chunk_id`, `content_hash`, and `embedding_model_version` fields. Delete API added to VectorStore.

### Changes by Crate

| Crate | Changes |
|-------|---------|
| coderag-types | Added 3 fields to all chunk types + `content_hash()` and `new_chunk_id()` helpers |
| coderag-store | Updated Arrow schemas, batch conversions, changed deps to `List<Utf8>`, added delete API |
| code-raptor | Updated ingestion to populate new fields |
| portfolio-rag-chat | Updated test fixtures |

### New Fields (all chunk types)

| Field | Type | Purpose |
|-------|------|---------|
| `chunk_id` | String (UUID v4) | Stable foreign key for Track C call graph edges |
| `content_hash` | String (SHA256) | Change detection for incremental ingestion |
| `embedding_model_version` | String | Prevents silent embedding inconsistency |

### New Dependencies

```toml
# coderag-types
sha2 = "0.10"
uuid = { version = "1.20", features = ["v4"] }

# coderag-store
arrow-buffer = "56.2"
```

### Delete API (VectorStore)

| Method | Purpose |
|--------|---------|
| `delete_chunks_by_file(table, path)` | For incremental file updates |
| `delete_chunks_by_project(table, project)` | For project removal |
| `delete_chunk_by_id(table, chunk_id)` | For individual chunk deletion |
| `get_chunks_by_file(table, path)` | Returns `(chunk_id, content_hash)` pairs for comparison |

### Schema Change: Dependencies

`crate_chunks.dependencies` changed from CSV string to `List<Utf8>` Arrow type. Enables future "what depends on X?" queries.

### Test Results

All 34 tests pass:
- coderag-types: 5 tests (hash/UUID helpers)
- coderag-store: 6 tests (batch conversion)
- code-raptor: 15 tests (parsing/ingestion)
- portfolio-rag-chat: 8 tests (context building)

### Migration

Existing databases incompatible. Requires full re-ingestion:
```bash
rm -rf data/portfolio.lance
cargo run --bin code-raptor -- ingest /path/to/projects --db-path data/portfolio.lance
```

### Unblocks

- V1.3: Incremental Ingestion (uses `content_hash` for change detection)
- Track C: Call Graph (uses `chunk_id` for foreign key references)

---

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
