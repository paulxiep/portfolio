# Development Log

## 2026-02-08: MVP.1 Foundation + LangGraph Scaffold

### Summary
Established the foundation for both `plotlint` (standalone visual compliance engine) and `autodash` (end-to-end data-to-dashboard pipeline). Defined state schemas as LangGraph TypedDicts, built two graph skeletons with stub nodes (convergence loop + pipeline), implemented cross-cutting utilities in `plotlint/core/` (LLM client, sandbox, parsing, errors, config), and defined all model types upfront as shared contracts. Everything built in MVP.2-9 plugs into this scaffold.

### Architecture

```
                    plotlint (standalone)              autodash (pipeline)
                    ┌──────────────────────┐           ┌──────────────────────────┐
                    │  Convergence Loop     │           │  Pipeline Graph           │
                    │  render → inspect →   │           │  load → plan → explore →  │
                    │  decide(patch|stop)   │           │  chart → comply → output  │
                    └──────┬───────────────┘           └──────────┬───────────────┘
                           │                                      │
                           │    plotlint/core/ (shared foundation) │
                           │    ┌─────────────────────────────────┘
                           ▼    ▼
                    ┌────────────────────────────────────────────┐
                    │  errors.py  │ config.py │ llm.py           │
                    │  sandbox.py │ parsing.py                   │
                    └────────────────────────────────────────────┘
```

Two separate state machines in two separate packages. `plotlint` has ZERO imports from `autodash`. `autodash` imports from `plotlint.core`. The `comply_node` in the pipeline bridges them by invoking the convergence graph for each chart.

### New Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Hatchling build config, both packages, dependencies |
| `plotlint/core/errors.py` | `AutoDashError` hierarchy (12 exception classes) |
| `plotlint/core/config.py` | `LLMConfig`, `SandboxConfig` frozen dataclasses |
| `plotlint/core/llm.py` | `LLMClient` protocol + `AnthropicClient` with retry |
| `plotlint/core/parsing.py` | `parse_code_from_response()`, `parse_json_from_response()` |
| `plotlint/core/sandbox.py` | `execute_code()` subprocess sandbox with temp file IPC |
| `plotlint/models.py` | `Issue`, `InspectionResult`, `FixAttempt`, `RenderResult`, `ConvergenceState` |
| `plotlint/config.py` | `ConvergenceConfig` frozen dataclass |
| `plotlint/renderer.py` | `RendererBundle` shell (fields `Any` until MVP.6/7) |
| `plotlint/loop.py` | Convergence graph: `render`→`inspect`→decide(`patch`\|`stop`) |
| `autodash/models.py` | All autodash types (MVP.2-9) + `PipelineState` TypedDict |
| `autodash/config.py` | `PipelineConfig` composing all sub-configs |
| `autodash/pipeline.py` | Pipeline graph: `load`→`plan`→`explore`→`chart`→`comply`→`output` |
| `tests/test_core_errors.py` | Error hierarchy and isinstance checks (6 tests) |
| `tests/test_core_config.py` | Config defaults, custom, frozen immutability (5 tests) |
| `tests/test_core_parsing.py` | Fenced blocks, plain code, JSON extraction (12 tests) |
| `tests/test_core_sandbox.py` | Success, errors, timeout, return value (8 tests) |
| `tests/test_convergence_graph.py` | Graph topology, stubs, all 4 stop conditions (13 tests) |
| `tests/test_pipeline_graph.py` | Graph topology, stubs, full pipeline passthrough (9 tests) |
| `tests/test_models.py` | Serialization, properties, construction (16 tests) |

### Convergence Loop Stop Conditions

`_make_should_continue(config)` returns a closure that checks (in order):

| Condition | State Fields | Config Fields | Result |
|-----------|-------------|---------------|--------|
| Perfect score | `score >= target_score` | `target_score` | `"stop"` |
| Max iterations | `iteration >= max_iterations` | `max_iterations` | `"stop"` |
| Render error | `render_error is not None` | — | `"stop"` |
| Score stagnation | `score_history` window range < threshold | `stagnation_window`, `score_improvement_threshold` | `"stop"` |
| Otherwise | — | — | `"patch"` |

State `max_iterations` overrides config `max_iterations` when present — allowing per-invocation limits.

### Autodash Model Types Defined Upfront

All model types from MVP.2-9 are defined as full dataclasses in `autodash/models.py` now. This lets `PipelineState` type-check cleanly and avoids forward-reference issues. Construction logic (LLM prompts, parsing, validation) comes in later MVPs.

| MVP | Types |
|-----|-------|
| MVP.2 | `SemanticType`, `ColumnProfile`, `DataProfile` (with `to_json`, `from_json`, `column_names`, `get_column`) |
| MVP.3 | `AggregationType`, `AnalysisStep` |
| MVP.4 | `InsightResult` (with `TYPE_CHECKING` pandas guard, `__post_init__`, `to_prompt_context`) |
| MVP.5 | `ChartType`, `ChartPriority`, `RendererType`, `DataMapping`, `ChartSpec`, `ChartPlan` |
| MVP.9 | `OutputFormat`, `OutputArtifact`, `OutputResult` |

### Key Design Decisions

1. **Closure factory for `should_continue` (G1)**: LangGraph conditional edge functions only receive state. `_make_should_continue(config)` closes over `ConvergenceConfig` values, making stop conditions config-driven without putting all config into state.

2. **`field(default_factory=...)` for sub-configs (G2)**: `PipelineConfig` composes `ConvergenceConfig`, `LLMConfig`, and `SandboxConfig` — all mutable defaults requiring factory pattern.

3. **All autodash models defined upfront (G3)**: Rather than stub forward-references, the full dataclass definitions (fields + methods) go in now. Only construction logic (LLM interaction) comes later.

4. **`plotlint/core/` is real implementations, not stubs (G6)**: `parsing.py` (pure string manipulation), `sandbox.py` (subprocess + temp file IPC), `llm.py` (protocol + Anthropic client with retry), `errors.py` (exception hierarchy), `config.py` (frozen dataclasses) — all testable independently.

5. **`START`/`END` constants (G7)**: Uses current LangGraph API (`from langgraph.graph import START, END`) instead of deprecated `set_entry_point()`.

6. **`TYPE_CHECKING` guard for pandas (G12)**: `InsightResult.result_df` is `pd.DataFrame` but pandas is only imported under `TYPE_CHECKING`. Avoids ~200ms startup cost for all consumers.

7. **`RendererBundle` shell with `Any` fields (G4, G13)**: Prevents renderer/extractor mismatches structurally. Fields typed as `Any` until `Renderer`/`Extractor` protocols are defined in MVP.6/7.

8. **Subprocess sandbox with temp file IPC**: `execute_code()` writes code to temp file, runs in subprocess, captures return value via pickled temp file. Avoids stdout corruption from user code printing.

### Gotchas Found During Implementation

13 gotchas identified and resolved before implementation (documented in mvp.1.md):

1. **G1: `should_continue` can't access config** — Closure factory pattern
2. **G2: Mutable defaults in frozen dataclasses** — `field(default_factory=...)`
3. **G3: `PipelineState` references types from MVP.2-5** — Define all types upfront
4. **G4: `RendererBundle` references MVP.6** — Shell with `Any` fields
5. **G5: Missing `Optional` import** — `from __future__ import annotations` everywhere
6. **G6: `plotlint/core/` stubs vs real** — Real implementations
7. **G7: Deprecated LangGraph API** — `START`/`END` constants
8. **G8: No `pyproject.toml`** — Created with hatchling build system
9. **G9: `CompiledStateGraph` not imported** — `from langgraph.graph.state import CompiledStateGraph`
10. **G10: Missing `RenderResult`/`RenderStatus`** — Defined in MVP.1 since `ConvergenceState` references them
11. **G11: `figure_pickle` vs `figure_data` naming** — Intentional: state name vs renderer-agnostic name, render_node maps
12. **G12: `InsightResult` has `pd.DataFrame` field** — `TYPE_CHECKING` guard
13. **G13: `RendererBundle` needs undefined protocols** — `Any` until MVP.6/7

### Test Results

All 69 tests pass (2.15s):
- `test_core_errors.py`: 6 tests (hierarchy, isinstance, catching)
- `test_core_config.py`: 5 tests (defaults, custom, frozen)
- `test_core_parsing.py`: 12 tests (fenced blocks, plain code, JSON extraction, edge cases)
- `test_core_sandbox.py`: 8 tests (success, stdout, syntax error, runtime error, import error, timeout, return value, execution time)
- `test_convergence_graph.py`: 13 tests (topology, stubs, all 4 stop conditions, config-driven)
- `test_pipeline_graph.py`: 9 tests (topology, stubs, full pipeline passthrough)
- `test_models.py`: 16 tests (serialization roundtrip, properties, construction, frozen checks)

### Dependencies

| Package | Version Floor | Purpose |
|---------|--------------|---------|
| `langgraph` | >=1.0 | StateGraph, conditional edges |
| `pandas` | >=2.3 | DataFrame (autodash data pipeline) |
| `anthropic` | >=0.79 | LLM API (optional, `[llm]` extra) |
| `pytest` | >=9.0 | Testing (dev) |
| `pytest-asyncio` | >=1.3 | Async test support (dev) |

### Unblocks

MVP.2-9 plug into this scaffold:
- **MVP.2** (Data Intelligence): Replaces `load_node` stub, uses `DataProfile`/`ColumnProfile` already defined
- **MVP.3** (Analysis Planning): Replaces `plan_node` stub, uses `AnalysisStep` already defined
- **MVP.4** (Data Exploration): Replaces `explore_node` stub, uses `InsightResult` + `execute_code()`
- **MVP.5** (Chart Planning): Replaces `chart_node` stub, uses `ChartSpec`/`ChartPlan`
- **MVP.6** (Renderer): Replaces `render_node` stub, fills `RendererBundle`
- **MVP.7** (Inspector): Replaces `inspect_node` stub, uses `Issue`/`InspectionResult`
- **MVP.8** (Patcher): Replaces `patch_node` stub, uses `FixAttempt` + `parse_code_from_response()`
- **MVP.9** (Output): Replaces `output_node` stub, uses `OutputArtifact`/`OutputResult`

**Packages:** plotlint, autodash
