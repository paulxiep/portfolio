# Development Log

## 2026-02-08: MVP.3 Analysis Planning

### Summary
Implemented the analysis planning module (`autodash/planner.py`). Accepts a `DataProfile` and user questions, calls an LLM to produce structured `AnalysisStep` objects describing what to compute, validates column references and semantic compatibility, and returns validated steps ready for the explorer (MVP.4). This is the first LLM-calling node in the pipeline — patterns established here (closure factory DI, prompt separation, validation location) set precedent for MVP.4, 5, and 8.

### Architecture

```
                   ┌──────────┐
                   │  MVP.2   │  DataProfile
                   │  data.py │──────────────┐
                   └──────────┘              │
                                             ▼
                                      ┌─────────────┐
                   user questions ───▶│   MVP.3     │
                                      │ planner.py  │
                                      └──────┬──────┘
                                             │ list[AnalysisStep]
                                             ▼
                                      ┌─────────────┐
                                      │   MVP.4     │
                                      │ explorer.py │
                                      └─────────────┘

planner.py internals (SoC):

    _profile_summary()          ← format profile for LLM prompt
          │
    build_planning_prompt()     ← assemble full user prompt
          │
    plan_analysis()             ← orchestrate: prompt → LLM → parse (async)
          │
    parse_analysis_response()   ← parse JSON, validate, build steps
          │
    ├── _parse_single_step()    ← validate fields, aggregation enum
    └── _validate_step()        ← column existence + semantic compatibility
```

### New / Modified Files

| File | Purpose |
|------|---------|
| `autodash/planner.py` | **New.** Core planning logic: `_profile_summary`, `build_planning_prompt`, `_validate_step`, `_check_semantic_compatibility`, `parse_analysis_response`, `_parse_single_step`, `plan_analysis` |
| `autodash/prompts/analysis_planning.py` | **New.** `SYSTEM_PROMPT`, `OUTPUT_FORMAT`, `build_user_prompt()` — prompt templates separated from logic |
| `plotlint/core/errors.py` | **Modified.** Added `PlanningError(PipelineError)` |
| `autodash/pipeline.py` | **Modified.** Replaced `plan_node` stub with closure factory `_make_plan_node(config, llm_client)`. Fallback `plan_node` returns error dict when no LLM client |
| `tests/test_planner.py` | **New.** 27 tests: prompt construction (4), validation (11), response parsing (11), async integration with MockLLMClient (4) |
| `tests/test_planner_prompts.py` | **New.** 8 tests: template content, field coverage, assembly |
| `tests/test_pipeline_graph.py` | **Modified.** Updated `test_plan_stub` → `test_plan_fallback_returns_error`. Added `test_pipeline_with_mock_llm_plans_successfully` integration test |

### Key Design Decisions

1. **Closure factory for LLM node injection**: `_make_plan_node(config, llm_client)` captures dependencies via closure. LangGraph nodes must have signature `(state) -> dict` — closure is the standard Python DI for fixed-signature functions. LangGraph configurables were ruled out because `LLMClient` isn't serializable. Matches existing `_make_should_continue(config)` pattern in `plotlint/loop.py`. Sets precedent for MVP.4, 5, 8.

2. **All validation in `planner.py`, model stays pure data**: No `validate_columns` method on `AnalysisStep`. Column existence checks and semantic compatibility warnings all live in `_validate_step()` in `planner.py`. Models describe data, they don't validate against other models. Any module needing validation (e.g. DI-4.2 HITL) imports from `planner.py`.

3. **Per-module prompt formatting**: `_profile_summary(profile)` in `planner.py`, not on `DataProfile`. MVP.4's explorer needs a different view of the profile (column names + sample data vs types + stats). Each LLM-calling module builds its own prompt representation.

4. **Prompt templates separated from logic**: `autodash/prompts/analysis_planning.py` contains `SYSTEM_PROMPT`, `OUTPUT_FORMAT`, and `build_user_prompt()`. Prompt iteration doesn't touch `planner.py`. Aggregation type list is built from the `AggregationType` enum to stay in sync automatically.

5. **Hard vs soft validation**: Missing columns → `PlanningError` (hard, blocks execution). Semantic mismatches (e.g. SUM on text column) → logged warning, not blocking. The LLM may have creative intent, and MVP.4's explorer will fail gracefully if the operation is truly invalid.

6. **Reuse of `parse_json_from_response`**: `parse_analysis_response` delegates JSON extraction to the existing utility in `plotlint/core/parsing.py` rather than reimplementing fenced-block/embedded-JSON handling.

7. **Fallback plan node**: When `llm_client is None`, `_make_plan_node` returns the module-level `plan_node` which adds an error to state. Keeps `build_pipeline_graph()` callable without args (topology tests pass). Pipeline continues with errors accumulated rather than crashing.

### Validation Rules

`_validate_step(step, profile)` returns `(missing_columns, warnings)`:

| Check | Type | Columns Checked | Result |
|-------|------|----------------|--------|
| Column existence | Hard | `target_columns`, `group_by_columns`, `sort_by` | `PlanningError` if any missing |
| SUM/MEAN/MEDIAN/MIN/MAX on non-numeric | Soft | `target_columns` | Warning logged |
| CORRELATION on non-numeric | Soft | `target_columns` | Warning logged |
| TIME_SERIES on non-datetime | Soft | `target_columns` | Warning logged |
| COUNT/GROUP_BY/DISTRIBUTION/COMPARISON/CUSTOM | — | — | No semantic check |

### Test Results

All 159 tests pass (2.95s):
- `test_planner.py`: 27 tests (prompt construction, validation, parsing, integration)
- `test_planner_prompts.py`: 8 tests (template content, assembly)
- Previous MVP.1 + MVP.2 tests: all still passing (124 tests)

### Unblocks

- **MVP.4** (Data Exploration): Consumes `list[AnalysisStep]` to generate pandas code. `AnalysisStep.description`, `target_columns`, `aggregation`, `group_by_columns` provide structured context for code generation prompts.
- **MVP.5** (Chart Planning): Uses `AnalysisStep.description` and `rationale` for chart title context.
- **DI-1.1** (Multi-step planning): `plan_analysis()` already returns `list[AnalysisStep]` and accepts `max_steps` parameter. DI-1.1 calls with `max_steps=N` — no API change needed.
- **DI-4.2** (HITL checkpoints): `plan_analysis` returns data. The LangGraph pipeline node can `interrupt()` after this node and let the user modify the list. No change to `planner.py`.

**Packages:** autodash

## 2026-02-08: MVP.2 Data Intelligence

### Summary
Implemented the data loading and profiling module (`autodash/data.py`). Loads tabular data from CSV, Excel, and Parquet files via a protocol-based loader registry, profiles every column (nulls, cardinality, statistics), detects semantic types (numeric, categorical, datetime, text, boolean, identifier), and produces a `DataProfile` consumed by downstream pipeline nodes. Added `ProfileConfig` for configurable detection thresholds. Wired the real `load_node` into the pipeline graph, replacing the MVP.1 stub.

### Architecture

```
                        ┌─────────────────────────┐
                        │      load_and_profile()  │  ← top-level entry point
                        └────────┬────────────────┘
                                 │
                    ┌────────────┼────────────────┐
                    ▼                             ▼
            ┌──────────────┐              ┌──────────────┐
            │ load_dataframe│              │ profile_dataframe │
            │  (registry)   │              │  (profiling)      │
            └──────┬───────┘              └───────┬──────────┘
                   │                              │
          ┌────────┼────────┐            ┌────────┼────────┐
          ▼        ▼        ▼            ▼        ▼        ▼
      CsvLoader  Excel   Parquet    profile_column  detect_semantic_type
                 Loader   Loader         │
                                  _detect_date_granularity
```

### New / Modified Files

| File | Purpose |
|------|---------|
| `autodash/data.py` | **New.** DataLoader protocol, 3 loader implementations (CSV, Excel, Parquet), loader registry, `detect_semantic_type`, `_detect_date_granularity`, `profile_column`, `profile_dataframe`, `load_and_profile` |
| `autodash/config.py` | **Modified.** Added `ProfileConfig` frozen dataclass (thresholds for semantic type detection, sampling) and composed it into `PipelineConfig` |
| `autodash/pipeline.py` | **Modified.** `load_node` now calls `load_and_profile()` with `source_path` from state, returns `data_profile` or appends error |
| `tests/test_data/sample.csv` | **New.** 20-row test dataset with 6 columns (id, category, revenue, signup_date, is_active, notes) covering numeric, categorical, datetime, boolean, and text types |
| `tests/test_data_loading.py` | **New.** Loader dispatch, supports/rejects, protocol checks, CSV loading, error handling, custom loader registry |
| `tests/test_data_profiling.py` | **New.** Semantic type detection (12 cases), date granularity (6 cases), column profiling (6 cases), DataFrame profiling (5 cases), JSON round-trip, integration tests (4 cases) |
| `tests/test_pipeline_graph.py` | **Modified.** Updated to test real `load_node` with `source_path` in state |
| `pyproject.toml` | **Modified.** Added `openpyxl` and `pyarrow` as optional extras (`[excel]`, `[parquet]`) |

### Semantic Type Detection Decision Tree

`detect_semantic_type(series, unique_ratio, config)` classifies columns in this order:

| Priority | Condition | Result |
|----------|-----------|--------|
| 1 | `datetime` in dtype string | `DATETIME` |
| 2 | dtype is `bool` | `BOOLEAN` |
| 3 | Numeric dtype + values in {0, 1} | `BOOLEAN` |
| 3 | Numeric dtype (otherwise) | `NUMERIC` |
| 4a | Object + ≤2 unique + values in {"true","false","yes","no","0","1",...} | `BOOLEAN` |
| 4b | Object + ≥80% parse as dates (sample of 100) | `DATETIME` |
| 4c | Object + unique ratio ≥ 0.95 | `IDENTIFIER` |
| 4d | Object + ≤20 unique AND ratio ≤ 0.5 | `CATEGORICAL` |
| 4e | Fallback | `TEXT` |

All thresholds are configurable via `ProfileConfig`.

### Key Design Decisions

1. **`ProfileConfig` as a frozen dataclass**: All detection thresholds (cardinality ratios, max unique counts, date parse sample size, boolean string values) are configurable without modifying detection logic. Composed into `PipelineConfig` via `field(default_factory=...)`.

2. **Dual guard for categorical**: Both `unique_count <= max_unique` AND `unique_ratio <= max_cardinality` must hold. Prevents a 100-row dataset with 20 unique values (20% ratio) from being classified the same as a 1M-row dataset with 20 unique values (0.002% ratio).

3. **Lazy imports for optional dependencies**: `ExcelLoader.load()` imports `openpyxl` at call time; `ParquetLoader.load()` imports `pyarrow`. Raises `DataError` with install instructions if missing. Avoids hard dependency on heavy packages.

4. **`random_state=42` for date sampling**: `detect_semantic_type` samples object columns to attempt date parsing. Fixed seed ensures deterministic profiling (important for LangGraph replay and testing).

5. **`load_node` error accumulation**: On failure, appends to `state["errors"]` list rather than raising. Lets the pipeline record errors without crashing the graph.

6. **Deferred import in `load_node`**: `from autodash.data import load_and_profile` inside the function body avoids circular import risk and keeps `pipeline.py` lightweight for graph topology tests.

### Test Results

All tests pass across both MVP.1 and MVP.2:
- `test_data_loading.py`: 13 tests (loader supports/rejects, protocol checks, CSV loading, error handling, registry)
- `test_data_profiling.py`: 33 tests (semantic type detection, date granularity, column profiling, DataFrame profiling, JSON round-trip, integration)
- Previous MVP.1 tests: all still passing

### Unblocks

- **MVP.3** (Analysis Planning): `DataProfile.to_json()` provides column names, types, stats, and sample rows for LLM prompt context. `column_names()` enables validation of planned analysis steps.
- **MVP.4** (Data Exploration): `load_and_profile()` can be called to get both the profile and (separately) the DataFrame for code execution.
- **MVP.5** (Chart Planning): Column semantic types inform which chart types are appropriate (e.g., datetime columns → time series charts).

**Packages:** autodash

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
