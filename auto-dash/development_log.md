# Development Log

## 2026-02-08: MVP.6 Renderer (matplotlib Sandbox)

### Summary
Implemented the matplotlib renderer (`plotlint/renderer.py`). Defines a `Renderer` protocol for chart renderers, a `MatplotlibRenderer` that executes chart code in a subprocess sandbox with Agg backend, captures the rendered Figure object (pickled) and PNG bytes, and returns a `RenderResult`. Wired the real `render_node` into the convergence loop, replacing the MVP.1 stub. The renderer reuses `plotlint/core/sandbox.py` for subprocess isolation — no separate worker script needed. The `RendererBundle` stub from MVP.1 is now functional with typed `Renderer` field and a working `matplotlib_bundle()` factory.

### Architecture

```
                     ┌──────────┐
                     │  MVP.5   │
                     │ charts.py│  ChartPlan.code
                     └────┬─────┘
                          │ source_code (str)
                          ▼
                   ┌─────────────┐       ┌──────────────────┐
                   │   MVP.6     │──────▶│ plotlint/core/   │
                   │ renderer.py │       │  sandbox.py      │
                   └──────┬──────┘       └──────────────────┘
                          │ RenderResult
                          │  ├─ png_bytes
                          │  ├─ figure_data (pickled Figure)
                          │  └─ status
                          ▼
                   ┌─────────────┐
                   │   MVP.7     │
                   │  inspector  │  unpickles Figure for bbox extraction
                   └─────────────┘

renderer.py internals (SoC):

    Renderer (Protocol)              ← interface for all renderers
          │
    MatplotlibRenderer               ← concrete implementation
    ├── render(code) → RenderResult  ← public API
    ├── _prepare_worker_code(code)   ← wraps user code with Agg + figure capture
    └── _to_render_result(exec)      ← maps ExecutionResult → RenderResult
          │
    RendererBundle                   ← pairs Renderer + Extractor
    matplotlib_bundle()              ← factory function
```

### New / Modified Files

| File | Purpose |
|------|---------|
| `plotlint/renderer.py` | **Rewritten.** `Renderer` protocol, `MatplotlibRenderer` dataclass, `_STATUS_MAP`, `RendererBundle` (typed `renderer` field), `matplotlib_bundle()` factory |
| `plotlint/loop.py` | **Modified.** Replaced `render_node` stub with `_make_render_node(renderer)` factory. `build_convergence_graph()` now creates default `matplotlib_bundle()` when no bundle provided. Added `asyncio.to_thread` for non-blocking subprocess execution |
| `tests/test_renderer.py` | **New.** 22 tests: protocol conformance (2), successful rendering (6), figure integrity + bbox thesis (4), error handling (5), monkey-patching safety (2), DPI validation (1), bundle construction (2) |
| `tests/test_convergence_graph.py` | **Modified.** Removed `test_render_stub_returns_empty`. Added `test_graph_with_explicit_bundle` |

### Code Wrapping Strategy

`_prepare_worker_code(user_code)` sandwiches user code between a preamble and postamble:

**Preamble** (before user code):
- `matplotlib.use('Agg')` — force headless backend before any matplotlib import
- `plt.show = lambda: None` — prevent blocking (defensive, no-op on Agg)
- `plt.close = lambda: None` — prevent figure loss before capture

**Postamble** (after user code):
- `fig = plt.gcf()` — get current figure
- Guard: `if not fig.get_axes()` → `__result__ = {"status": "no_figure"}`
- `fig.set_dpi(dpi)` — standardize DPI
- `pickle.dumps(fig)` → figure bytes
- `fig.savefig(buf, format='png', dpi=dpi)` → PNG bytes (no `bbox_inches='tight'`)
- Set `__result__` dict → captured by sandbox's `_WORKER_TEMPLATE`

Preamble and postamble are `textwrap.dedent`'ed separately from user code to avoid indentation conflicts with multi-line user code.

### ExecutionResult → RenderResult Mapping

Declarative `_STATUS_MAP` dict maps sandbox statuses to render statuses:

| ExecutionStatus | RenderStatus | Meaning |
|----------------|-------------|---------|
| `SUCCESS` + `__result__["status"]=="success"` | `SUCCESS` | Chart rendered, PNG + Figure captured |
| `SUCCESS` + `__result__["status"]=="no_figure"` | `NO_FIGURE` | Code ran but no axes created |
| `SUCCESS` + `return_value is None` | `RUNTIME_ERROR` | Wrapper failed to set `__result__` |
| `SYNTAX_ERROR` | `SYNTAX_ERROR` | Code has syntax error |
| `RUNTIME_ERROR` | `RUNTIME_ERROR` | Exception during execution |
| `TIMEOUT` | `TIMEOUT` | Exceeded `timeout_seconds` |
| `IMPORT_ERROR` | `IMPORT_ERROR` | Missing module |

### Key Design Decisions

1. **Reuse `sandbox.execute_code()` with code wrapping**: The spec suggested a separate `_render_worker.py` subprocess script. Instead, `_prepare_worker_code()` wraps user code with matplotlib instrumentation and passes the wrapped code to the existing sandbox. This reuses all of sandbox's infrastructure (temp-file IPC, timeout, error categorization, cleanup) without duplication. SoC is maintained by composition: sandbox handles process isolation, renderer handles matplotlib specifics.

2. **Factory pattern for `render_node`**: `_make_render_node(renderer)` follows the established `_make_should_continue(config)` closure pattern. The renderer instance is injected at graph construction time via `build_convergence_graph(bundle=...)`. OCP: passing a different bundle (future Plotly) requires no changes to `loop.py`.

3. **Sync `Renderer` protocol, async `render_node`**: The `Renderer.render()` method is synchronous — simpler interface, portable across contexts. The convergence loop's `render_node` wraps it in `asyncio.to_thread()` to avoid blocking the event loop. Async is the loop's concern, not the renderer's.

4. **No `bbox_inches='tight'`**: Using fixed figsize/dpi means PNG pixel dimensions = `figsize_inches * dpi`, making coordinate mapping to MVP.7's bounding boxes predictable. `bbox_inches='tight'` would alter dimensions and break the mapping.

5. **Defensive monkey-patching of `plt.show`/`plt.close`**: Even though LLM-generated code (from MVP.5) shouldn't call these, the wrapper neutralizes them to prevent subtle figure loss. Low cost, prevents a class of hard-to-debug failures.

6. **`extractor=None` in `matplotlib_bundle()`**: MVP.7 will provide the real `MatplotlibExtractor`. Acceptable because MVP.6 doesn't use the extractor field, and the factory signature won't change when MVP.7 adds it.

7. **`matplotlib.use('Agg')` in test module**: Unpickling a matplotlib Figure in the test process triggers backend initialization. Without forcing Agg, the default TkAgg backend tries to create a Tk window, which fails in headless/CI environments.

### Test Results

All 313 tests pass (20.75s):
- `test_renderer.py`: 22 tests (protocol, rendering, figure integrity, bbox thesis, errors, monkey-patching, DPI, bundle)
- `test_convergence_graph.py`: 13 tests (topology, stubs, stop conditions, explicit bundle)
- Previous MVP.1 + MVP.2 + MVP.3 + MVP.4 + MVP.5 tests: all still passing (278 tests)

### Core Thesis Validation

`test_bbox_thesis` proves the fundamental plotlint approach works: after rendering in a subprocess and unpickling the Figure, `fig.canvas.draw()` → `ax.get_xticklabels()[0].get_window_extent(renderer)` returns non-degenerate bounding boxes. This confirms that MVP.7's element extraction strategy (walking the artist tree for bboxes) is viable.

### Unblocks

- **MVP.7** (Inspector): Receives `RenderResult.figure_data` — pickled Figure with intact artist tree. Unpickle → `fig.canvas.draw()` → extract bboxes. `RendererBundle.extractor` field is ready for `MatplotlibExtractor`.
- **MVP.8** (Patcher): Render errors populate `ConvergenceState.render_error`, triggering stop condition. Patched code re-enters `render_node` on the next iteration.
- **PL-1.5** (Convergence GIF): `RenderResult.png_bytes` captured at each iteration. GIF generator reads from `ConvergenceState.png_bytes` history.
- **PL-1.6** (Plotly): Implement `Renderer` protocol with Playwright backend. `RendererBundle` and convergence loop are renderer-agnostic — no modifications needed.

**Packages:** plotlint

## 2026-02-08: MVP.5 Chart Planning + Code Generation

### Summary
Implemented the chart planning and code generation module (`autodash/charts.py`). Accepts `InsightResult`(s) and user questions, uses two LLM calls — one to produce a renderer-agnostic `ChartSpec` (JSON), one to generate self-contained matplotlib code — and returns `ChartPlan`(s) pairing spec with code. The two-step split is critical: DI-1.3 (multi-chart) needs to intervene between planning and generation to assign priorities and avoid duplicate charts. All data models and error types were already defined in MVP.1; no model/config changes needed.

### Architecture

```
               ┌──────────┐   ┌──────────┐
               │  MVP.4   │   │  user     │
               │ explorer │   │ questions │
               └────┬─────┘   └────┬─────┘
                    │ InsightResult │
                    └───────┬───────┘
                            ▼
                     ┌─────────────┐
                     │   MVP.5     │
                     │  charts.py  │
                     └──────┬──────┘
                            │ list[ChartPlan]
                            ▼
                     ┌─────────────┐
                     │   MVP.6     │
                     │ renderer.py │  executes the code
                     └─────────────┘

charts.py internals (SoC):

    _serialize_df_for_prompt()        ← DataFrame → Python dict literal (NaN→None, datetime→ISO)
    _to_python_native()               ← single value → JSON-safe native type
          │
    build_chart_planning_prompt()     ← assemble user prompt for spec generation
    build_code_generation_prompt()    ← assemble user prompt for code generation
          │
    plan_charts()                     ← Step 1: prompt → LLM → JSON → ChartSpec list (async)
          │
    ├── parse_chart_specs()           ← parse JSON, validate, build specs
    ├── _parse_single_spec()          ← validate fields, chart type enum, source_step_index
    └── _validate_data_mapping()      ← per-chart-type structural + column existence checks
          │
    generate_chart_code()             ← Step 2: spec + data → LLM → Python code → ChartPlan (async)
          │
    └── parse_code_from_response()    ← reused from plotlint.core.parsing
          │
    plan_and_generate()               ← combined entry point for pipeline node
```

### New / Modified Files

| File | Purpose |
|------|---------|
| `autodash/charts.py` | **New.** Core logic: `_serialize_df_for_prompt`, `_to_python_native`, `_validate_data_mapping`, `build_chart_planning_prompt`, `build_code_generation_prompt`, `parse_chart_specs`, `_parse_single_spec`, `plan_charts`, `generate_chart_code`, `plan_and_generate` |
| `autodash/prompts/chart_planning.py` | **New.** `SYSTEM_PROMPT` (chart type guidance, DataMapping rules per type), `OUTPUT_FORMAT` (JSON schema), `build_user_prompt()` — enum values built from `ChartType`/`ChartPriority` to stay in sync |
| `autodash/prompts/code_generation.py` | **New.** `SYSTEM_PROMPT` (matplotlib code rules: no savefig/show/close, no backend, tight_layout, self-contained), `build_user_prompt()` — `renderer_type` param enables PL-1.6 Plotly with new template |
| `autodash/pipeline.py` | **Modified.** Replaced `chart_node` stub with closure factory `_make_chart_node(config, llm_client)`. Fallback `chart_node` returns error dict when no LLM client. Wired into `build_pipeline_graph()` |
| `tests/test_charts.py` | **New.** 67 tests: native conversion (10), serialization (6), data mapping validation (20), spec parsing (13), prompt construction (7), async integration with MockLLMClient (11) |
| `tests/test_chart_prompts.py` | **New.** 22 tests: system prompt content (chart types, priorities, mapping rules), output format fields, user prompt assembly, code gen rules (no savefig, no backend, tight_layout) |
| `tests/test_pipeline_graph.py` | **Modified.** Updated `test_chart_stub` → `test_chart_fallback_returns_error` |

### Two-Step LLM Process

```
Step 1: plan_charts(insights, questions, llm_client, max_charts)
  → build_chart_planning_prompt()
  → LLM call (JSON response)
  → parse_chart_specs() with validation
  → list[ChartSpec]

Step 2: generate_chart_code(spec, insight, llm_client, renderer_type)
  → build_code_generation_prompt()  (embeds data as dict literal)
  → LLM call (Python code response)
  → parse_code_from_response()
  → ChartPlan(spec=spec, code=code)

Combined: plan_and_generate() calls Step 1 then Step 2 for each spec.
```

### DataMapping Validation Rules

`_validate_data_mapping(mapping, chart_type, available_columns)` returns error list:

| Chart Type | Required Fields | Optional |
|-----------|----------------|----------|
| BAR, LINE, AREA, SCATTER, HEATMAP | x, y | color, size, label |
| GROUPED_BAR, STACKED_BAR | x, y, color | label |
| PIE | values, categories | label |
| HISTOGRAM | x OR y | color |
| BOX | y | x (grouping) |

All non-None column references validated against `InsightResult.column_names`.

### Key Design Decisions

1. **Dict literal for data embedding**: Generated code uses `pd.DataFrame({...})` with inline dict. `_serialize_df_for_prompt()` handles NaN→None, datetime→ISO string, numpy→Python native. `PipelineConfig.inline_data_max_rows=50` as size guard. Cleaner than CSV string, no file path dependency (result_df doesn't exist as a file).

2. **Fully self-contained generated code**: No `inject_globals`, no variables expected in scope. Code is copy-pasteable and debuggable. Conventions: `fig, ax = plt.subplots(figsize=...)`, no `plt.savefig()`/`plt.show()`/`plt.close()` (MVP.6 renderer captures `gcf()`), no `matplotlib.use('Agg')` (renderer worker sets backend), `plt.tight_layout()` at end.

3. **Single-shot code generation, no retry**: Unlike explorer.py which retries on execution errors, chart code generation just produces a string — there's no execution feedback within MVP.5. The convergence loop (MVP.6-8: render→inspect→patch) handles execution failures. Adding retry here would duplicate that responsibility.

4. **`source_step_index` for insight→spec linking**: `ChartSpec.source_step_index` (int) references the insight that produced the data. `parse_chart_specs()` validates it's in range. `plan_and_generate()` uses it to pair each spec with the correct `InsightResult` for code generation.

5. **Per-module profile view**: `_serialize_df_for_prompt()` produces a dict literal + column info for code generation context. Different from planner's `_profile_summary()` (types + stats) and explorer's `_exploration_profile_summary()` (dtypes for code gen). Each LLM-calling module formats data for its specific needs.

6. **Enum values auto-synced in prompts**: `SYSTEM_PROMPT` in `chart_planning.py` builds chart type and priority lists from the enums (`", ".join(t.value for t in ChartType)`), matching the pattern in `analysis_planning.py`. Adding a new `ChartType` variant automatically appears in prompts.

7. **Sequential code generation**: `plan_and_generate()` generates code for each spec sequentially. For MVP (max_charts=1), moot. DI-1.3 can trivially switch to `asyncio.gather()` without changing function signatures.

### Test Results

All 291 tests pass (11.95s):
- `test_charts.py`: 67 tests (native conversion, serialization, validation, parsing, prompt construction, integration)
- `test_chart_prompts.py`: 22 tests (system prompt content, output format, user prompt assembly)
- Previous MVP.1 + MVP.2 + MVP.3 + MVP.4 tests: all still passing (202 tests)

### Unblocks

- **MVP.6** (Renderer): Receives `ChartPlan.code` — a self-contained matplotlib script. Executes in subprocess sandbox, captures `plt.gcf()`. No coupling to how the code was generated.
- **MVP.8** (Patcher): Replaces `ChartPlan.code` after patching. `ChartPlan` is mutable; `ChartSpec` is frozen (spec doesn't change, only implementation).
- **DI-1.3** (Multi-chart): `plan_charts()` already returns `list[ChartSpec]`. `ChartPriority` enum exists from day one. DI-1.3 calls with `max_charts=N` and intervenes between plan and generate steps.
- **PL-1.6** (Plotly): `generate_chart_code(renderer_type=RendererType.PLOTLY)` selects a different prompt template. No logic change in `charts.py`.
- **DI-2.2** (Style harmonizer): Optional `color_palette` field already on `ChartSpec`. Harmonizer populates it before code generation.

**Packages:** autodash

## 2026-02-08: MVP.4 Data Exploration

### Summary
Implemented the data exploration module (`autodash/explorer.py`). Accepts an `AnalysisStep` and a DataFrame, uses an LLM to generate pandas code, executes it in the subprocess sandbox with retry (up to 3 attempts), normalizes the result, and produces an `InsightResult` containing the computed DataFrame, a template-based summary, and the generated code. This is the second LLM-calling node in the pipeline and the first to use the subprocess sandbox for code execution. Also created the shared sandbox execution pattern that MVP.6 (renderer) will reuse.

### Architecture

```
                   ┌──────────┐   ┌──────────┐
                   │  MVP.2   │   │  MVP.3   │
                   │  data.py │   │ planner  │
                   └────┬─────┘   └────┬─────┘
                        │ DataFrame     │ AnalysisStep
                        └───────┬───────┘
                                ▼
                         ┌─────────────┐       ┌──────────────────┐
                         │   MVP.4     │──────▶│ plotlint/core/   │
                         │ explorer.py │       │  sandbox.py      │
                         └──────┬──────┘       └──────────────────┘
                                │                            ▲
                                │ InsightResult       reused by │
                                ▼                              │
                         ┌─────────────┐       ┌─────────────┐
                         │   MVP.5     │       │   MVP.6     │
                         │  charts.py  │       │ renderer.py │
                         └─────────────┘       └─────────────┘

explorer.py internals (SoC):

    _exploration_profile_summary()  ← format profile with pandas dtypes
          │
    _step_details()                 ← serialize AnalysisStep fields
          │
    build_exploration_prompt()      ← assemble full user prompt (+ error context on retry)
          │
    explore_step()                  ← orchestrate: prompt → LLM → parse → sandbox → normalize (async)
          │
    ├── parse_code_from_response()  ← reused from plotlint.core.parsing
    ├── execute_code()              ← reused from plotlint.core.sandbox
    ├── _normalize_result()         ← DataFrame/Series/scalar → DataFrame
    └── summarize_result()          ← template-based summary
```

### New / Modified Files

| File | Purpose |
|------|---------|
| `autodash/explorer.py` | **New.** Core exploration logic: `_exploration_profile_summary`, `_step_details`, `build_exploration_prompt`, `_normalize_result`, `summarize_result`, `explore_step` |
| `autodash/prompts/data_exploration.py` | **New.** `SYSTEM_PROMPT`, `ERROR_RETRY_BLOCK`, `build_user_prompt()` — prompt templates separated from logic |
| `autodash/pipeline.py` | **Modified.** Replaced `explore_node` stub with closure factory `_make_explore_node(config, llm_client)`. Fallback `explore_node` returns error dict when no LLM client. Re-loads DataFrame from `source_path` to keep state serializable |
| `tests/test_explorer.py` | **New.** 25 tests: profile summary (5), step details (3), prompt construction (5), result normalization (8), summarization (4), async integration with MockLLMClient (10 — success, retry, failure, edge cases) |
| `tests/test_explorer_prompts.py` | **New.** 8 tests: template content, placeholder coverage, assembly |
| `tests/test_pipeline_graph.py` | **Modified.** Updated `test_explore_stub` → `test_explore_fallback_returns_error` |

### Explore Step Retry Flow

`explore_step(step, df, profile, llm_client, max_attempts=3)`:

```
for attempt in 1..max_attempts:
  1. Build prompt (+ error context if retry)
  2. LLM generates pandas code
  3. Parse code from response (parse_code_from_response)
  4. Execute in sandbox (inject_globals={"df": df})
  5. Check result:
     - SUCCESS + __result__ set     → normalize → summarize → return InsightResult
     - SUCCESS + __result__ missing → retry with "assign to __result__" hint
     - RUNTIME_ERROR/SYNTAX_ERROR   → retry with error message + failed code
     - CODE_PARSE_FAILURE           → retry with raw response snippet
  6. LLM call failure → raise immediately (LLMClient has its own retries)

All attempts exhausted → raise ExplorationError
```

### Key Design Decisions

1. **Closure factory for explore node**: `_make_explore_node(config, llm_client)` follows the precedent set by `_make_plan_node` in MVP.3. Captures dependencies via closure for LangGraph's `(state) -> dict` node signature.

2. **DataFrame re-loaded from `source_path`**: The explore node calls `load_dataframe(Path(source_path))` rather than carrying the DataFrame in `PipelineState`. Keeps state serializable for LangGraph checkpointing (DI-4.2). Double I/O is negligible for MVP-scale data.

3. **Template-based summarization**: `summarize_result()` uses string formatting, not an LLM call. Cheaper, faster, deterministic, and sufficient for MVP since MVP.5 uses `InsightResult.to_prompt_context()` (shape + sample data) for chart planning context. DI-1.2 can wrap with LLM summarization without modifying `explore_step`.

4. **Result normalization accepts Series and scalars**: `_normalize_result()` converts `pd.Series` → `.to_frame()` and scalars → 1x1 DataFrame. LLMs frequently produce Series from `groupby().agg()` and scalar from `.sum()`. Unsupported types raise `ExplorationError`.

5. **`previous_code` in retry prompt**: Added beyond the spec's function signature. The spec's pitfall #5 explicitly requires the failed code in retry context so the LLM avoids repeating the same mistake. `ERROR_RETRY_BLOCK` template includes `{error_type}`, `{error_message}`, and `{previous_code}`.

6. **Per-module profile summary**: `_exploration_profile_summary()` includes pandas dtypes (not just semantic types) because the LLM needs correct dtype information for pandas code generation. Different emphasis from `_profile_summary()` in planner.py which focuses on semantic types and stats for analysis choice.

7. **LLM failure is not retried in the explore loop**: The `LLMClient` protocol implementations already have retry logic (`max_retries` with exponential backoff in `AnthropicClient`/`GeminiClient`). Retrying at the explore level would create double-retry. Only code execution failures trigger the retry loop.

8. **Sandbox remains pandas-agnostic**: The explorer passes the DataFrame via `inject_globals={"df": df}`. The sandbox pickles and unpickles it without knowing it's a DataFrame. No pandas-specific logic in `sandbox.py` (spec pitfall #1).

### Test Results

All 202 tests pass (9.34s):
- `test_explorer.py`: 25 tests (profile summary, step details, prompt construction, normalization, summarization, integration)
- `test_explorer_prompts.py`: 8 tests (template content, placeholders, assembly)
- Previous MVP.1 + MVP.2 + MVP.3 tests: all still passing (169 tests)

### Unblocks

- **MVP.5** (Chart Planning): Consumes `InsightResult`. `to_prompt_context()` provides shape, columns, summary, and sample data for chart type selection and code generation prompts.
- **MVP.6** (Renderer): Reuses `plotlint/core/sandbox.py` — the `execute_code()` function with `inject_globals` pattern is now proven end-to-end.
- **DI-1.2** (Agent loop exploration): `explore_step()` handles one step. DI-1.2 iterates over `list[AnalysisStep]` and calls it for each. Retry/review loop can be extended by wrapping `explore_step()`.
- **DI-1.2** (LLM summarization): Replace `summarize_result()` call without modifying `explore_step`.

**Packages:** autodash

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
