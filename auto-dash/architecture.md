# AutoDash / plotlint: Architecture

This document is the normative architectural reference for the AutoDash project. It defines package boundaries, dependency rules, data flow, cross-cutting concerns, protocols, state machines, and extension points for all phases (MVP through PL-3 and DI-4).

Refer to [development_plan.md](development_plan.md) for iteration structure, timelines, and success metrics. Refer to individual `mvp.X.md` specs for implementation-level detail per module.

---

## 1. Guiding Principles

> **"Declarative, Modular, SoC"**

| Principle | Meaning | Architectural Implication |
|-----------|---------|--------------------------|
| **Declarative** | Describe *what*, not *how*. Config over code. | Defect checks declare severity thresholds. Convergence behavior is driven by `ConvergenceConfig`, not if-else chains. Prompt templates are data, not logic. |
| **Modular** | Components are self-contained, swappable, independently testable. | plotlint works standalone or inside AutoDash. Swap matplotlib renderer for Plotly without touching collision detection. |
| **SoC** | Each module has ONE job. No god objects. Clear boundaries. | Inspector detects. Patcher fixes. Convergence loop orchestrates. Renderer executes. No component does two things. |

**Additional principles:** Measure Don't Guess, Risk-First Validation, Incremental Value, Portfolio-First Ordering. See [development_plan.md](development_plan.md) for details.

---

## 2. Package Boundary

This is the single most important architectural constraint.

### The Hard Rule

```
plotlint/       → ZERO imports from autodash. Self-contained. Pip-installable independently.
plotlint/core/  → Foundation utilities. Imported by plotlint modules AND autodash.
autodash/       → MAY import from plotlint (including plotlint.core). Never the reverse.
```

### Dependency Direction

```
         plotlint/core/
        (llm, sandbox, parsing,
         errors, config)
           ↑         ↑
           │         │
      plotlint/    autodash/
    (compliance   (pipeline,
     engine)       planning,
                   exploration)
```

autodash depends on plotlint. plotlint depends on nothing outside itself. This ensures `pip install plotlint` delivers a fully self-contained visual compliance tool.

### Optional Dependency Groups

plotlint supports tiered installation via optional dependencies in `pyproject.toml`:

| Install Command | What You Get |
|----------------|-------------|
| `pip install plotlint` | Inspector + geometry + checks. No LLM, no LangGraph. For CI gates, static analysis. |
| `pip install plotlint[llm]` | + Patcher + Critic. Requires `anthropic` SDK. For fix generation. |
| `pip install plotlint[full]` | + Convergence loop + CLI. Requires `langgraph`. Full plotlint experience. |
| `pip install autodash` | Everything above + pipeline + data intelligence. Requires `pandas`, `matplotlib`, etc. |

---

## 3. Package Structure

### plotlint (visual compliance engine)

```
plotlint/
    __init__.py
    core/                      # Foundation utilities — standalone-safe
        __init__.py
        llm.py                 # LLMClient protocol + AnthropicClient implementation
        sandbox.py             # execute_code(), ExecutionResult, ExecutionStatus
        parsing.py             # parse_code_from_response(), parse_json_from_response()
        errors.py              # Exception hierarchy: AutoDashError → PlotlintError, etc.
        config.py              # LLMConfig, SandboxConfig

    models.py                  # ConvergenceState, Issue, InspectionResult, RenderResult,
                               #   FixAttempt, PatchResult, Severity, DefectType, RenderStatus
    config.py                  # ConvergenceConfig
    geometry.py                # BoundingBox dataclass + geometric operations
    elements.py                # ElementMap, ElementInfo, ElementCategory, Extractor protocol
    inspector.py               # Inspector orchestrator (runs registered checks)
    renderer.py                # Renderer protocol + MatplotlibRenderer + RendererBundle
    _render_worker.py          # Subprocess worker script for matplotlib Agg
    patcher.py                 # LLM fix generation (select issue, prompt, validate)
    scoring.py                 # Issues → numeric score (0.0–1.0)
    loop.py                    # LangGraph convergence loop graph

    checks/                    # Defect check implementations (one per file)
        __init__.py            # Check protocol + registry + auto-import
        overlap.py             # label_overlap (MVP)
        cutoff.py              # element_cutoff (MVP)
        # PL-1.1: legend.py, empty.py
        # PL-2.1: readability.py, formatting.py
        # PL-2.2: color.py

    extractors/                # Bounding box extraction implementations
        __init__.py            # Extractor registry
        matplotlib.py          # MatplotlibExtractor (walks artist tree)
        # PL-1.6: plotly.py

    prompts/                   # LLM prompt templates (separate from logic)
        __init__.py
        patching.py            # Patcher prompts, keyed by renderer type

    # PL-1: cli.py, critic.py, viz.py
    # PL-2: color_utils.py, gallery.py
    # PL-3: jupyter.py

    dashboard/                 # Dashboard-level compliance (DI-2 + DI-3)
        __init__.py
        # DI-2.1: layout.py
        # DI-2.2: style.py
        # DI-3.1: renderer.py, inspector.py
        # DI-3.2: loop.py
        # DI-3.1: checks/
```

### autodash (full pipeline)

```
autodash/
    __init__.py
    models.py                  # DataProfile, ColumnProfile, SemanticType, AggregationType,
                               #   AnalysisStep, InsightResult, ChartType, ChartPriority,
                               #   RendererType, DataMapping, ChartSpec, ChartPlan,
                               #   PipelineState, OutputFormat, OutputArtifact, OutputResult
    config.py                  # PipelineConfig (wraps plotlint configs)
    data.py                    # DataLoader protocol + CsvLoader, ExcelLoader, ParquetLoader
    planner.py                 # Analysis planning (LLM → AnalysisStep)
    explorer.py                # Data exploration (LLM code gen + sandbox → InsightResult)
    charts.py                  # Chart planning + code generation (LLM → ChartPlan)
    pipeline.py                # LangGraph pipeline graph
    output.py                  # OutputWriter protocol + PNGWriter

    prompts/                   # LLM prompt templates
        analysis_planning.py
        data_exploration.py
        chart_planning.py
        code_generation.py

    # DI-1.5: cli.py
    # Note: layout lives in plotlint/dashboard/layout.py, not here
```

---

## 4. Cross-Cutting Concerns

### 4.1 LLM Client (`plotlint/core/llm.py`)

All LLM-calling modules depend on a shared `LLMClient` protocol via dependency injection. No module directly instantiates the Anthropic SDK.

```python
# plotlint/core/llm.py

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM API calls.

    All modules depend on this, not on anthropic directly.
    Enables testing with MockLLMClient.
    """

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Return the text content of the LLM response."""
        ...

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Return the text content of the LLM response (vision)."""
        ...


class AnthropicClient:
    """Default LLMClient implementation using the Anthropic API.

    Provides retry logic, rate limiting, and cost tracking.
    Configured via LLMConfig.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # import anthropic lazily (optional dependency)
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)

    async def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        response = await self._client.messages.create(
            model=model or self.config.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image_bytes: bytes,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        import base64
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = await self._client.messages.create(
            model=model or self.config.vision_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": user},
            ]}],
        )
        return response.content[0].text
```

**Usage pattern** (same for planner, explorer, charts, patcher):

```python
async def plan_analysis(
    profile: DataProfile,
    questions: str,
    llm_client: LLMClient,          # injected — never created internally
    max_steps: int = 1,
) -> list[AnalysisStep]:
    prompt = build_planning_prompt(profile, questions, max_steps)
    raw = await llm_client.complete(system=SYSTEM_PROMPT, user=prompt)
    return parse_analysis_response(raw, profile)
```

### 4.2 Code Execution Sandbox (`plotlint/core/sandbox.py`)

Generic subprocess execution. Used by both `plotlint/renderer.py` (matplotlib scripts) and `autodash/explorer.py` (pandas code).

```python
# plotlint/core/sandbox.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing Python code in a subprocess."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    return_value: Any = None
    execution_time_ms: int = 0


def execute_code(
    code: str,
    timeout_seconds: int = 30,
    allowed_imports: Optional[set[str]] = None,
    inject_globals: Optional[dict[str, Any]] = None,
) -> ExecutionResult:
    """Execute Python code in a subprocess sandbox.

    The subprocess receives code via temp file, executes with timeout,
    returns stdout/stderr and an optional pickled return value via temp file.

    Args:
        code: Python source code to execute.
        timeout_seconds: Kill subprocess after this many seconds.
        allowed_imports: Restrict imports (future hardening).
        inject_globals: Variables to inject into execution namespace.
                        For explorer: {"df": the_dataframe}
                        For renderer: {} (matplotlib scripts are self-contained)
    """
    ...
```

### 4.3 Response Parsing (`plotlint/core/parsing.py`)

Shared utility for extracting code and JSON from LLM responses. Used by patcher, explorer, and charts modules.

```python
# plotlint/core/parsing.py

from __future__ import annotations

from typing import Any


def parse_code_from_response(raw_response: str) -> str:
    """Extract a Python code block from an LLM response.

    Handles:
    - Markdown fenced code blocks (```python ... ```)
    - Plain code responses (no fences)
    - Multiple code blocks (takes the last one)

    Raises ValueError if no code block is found.
    """
    ...


def parse_json_from_response(raw_response: str) -> Any:
    """Extract and parse JSON from an LLM response.

    Handles:
    - Markdown fenced blocks (```json ... ```)
    - Plain JSON responses
    - JSON embedded in prose (extracts first valid JSON object/array)

    Raises ValueError if no valid JSON is found.
    """
    ...
```

### 4.4 Error Hierarchy (`plotlint/core/errors.py`)

Consistent exception types across the project. The pipeline can catch broad categories; tests can assert specific types; the CLI can format user-friendly messages.

```python
# plotlint/core/errors.py

class AutoDashError(Exception):
    """Base exception for the entire AutoDash/plotlint project."""
    pass


# --- Shared errors ---

class ConfigError(AutoDashError):
    """Configuration errors: missing API key, invalid config values."""
    pass


class LLMError(AutoDashError):
    """LLM API call failures: network errors, rate limits, malformed responses."""
    pass


class SandboxError(AutoDashError):
    """Code execution failures in the subprocess sandbox."""
    pass


# --- plotlint-specific ---

class PlotlintError(AutoDashError):
    """Base for plotlint-specific errors."""
    pass


class RenderError(PlotlintError):
    """Rendering failures: syntax errors in chart code, timeout, no figure produced."""
    pass


class ExtractionError(PlotlintError):
    """Element extraction failures: unpickling failed, artist tree inaccessible."""
    pass


class PatchError(PlotlintError):
    """Patching failures: LLM returned invalid code, all retries exhausted."""
    pass


# --- autodash-specific ---

class PipelineError(AutoDashError):
    """Base for AutoDash pipeline errors."""
    pass


class DataError(PipelineError):
    """Data loading or profiling failures: unsupported format, corrupt file."""
    pass


class ExplorationError(PipelineError):
    """Data exploration failures: all code execution attempts exhausted."""
    pass


class ChartGenerationError(PipelineError):
    """Chart code generation failures: invalid spec, code gen failed."""
    pass
```

### 4.5 Configuration (`plotlint/core/config.py`, `plotlint/config.py`, `autodash/config.py`)

Declarative configuration hierarchy. All behavior-controlling parameters live in config dataclasses, not as magic numbers in code.

```python
# plotlint/core/config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM API calls."""
    api_key: str = ""                              # from env var ANTHROPIC_API_KEY
    default_model: str = "claude-sonnet-4-5-20250929"  # configurable; use latest at impl time
    vision_model: str = "claude-sonnet-4-5-20250929"   # configurable; use latest at impl time
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for subprocess code execution."""
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    allowed_imports: Optional[frozenset[str]] = None
```

```python
# plotlint/config.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configurable parameters for the convergence loop.

    Declarative: behavior is driven by these values, not if-else chains.
    """
    max_iterations: int = 3
    target_score: float = 1.0
    stagnation_window: int = 2
    score_improvement_threshold: float = 0.01
```

```python
# autodash/config.py

from __future__ import annotations

from dataclasses import dataclass, field

from plotlint.core.config import LLMConfig, SandboxConfig
from plotlint.config import ConvergenceConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the AutoDash pipeline.

    Composes all sub-configs. Each module receives the relevant slice.
    """
    output_dir: str = "output"
    output_format: str = "png"
    max_charts: int = 1                            # MVP=1, DI-1.3=N
    max_analysis_steps: int = 1                    # MVP=1, DI-1.1=N
    max_exploration_attempts: int = 3
    inline_data_max_rows: int = 50                 # embed data in code up to this size

    # Sub-configs
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
```

---

## 5. Data Flow Architecture

### 5.1 plotlint Standalone Flow

```
Input: source_code (str)
  │
  ▼
[Renderer]               Execute code in subprocess, force Agg backend
  │                       Produce RenderResult (png_bytes, figure_data)
  ▼
[Extractor]              Walk figure internals (artist tree / DOM)
  │                       Produce ElementMap (list[ElementInfo] + figure_bbox)
  ▼
[Inspector]              Run all registered Checks against ElementMap
  │                       Produce InspectionResult (list[Issue], score)
  ▼
[should_continue?]       Evaluate ConvergenceConfig stop conditions
  │           │
  │ stop      │ continue
  ▼           ▼
[END]       [Patcher]    LLM generates surgical fix for top issue
              │           Produce PatchResult (patched_code, code_hash)
              ▼
            [back to Renderer with updated source_code]
```

### 5.2 AutoDash Pipeline Flow

```
Input: (source_path: str, questions: str)
  │
  ▼
[load_node]              Load file, profile columns → DataProfile
  │                       DataFrame held in memory (not in serializable state)
  ▼
[plan_node]              LLM: DataProfile + questions → list[AnalysisStep]
  ▼
[explore_node]           LLM generates pandas code → sandbox executes
  │                       Produce list[InsightResult] (DataFrame + summary)
  ▼
[chart_node]             LLM: insights → list[ChartPlan] (ChartSpec + code)
  ▼
[comply_node]            Invoke plotlint convergence loop per chart
  │                       Produce list[ChartPlan] (polished)
  ▼
[output_node]            Write PNG + code to disk → OutputResult
```

### 5.3 Data Ownership Table

| Data Type | Produced By | Consumed By | Serializable? | Notes |
|-----------|------------|-------------|---------------|-------|
| `DataProfile` | MVP.2 `data.py` | MVP.3 `planner.py`, MVP.1 `PipelineState` | Yes (JSON) | Frozen dataclass. `to_json()` / `from_json()`. |
| `pd.DataFrame` | MVP.2 `data.py` | MVP.4 `explorer.py` | No | Held in memory. Reloaded from `source_path` in explore_node for checkpointing compatibility. |
| `AnalysisStep` | MVP.3 `planner.py` | MVP.4 `explorer.py` | Yes (JSON) | Frozen dataclass. |
| `InsightResult` | MVP.4 `explorer.py` | MVP.5 `charts.py` | Partial | `result_df` not serializable. `to_prompt_context()` for LLM use. Checkpointing deferred to DI-4.2. |
| `ChartSpec` | MVP.5 `charts.py` | MVP.5 code gen, layout | Yes (JSON) | Frozen, renderer-agnostic. |
| `ChartPlan` | MVP.5 `charts.py` | MVP.6 renderer, MVP.8 patcher | Partial | `code` is a string (serializable). `spec` is frozen (serializable). Mutable because patcher replaces `code`. |
| `RenderResult` | MVP.6 `renderer.py` | MVP.7 inspector, loop | No | Contains bytes (png, pickled figure). In-memory only. |
| `ElementMap` | MVP.7 `extractors/` | MVP.7 `checks/`, `inspector.py` | No | In-memory. Intermediate representation between extraction and checking. |
| `InspectionResult` | MVP.7 `inspector.py` | MVP.8 `patcher.py`, loop | Yes (JSON) | Frozen dataclass. |
| `PatchResult` | MVP.8 `patcher.py` | Loop (`loop.py`) | Yes (JSON) | Frozen dataclass. |
| `OutputResult` | MVP.9 `output.py` | Pipeline, CLI | Yes | Tracks written files. |

---

## 6. Protocol & Registry Summary

### 6.1 Protocol Table

| Protocol | Location | Implementations | Registration | Used By |
|----------|---------|-----------------|-------------|---------|
| `LLMClient` | `plotlint/core/llm.py` | `AnthropicClient`, `MockLLMClient` (tests) | Dependency injection | planner, explorer, charts, patcher, critic |
| `DataLoader` | `autodash/data.py` | `CsvLoader`, `ExcelLoader`, `ParquetLoader` | `register_loader()` | `data.py` loader resolution |
| `Renderer` | `plotlint/renderer.py` | `MatplotlibRenderer`, (PL-1.6: `PlotlyRenderer`) | Via `RendererBundle` | `loop.py` render_node |
| `Extractor` | `plotlint/elements.py` | `MatplotlibExtractor`, (PL-1.6: `PlotlyExtractor`) | Paired in `RendererBundle` | `inspector.py` |
| `Check` | `plotlint/checks/__init__.py` | `LabelOverlapCheck`, `ElementCutoffCheck`, ... | `@check()` decorator | `inspector.py` via registry |
| `OutputWriter` | `autodash/output.py` | `PNGWriter`, (DI-2.3: `HTMLWriter`, DI-4.1: `PDFWriter`, DI-4.3: `JSONWriter`) | `register_writer()` | `output.py` |

### 6.2 Check Protocol

```python
# plotlint/checks/__init__.py

@runtime_checkable
class Check(Protocol):
    """A single visual defect check. Pure function of the ElementMap."""

    @property
    def name(self) -> str:
        """Unique name matching a DefectType value."""
        ...

    def __call__(self, elements: ElementMap) -> list[Issue]:
        """Run the check. Return detected issues (empty if none)."""
        ...


# Registry
_CHECKS: dict[str, Check] = {}

def check(name: str):
    """Decorator for registering a check class."""
    def decorator(cls):
        _CHECKS[name] = cls()
        return cls
    return decorator

def get_registered_checks() -> dict[str, Check]:
    return dict(_CHECKS)


# Auto-import check modules so @check decorators execute
from plotlint.checks import overlap   # noqa: F401
from plotlint.checks import cutoff    # noqa: F401
# PL-1.1: from plotlint.checks import legend, empty
# PL-2.1: from plotlint.checks import readability, formatting
# PL-2.2: from plotlint.checks import color
```

### 6.3 Renderer + Extractor Pairing

Renderers and extractors are always paired: `MatplotlibRenderer` must use `MatplotlibExtractor`. A `RendererBundle` prevents mismatches.

```python
# plotlint/renderer.py

@dataclass
class RendererBundle:
    """Pairs a Renderer with its matching Extractor.

    The convergence loop uses the bundle. When PL-1.6 adds Plotly,
    it registers a new bundle. Prevents renderer/extractor mismatches.
    """
    renderer: Renderer
    extractor: Extractor
    renderer_type: str


def matplotlib_bundle(
    dpi: int = 100,
    timeout_seconds: int = 30,
) -> RendererBundle:
    """Create a matplotlib renderer+extractor bundle."""
    from plotlint.extractors.matplotlib import MatplotlibExtractor
    return RendererBundle(
        renderer=MatplotlibRenderer(dpi=dpi, timeout_seconds=timeout_seconds),
        extractor=MatplotlibExtractor(),
        renderer_type="matplotlib",
    )
```

### 6.4 DataLoader Protocol

```python
# autodash/data.py

@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading tabular data from a file."""

    def supports(self, path: Path) -> bool: ...
    def load(self, path: Path) -> pd.DataFrame: ...

    @property
    def format_name(self) -> str: ...


_LOADERS: list[DataLoader] = [CsvLoader(), ExcelLoader(), ParquetLoader()]

def register_loader(loader: DataLoader) -> None:
    _LOADERS.append(loader)
```

### 6.5 OutputWriter Protocol

```python
# autodash/output.py

@runtime_checkable
class OutputWriter(Protocol):
    """Protocol for output writers."""

    @property
    def format(self) -> OutputFormat: ...

    def write(
        self,
        png_bytes: bytes,
        code: str,
        output_dir: Path,
        name: str = "chart",
    ) -> OutputResult: ...


_WRITERS: dict[OutputFormat, OutputWriter] = {}

def register_writer(writer: OutputWriter) -> None:
    _WRITERS[writer.format] = writer

def get_writer(format: OutputFormat = OutputFormat.PNG) -> OutputWriter:
    if format not in _WRITERS:
        raise ConfigError(f"No writer for format: {format}")
    return _WRITERS[format]

# Register defaults
register_writer(PNGWriter())
```

---

## 7. State Machine Specifications

### 7.1 ConvergenceState (plotlint)

```python
# plotlint/models.py

class ConvergenceState(TypedDict, total=False):
    """LangGraph state for the plotlint convergence loop.

    total=False: all fields optional, enabling incremental building.
    Nodes read/write only their fields. Future phases add fields additively.
    """
    # Core (set at initialization)
    source_code: str                           # current chart code
    original_code: str                         # immutable copy of input

    # Renderer output (set by render_node)
    png_bytes: bytes
    render_error: Optional[str]
    figure_pickle: Optional[bytes]             # pickled Figure (matplotlib) or HTML (Plotly)

    # Inspector output (set by inspect_node)
    inspection: Optional[InspectionResult]
    score: float                               # 0.0–1.0

    # Convergence tracking
    iteration: int
    max_iterations: int
    score_history: list[float]
    fix_history: list[FixAttempt]

    # Best-seen tracking (PL-1.2 rollback)
    best_code: str
    best_score: float

    # Patcher output (set by patch_node)
    patch_applied: bool

    # Critic output (PL-1.3)
    critic_feedback: Optional[str]
    critic_invoked: bool

    # Renderer bundle reference
    renderer_type: str                         # "matplotlib" or "plotly"

    # Spec context (PL-1.3 — for Critic spec conformance checking)
    # When invoked from AutoDash pipeline, contains serialized ChartSpec as text.
    # When invoked standalone (plotlint script.py), None.
    spec_context: Optional[str]
```

### 7.2 PipelineState (autodash)

```python
# autodash/models.py

class PipelineState(TypedDict, total=False):
    """LangGraph state for the AutoDash pipeline.

    Each node reads/writes only its relevant fields.
    DI phases add fields additively.

    Note: InsightResult.result_df is not JSON-serializable.
    LangGraph checkpointing for HITL (DI-4.2) requires a serialization
    layer. For MVP, run without checkpointer (in-memory only).
    """
    # Inputs
    source_path: str
    questions: str

    # MVP.2: Data Intelligence
    data_profile: Optional[DataProfile]

    # MVP.3: Analysis Planning
    analysis_steps: list[AnalysisStep]

    # MVP.4: Data Exploration
    insights: list[InsightResult]

    # MVP.5: Chart Planning
    chart_plans: list[ChartPlan]

    # MVP.6-8: Convergence (per chart)
    polished_charts: list[ChartPlan]

    # MVP.9: Output
    output_path: Optional[str]

    # Error tracking
    errors: list[str]

    # DI-1.4: layout_spec: Optional[LayoutSpec]
    # DI-3: dashboard_inspection: Optional[InspectionResult]
```

### 7.3 Convergence Loop Graph

```python
# plotlint/loop.py

async def render_node(state: ConvergenceState) -> dict:
    """Execute chart code and capture figure + PNG.
    Increments iteration counter before rendering.
    """
    ...

async def inspect_node(state: ConvergenceState) -> dict:
    """Run Extractor + Inspector on rendered figure."""
    ...

async def patch_node(state: ConvergenceState) -> dict:
    """Generate a fix for the highest-severity issue."""
    ...

def should_continue(state: ConvergenceState) -> str:
    """Conditional edge: 'patch' or 'stop'.

    Stop conditions (checked in order):
    1. score >= target_score
    2. iteration >= max_iterations
    3. render_error is set
    4. score stagnant for stagnation_window iterations
    """
    ...

def build_convergence_graph(
    config: ConvergenceConfig = ConvergenceConfig(),
    bundle: Optional[RendererBundle] = None,
    llm_client: Optional[LLMClient] = None,
) -> CompiledStateGraph:
    """Build the convergence loop.

    Graph: render → inspect → decide → (patch → render | END)

    Args:
        config: Controls stop conditions.
        bundle: Renderer+Extractor pair. Default: matplotlib.
        llm_client: For the patcher node. Required for fix mode.
    """
    graph = StateGraph(ConvergenceState)
    graph.add_node("render", render_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("patch", patch_node)
    graph.set_entry_point("render")
    graph.add_edge("render", "inspect")
    graph.add_conditional_edges("inspect", should_continue, {"patch": "patch", "stop": END})
    graph.add_edge("patch", "render")
    return graph.compile()
```

**All node functions are `async def`** and **return partial dicts** (only the keys they modify). LangGraph merges partial updates into state automatically.

### 7.4 Pipeline Graph

```python
# autodash/pipeline.py

def build_pipeline_graph(
    config: PipelineConfig = PipelineConfig(),
    llm_client: Optional[LLMClient] = None,
    enable_hitl: bool = False,           # DI-4.2
    extra_nodes: Optional[dict] = None,  # DI phase extensions
) -> CompiledStateGraph:
    """Build the AutoDash pipeline.

    MVP: load → plan → explore → chart → comply → output → END
    DI:  load → plan → explore → chart → comply → layout → dashboard_comply → output → END
         + optional interrupt() at plan, explore, chart (DI-4.2)

    Args:
        config: Pipeline configuration.
        llm_client: Shared LLM client for all LLM-calling nodes.
        enable_hitl: Insert interrupt() at checkpoint nodes.
        extra_nodes: Additional nodes to insert (e.g., layout, dashboard_comply).
    """
    ...
```

---

## 8. Module Dependency Matrix

What each module is allowed to import. Violations are flagged during code review.

| Module | plotlint.core | plotlint (non-core) | autodash | External |
|--------|:---:|:---:|:---:|:---:|
| **plotlint/core/llm.py** | — | — | NEVER | anthropic (lazy) |
| **plotlint/core/sandbox.py** | — | — | NEVER | subprocess, pickle |
| **plotlint/core/parsing.py** | — | — | NEVER | json, re |
| **plotlint/core/errors.py** | — | — | NEVER | — |
| **plotlint/core/config.py** | — | — | NEVER | — |
| **plotlint/geometry.py** | — | — | NEVER | — |
| **plotlint/elements.py** | — | geometry | NEVER | — |
| **plotlint/models.py** | — | geometry, elements | NEVER | — |
| **plotlint/scoring.py** | — | models | NEVER | — |
| **plotlint/inspector.py** | — | elements, models, scoring, checks | NEVER | — |
| **plotlint/renderer.py** | sandbox | models, elements | NEVER | matplotlib |
| **plotlint/patcher.py** | llm, parsing | models | NEVER | ast, hashlib |
| **plotlint/loop.py** | — | models, config, renderer, inspector, patcher | NEVER | langgraph |
| **plotlint/checks/*.py** | — | elements, models, geometry | NEVER | — |
| **plotlint/extractors/*.py** | — | elements, geometry | NEVER | matplotlib / playwright |
| **autodash/models.py** | — | — | — | pandas (TYPE_CHECKING) |
| **autodash/config.py** | config | config (convergence) | — | — |
| **autodash/data.py** | errors | — | models | pandas |
| **autodash/planner.py** | llm, parsing, errors | — | models, prompts | — |
| **autodash/explorer.py** | llm, sandbox, parsing, errors | — | models, prompts | pandas |
| **autodash/charts.py** | llm, parsing, errors | — | models, prompts | — |
| **autodash/pipeline.py** | — | loop | models, config, data, planner, explorer, charts, output | langgraph |
| **autodash/output.py** | errors | — | models | — |

---

## 9. Configuration Hierarchy

```
PipelineConfig (autodash/config.py)
  ├── llm: LLMConfig (plotlint/core/config.py)
  │     ├── api_key: str
  │     ├── default_model: str
  │     ├── vision_model: str
  │     ├── max_retries: int
  │     ├── temperature: float
  │     └── max_tokens: int
  ├── sandbox: SandboxConfig (plotlint/core/config.py)
  │     ├── timeout_seconds: int
  │     └── allowed_imports: Optional[frozenset[str]]
  ├── convergence: ConvergenceConfig (plotlint/config.py)
  │     ├── max_iterations: int
  │     ├── target_score: float
  │     ├── stagnation_window: int
  │     └── score_improvement_threshold: float
  ├── output_dir: str
  ├── output_format: str
  ├── max_charts: int
  ├── max_analysis_steps: int
  ├── max_exploration_attempts: int
  └── inline_data_max_rows: int
```

Config loading pattern: environment variables → config file (TOML/YAML, optional) → defaults.

```python
# Example: constructing config from environment
import os

config = PipelineConfig(
    llm=LLMConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    ),
    convergence=ConvergenceConfig(max_iterations=3),
)
```

---

## 10. Testing Architecture

### 10.1 Three Tiers

| Tier | What | Dependencies | Speed | When |
|------|------|-------------|-------|------|
| **Unit** | Pure functions: geometry, scoring, parsing, checks, models | None (no LLM, no subprocess) | Fast (<1s) | Every commit |
| **Integration** | Full modules: renderer subprocess, extractor, inspector chain | matplotlib (real renders), MockLLMClient | Medium (5-30s) | Every PR |
| **System** | End-to-end: pipeline, convergence loop with real LLM | Anthropic API, matplotlib | Slow (30-120s) | Nightly / manual |

### 10.2 MockLLMClient for Testing

```python
class MockLLMClient:
    """Test double for LLMClient.

    Returns canned responses keyed by system prompt substring.
    Tracks call history for assertions.
    """
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[dict] = []

    async def complete(self, system, user, **kwargs) -> str:
        self.calls.append({"system": system, "user": user, **kwargs})
        for key, response in self.responses.items():
            if key in system or key in user:
                return response
        return self.responses.get("default", "")

    async def complete_with_image(self, system, user, image_bytes, **kwargs) -> str:
        self.calls.append({"system": system, "user": user, "has_image": True, **kwargs})
        return self.responses.get("vision", "")
```

### 10.3 Thesis Validation Test (Critical)

The single most important test in the project. If this fails, the plotlint approach needs rethinking.

```python
def test_bbox_extraction_from_rendered_chart():
    """Validate that matplotlib bounding boxes are accessible after render.

    This test validates the CORE THESIS: get_window_extent() returns
    non-zero, meaningful bounding boxes for chart elements rendered
    with the Agg backend.
    """
    renderer = MatplotlibRenderer(dpi=100)
    result = renderer.render(OVERLAPPING_LABELS_FIXTURE)
    assert result.succeeded

    extractor = MatplotlibExtractor()
    elements = extractor.extract(result.figure_data)

    # Verify we got tick labels
    x_labels = elements.tick_labels(axis="x")
    assert len(x_labels) >= 2

    # Verify bounding boxes are non-zero
    for label in x_labels:
        assert label.bbox.area > 0, f"Zero-area bbox for {label.element_id}"

    # Verify overlapping labels ARE detected as overlapping
    overlaps = sum(
        1 for i in range(len(x_labels) - 1)
        if x_labels[i].bbox.overlaps(x_labels[i + 1].bbox)
    )
    assert overlaps > 0, "Expected overlapping labels but none detected"
```

### 10.4 Fixture Strategy

```
tests/
    fixtures/
        simple_bar.py          # Clean bar chart, no issues
        overlapping_labels.py  # 12-month labels, known overlap
        cutoff_title.py        # Title extends beyond figure
        empty_chart.py         # No data elements
        clean_chart.py         # All checks pass, score = 1.0
    test_data/
        sample.csv             # 5-10 rows, mixed column types
        sample.xlsx
        sample.parquet
```

---

## 11. Extension Points — Post-MVP Phase Mapping

### 11.1 PL Extension Points (plotlint enhancements)

| Phase | What's Added | Architectural Hook | New/Modified Files |
|-------|-------------|-------------------|-------------------|
| **PL-1.1** | 3 new checks: `legend_occlusion`, `empty_plot`, `unnecessary_legend` | `@check()` decorator + import in `checks/__init__.py` | New: `checks/legend.py`, `checks/empty.py`. Mod: `checks/__init__.py`, `models.py` (DefectType) |
| **PL-1.2** | Rollback, fix dedup, score-drop detection | Extend `ConvergenceState` (fields already reserved). New edges in `build_convergence_graph()`. | Mod: `loop.py` |
| **PL-1.3** | Critic (LLM vision) | New graph node after Inspector when score==1.0. Uses `LLMClient.complete_with_image()`. Optionally accepts `ChartSpec` for spec conformance checking (e.g., verifies grouped bar vs stacked). | New: `critic.py`. Mod: `loop.py` |
| **PL-1.4** | plotlint CLI | New entry point: `plotlint script.py` | New: `cli.py` |
| **PL-1.5** | Convergence GIF | Read `png_bytes` per iteration from state history | New: `viz.py` |
| **PL-1.6** | Plotly renderer + inspector | New `Renderer` + `Extractor` as `RendererBundle`. All checks reused. | New: `renderers/plotly.py`, `extractors/plotly.py`. Mod: `prompts/patching.py` |
| **PL-1.7** | Smoke test suite | Test fixtures + pytest | New: `tests/smoke/`, `tests/fixtures/` |
| **PL-2.1** | Readability + y_format checks | `@check()` decorator | New: `checks/readability.py`, `checks/formatting.py` |
| **PL-2.2** | Color contrast + colorblind safety | `@check()` + color utilities | New: `checks/color.py`, `color_utils.py` |
| **PL-2.3-5** | Benchmarks, gallery, report | Independent tooling | New: `benchmarks/` |
| **PL-3.1** | Docker code sandbox | `execute_code()` uses `docker exec` instead of `subprocess.run`. `Renderer` protocol unchanged. | Mod: `plotlint/core/sandbox.py`, `plotlint/renderer.py` |
| **PL-3.2** | CI gate | CLI extension | Mod: `cli.py` |
| **PL-3.3** | Jupyter cell magic | IPython integration | New: `jupyter.py` |

**Key pattern**: The Check registry grows from 2 (MVP) to 9 (PL-2) checks. `inspector.py` never changes. Each new check = one file + one import line in `checks/__init__.py`.

### 11.2 DI Extension Points (dashboard intelligence)

| Phase | What's Added | Architectural Hook | New/Modified Files |
|-------|-------------|-------------------|-------------------|
| **DI-1.1** | Multi-step analysis | `plan_analysis(max_steps=N)` — already parameterized | Callsite change only |
| **DI-1.2** | Agent-loop exploration | Outer loop calling `explore_step()` per step | Mod: `autodash/explorer.py` |
| **DI-1.3** | Multi-chart planning | `plan_charts(max_charts=N)` — already parameterized | Callsite change only |
| **DI-1.4** | Grid layout | New module, new pipeline node | New: `plotlint/dashboard/layout.py`. Mod: `pipeline.py` |
| **DI-1.5** | Dashboard output + CLI | New OutputWriter + CLI entry | New: `autodash/cli.py`. Mod: `output.py` |
| **DI-2.1** | Full layout engine | Enhances DI-1.4 simple grid in-place | Mod: `plotlint/dashboard/layout.py` |
| **DI-2.2** | Style harmonizer | Cross-chart normalization | New: `plotlint/dashboard/style.py` |
| **DI-2.3** | HTML output | New OutputWriter | Mod: `autodash/output.py` |
| **DI-3.1** | Dashboard inspector | Check protocol applied to dashboard ElementMap | New: `plotlint/dashboard/checks/`, `inspector.py`, `renderer.py` |
| **DI-3.2** | Dashboard convergence loop | Separate LangGraph graph, same pattern | New: `plotlint/dashboard/loop.py` |
| **DI-3.3** | Dashboard benchmarks | Test infrastructure | New: `benchmarks/dashboard/` |
| **DI-4.1** | PDF output | New OutputWriter | Mod: `autodash/output.py` |
| **DI-4.2** | HITL checkpoints | `interrupt()` at pipeline nodes. State persistence. | Mod: `autodash/pipeline.py` |
| **DI-4.3** | JSON output | New OutputWriter (`JSONWriter`) | Mod: `autodash/output.py` |

### 11.3 plotlint/dashboard/ Sub-Package

Dashboard compliance lives in plotlint (not autodash) because it reuses the Check protocol, geometric primitives, and scoring system:

```
plotlint/dashboard/
    __init__.py
    layout.py              # LayoutSpec, layout engine (DI-2.1)
    style.py               # StyleSpec, harmonizer (DI-2.2)
    renderer.py            # Dashboard rendering (DI-3.1)
    inspector.py           # Dashboard inspection (DI-3.1)
    loop.py                # Dashboard convergence loop (DI-3.2)
    checks/
        __init__.py
        spacing.py         # Inconsistent inter-chart gaps
        alignment.py       # Charts misaligned from grid
        style_drift.py     # Inconsistent fonts/styles across charts
        color_collision.py # Same color, different meaning across charts
```

**Shared with single-chart plotlint**: `geometry.py`, `Check` protocol, `scoring.py`, `plotlint.core.*`. Dashboard checks produce the same `Issue` and `InspectionResult` types.

### 11.4 Pipeline Graph Extension Pattern

MVP pipeline (linear):
```
load → plan → explore → chart → comply → output → END
```

DI-1/2/3 inserts nodes:
```
load → plan → explore → chart → comply → layout → dash_comply → output → END
```

DI-4.2 adds HITL interrupts:
```
load → plan → [INTERRUPT?] → explore → [INTERRUPT?] → chart → [INTERRUPT?] → comply → ...
```

`build_pipeline_graph()` accepts `extra_nodes` and `enable_hitl` parameters to support this without rewriting the graph builder.

### 11.5 ConvergenceState Extension Pattern

TypedDict `total=False` enables additive field extension. Fields for PL-1.2 (rollback) and PL-1.3 (critic) are already reserved in the MVP schema — the schema is stable from MVP through PL-1.

### 11.6 Check Growth Path

```
MVP:  2 checks (label_overlap, element_cutoff)
PL-1: 5 checks (+legend_occlusion, +empty_plot, +unnecessary_legend)
PL-2: 9 checks (+readability, +y_format, +color_contrast, +colorblind_safety)
```

Each addition: one new file in `checks/`, one import line in `checks/__init__.py`, one enum value in `DefectType`. `inspector.py` unchanged at every step.

---

## 12. Additional Design Decisions

### 12.1 Async Convention

All LangGraph node functions are `async def`. Sync functions (data loading, rendering, file output) are called inside async nodes without wrapping — `await` is only used for actual async operations (LLM calls).

```python
# Example: sync function inside async node
async def load_node(state: PipelineState) -> dict:
    profile = load_and_profile(state["source_path"])  # sync — no await
    return {"data_profile": profile}

# Example: async function inside async node
async def plan_node(state: PipelineState) -> dict:
    steps = await plan_analysis(                       # async — uses await
        profile=state["data_profile"],
        questions=state["questions"],
        llm_client=state_llm_client,
    )
    return {"analysis_steps": steps}
```

### 12.2 LangGraph Partial Returns

Nodes return only the keys they modify, not the full state. LangGraph merges partial updates automatically.

```python
# Good: return only changed keys
async def render_node(state: ConvergenceState) -> dict:
    result = renderer.render(state["source_code"])
    return {
        "png_bytes": result.png_bytes,
        "figure_pickle": result.figure_data,
        "render_error": result.error_message if not result.succeeded else None,
        "iteration": state.get("iteration", 0) + 1,
    }

# Avoid: copying entire state
# return {**state, "png_bytes": result.png_bytes, ...}
```

### 12.3 TYPE_CHECKING Guard for pandas

`autodash/models.py` imports `pd.DataFrame` for `InsightResult`. To avoid triggering the ~200ms pandas import for lightweight consumers:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

class InsightResult:
    result_df: pd.DataFrame  # evaluated only at type-check time
```

### 12.4 DataFrame Handling in Pipeline

The `pd.DataFrame` loaded in `load_node` must reach `explore_node` for code execution. Rather than passing it through LangGraph state (non-serializable), `explore_node` reloads from `source_path`:

```python
async def explore_node(state: PipelineState) -> dict:
    from autodash.data import load_dataframe
    df, _ = load_dataframe(Path(state["source_path"]))
    # ... explore using df ...
```

This is slightly slower (one extra file read) but eliminates DataFrame serialization concerns and is cleanly compatible with LangGraph checkpointing (DI-4.2).

### 12.5 Iteration Counter

`render_node` increments the `iteration` counter before rendering. This ensures exactly one increment per loop iteration and the counter is available for logging during the render.
