# AutoDash / plotlint: Development Plan

Ideated with LLM assistance, structured for agile-friendly milestones.

Refer to [project vision](vision.md) for full architecture and component specifications.

---

## Philosophy

**Build vertically, not horizontally.**

Each iteration delivers a thin slice through the entire plotlint loop (render, inspect, patch, converge). Every version produces something *runnable* and *demonstrable*.

**Value proposition: Deterministic measurement beats blind guessing.**

Programmatic bounding-box extraction gives exact, reproducible spatial measurements for free. By offloading "what is broken and by how much" to the Inspector, the LLM's complexity budget can be spent on generating surgical fixes — not guessing what's wrong from a screenshot. This scales independently of model choice: better measurement benefits Haiku and Opus alike.

---

## Guiding Mantra

> **"Declarative, Modular, SoC"**

Every implementation decision should be evaluated against these three principles:

| Principle | Meaning | Example |
|-----------|---------|---------|
| **Declarative** | Describe *what*, not *how*. Config over code. Data-driven behavior. | Defect checks declare their severity thresholds; chart specs define what to render, not how. Inspector checks are data (defect taxonomy), not hardcoded if-else chains. |
| **Modular** | Components are self-contained, swappable, and independently testable. plotlint works standalone or inside AutoDash. | Renderer, Inspector, Patcher, Critic are independent modules. Swap matplotlib backend for plotly without touching collision detection. |
| **SoC** (Separation of Concerns) | Each module has ONE job. No god objects. Clear boundaries. | Inspector detects issues. Patcher fixes them. Convergence loop orchestrates. No component does two things. |

**Before writing code, ask:**
1. Am I describing behavior or implementing mechanics? (Declarative)
2. Can this be swapped out without ripple effects? (Modular)
3. Does this component have exactly one responsibility? (SoC)

---

## Additional Principles

| Principle | Meaning | Example |
|-----------|---------|---------|
| **Measure, Don't Guess** | Prefer programmatic measurement over LLM inference. Use LLM only where judgment is required. | Inspector extracts pixel-precise bounding boxes; Critic invoked only for semantic checks that require visual reasoning. |
| **Risk-First Validation** | Highest-risk assumptions get validated earliest | Bounding box extraction reliability is validated in the MVP, not in a separate spike |
| **Incremental Value** | Each version improves user-facing capability | MVP produces a working fix loop, not just "rendering works" |
| **Portfolio-First Ordering** | The most impressive artifacts are built first | plotlint convergence GIF is demoable before the full dashboard pipeline exists |

---

## Architecture Overview

```
plotlint/core/ (foundation)             AutoDash (full pipeline)
    │ LLM client, sandbox,                  │
    │ parsing, errors, config               │ Pipeline Orchestration (LangGraph)
    ▼                                        ▼
plotlint (standalone)                    [Data Intelligence]
    │ Visual Compliance Engine           [Analysis Planning]
    ▼                                    [Data Exploration]
[Renderer] → [Inspector] → [Patcher]    [Chart Planning]
     ▲              │             │       [Output Generation]
     │              ▼             │            │
     └────[Convergence Loop]◄────┘            ▼
          (LangGraph graph)           [Dashboard Compliance]
              [Critic (LLM)]            Layout + Style + Spacing
```

> **Dependency rule:** plotlint has ZERO imports from autodash. autodash imports from plotlint (including plotlint.core). See [architecture.md](architecture.md) for full details.

---

## Dependency Graph

### plotlint (Core Engine) Dependencies

```
Renderer (matplotlib Agg sandbox)
    └──► Inspector (bbox extraction + collision detection)
            ├──► Patcher (LLM fix generation) [needs issue list from Inspector]
            │       └──► Convergence Loop (LangGraph) [orchestrates Render→Inspect→Patch]
            │               └──► Critic (LLM vision, selective) [invoked by Loop]
            └──► plotlint CLI (wraps Loop for standalone use)

Inspector defect types (incremental, within Inspector module):
    label_overlap, element_cutoff ────── MVP (foundational, validates bbox extraction)
    legend_occlusion, empty_plot ─────── PL-1
    unnecessary_legend ─────────────── PL-1
    readability, y_format ──────────── PL-2
    color_contrast, colorblind_safety ── PL-2 (WCAG math, color extraction)
```

### Dashboard Intelligence Dependencies

```
plotlint Convergence Loop (must work for single charts first)
    └──► Multi-Chart Foundation (grid layout, multi-chart planning)
            └──► Dashboard Inspector (multi-chart bounding boxes)
                    ├──► Layout Engine (smart grid placement)
                    └──► Style Harmonizer (palette + typography)
```

### AutoDash Pipeline Dependencies

```
Data Intelligence ──────── independent (pandas, no plotlint dependency)
    └──► Analysis Planning [needs profile from Data Intelligence]
            └──► Data Exploration [needs plan]
                    └──► Chart Planning [needs insights]
                            └──► plotlint (per-chart polishing)
                                    └──► Dashboard Composition (multi-chart)
                                            └──► Output Generation
```

---

## Ordering Rationale

**MVP-first, then parallel component tracks.**

The MVP delivers a thin vertical slice through the entire AutoDash pipeline: data loading → analysis → chart generation → visual compliance → output. This validates both the pipeline concept and the highest-risk assumption (matplotlib bounding box extraction) in a single phase.

After the MVP, two component-based tracks (PL: plotlint, DI: Dashboard Intelligence) expand the product in parallel. Each track has 3-4 sub-phases, each independently demoable.

**Why MVP-first:**
1. **De-risk integration early.** Pipeline-plotlint integration bugs are discovered in week 2, not week 8.
2. **bbox extraction still validated early.** MVP.7 validates the thesis in the same timeframe as the original plan.
3. **Each phase is independently demoable.** MVP: "end-to-end works." PL: "the compliance engine is deep." DI: "the dashboards are intelligent."
4. **The novel work still gets full depth.** PL-1 through PL-3 spend ~4-5 weeks deepening plotlint — the same investment as before.

---

## Iteration Structure

```
MVP: Single Chart Flow (sequential, ~2-2.5w)
 │
 ├──► PL: plotlint                                ◄── parallel with DI
 │     │
 │     ├── PL-1: Core Engine (~2-2.5w)
 │     │     Inspector +3 checks, robustness, Critic, CLI, GIF, smoke tests
 │     │
 │     ├── PL-2: Full Taxonomy + Proof (~1.5-2w)
 │     │     Inspector +4 checks (color, readability), benchmarks, gallery, report
 │     │
 │     └── PL-3: Hardening & Integration (selective)
 │           Docker sandbox, CI gate, Jupyter
 │
 └──► DI: Dashboard Intelligence                  ◄── parallel with PL
       │
       ├── DI-1: Multi-Chart Foundation (~1.5-2w)
       │     Multi-step analysis, agent-loop exploration, multi-chart planning,
       │     grid layout, dashboard output + CLI
       │
       ├── DI-2: Layout & Style (~1.5-2w)
       │     Smart layout engine, style harmonizer, HTML output
       │
       ├── DI-3: Compliance (~1.5-2w)
       │     Dashboard inspector, convergence loop, dashboard benchmarks
       │
       └── DI-4: Pipeline Polish (selective)
             PDF output, HITL checkpoints
```

PL and DI share NO dependencies beyond MVP. Sub-phases within each track are sequential, but PL and DI sub-phases interleave freely.

### Effort Summary

| Phase | Effort | Parallel? | Key Deliverable |
|-------|--------|-----------|-----------------|
| **MVP** | 2-2.5 weeks | Sequential | Data → single polished chart PNG |
| **PL-1** Core Engine | 3-3.5 weeks | Parallel with DI-1 | `plotlint script.py` standalone + GIF. 5 defect types. Plotly support |
| **DI-1** Multi-Chart Foundation | 1.5-2 weeks | Parallel with PL-1 | `autodash` → multi-chart dashboard PNG |
| **PL-2** Full Taxonomy + Proof | 1.5-2 weeks | Parallel with DI-2 | 9 defect types + benchmarks + gallery |
| **DI-2** Layout & Style | 1.5-2 weeks | Parallel with PL-2 | Smart layout + style + HTML |
| **DI-3** Compliance | 1.5-2 weeks | Parallel with PL-3 | Dashboard inspector + convergence loop |
| **PL-3** Hardening | 1-2 weeks selective | Any time | Docker sandbox, CI gate, Jupyter |
| **DI-4** Pipeline Polish | 1-1.5 weeks selective | Any time | PDF, HITL checkpoints, JSON output |

**Best case (max parallelization):** MVP (~2.5w) + max(PL-1, DI-1) (~2.5w) + max(PL-2, DI-2) (~2w) + max(DI-3, ...) = **~9 weeks** to core features

---

## MVP: Single Chart Flow

**Goal:** Data in → single insight → single chart → visual compliance → fixed chart PNG. Validates both the pipeline concept and the bbox extraction thesis.

**Estimated effort:** 2-2.5 weeks total

| Item | Effort | Notes |
|------|--------|-------|
| MVP.1 Foundation + LangGraph Scaffold | 2-3 days | State definitions, graph skeleton, stub nodes, plotlint/core/ foundation |
| MVP.2 Data Intelligence | 2-3 days | Load, profile, detect types |
| MVP.3 Analysis Planning | 1-2 days | LLM: questions + profile → single analysis step |
| MVP.4 Data Exploration | 2-3 days | Single-shot: LLM generates pandas code → execute |
| MVP.5 Chart Planning + Code Generation | 2-3 days | Single chart spec + matplotlib code |
| MVP.6 Renderer | 2-3 days | matplotlib Agg subprocess sandbox |
| MVP.7 Inspector Minimal | 3-4 days | bbox extraction + label_overlap + element_cutoff |
| MVP.8 Patcher | 2-3 days | LLM fix generation, one issue per iteration |
| MVP.9 Output | 1 day | Single chart PNG |
| MVP.10 Docker Packaging | 1-2 days | Dockerfile for demo without installation |

### MVP.1: Foundation + LangGraph Scaffold

**Why first:** LangGraph is the orchestration backbone for both the plotlint convergence loop and the auto-dash pipeline. Defining the state schemas and graph skeletons first means Renderer, Inspector, and Patcher are built as graph nodes from day one — not retrofitted later.

**Scope:**
- Define `plotlint/core/` foundation sub-package: `LLMClient` protocol, `execute_code()` sandbox, `parse_code_from_response()` / `parse_json_from_response()` parsing, error hierarchy, `LLMConfig` / `SandboxConfig`
- Define `ConvergenceState` TypedDict (current code, score, iteration count, score history, fix history, best code seen, renderer_type, spec_context)
- Define `PipelineState` TypedDict (data profile, analysis steps, insights, chart specs, rendered charts)
- Define `ConvergenceConfig` (max iterations, target score, stagnation window, score improvement threshold)
- Build convergence loop graph skeleton: render → inspect → decide (conditional edge: patch or stop) → render
- Build pipeline graph skeleton: load → plan → explore → chart → render → comply → output
- Stub nodes that pass through state (replaced by real implementations in MVP.6-8)
- Stop condition edges: score == 1.0, max iterations (default 3), score stagnant for 2 iterations

**Module:** `plotlint/core/`, `plotlint/models.py`, `plotlint/config.py`, `plotlint/loop.py`, `autodash/models.py`, `autodash/config.py`, `autodash/pipeline.py`

### MVP.2: Data Intelligence

**Scope:**
- Load CSV, Excel (.xlsx), Parquet via pandas
- Profile: column types, distributions, nulls, cardinality, min/max/mean
- Detect date columns, categorical columns, high-cardinality text
- Output: `DataProfile` dataclass (JSON-serializable)

**Module:** `autodash/data.py`

### MVP.3: Analysis Planning

**Scope:**
- LLM receives `DataProfile` + user questions
- Outputs a single `AnalysisStep` (what to compute, which columns, what aggregation)
- Validate: step references columns that exist

**Module:** `autodash/planner.py`

### MVP.4: Data Exploration

**Scope:**
- LLM generates pandas code for the analysis step → execute in sandbox → review result
- Max 3 attempts if execution fails
- Output: `InsightResult` (dataframe + text summary)

**Module:** `autodash/explorer.py`

### MVP.5: Chart Planning + Code Generation

**Scope:**
- LLM converts insight into a single `ChartSpec` (chart type, data mapping, title)
- LLM generates matplotlib code for the spec
- Combined because they share prompt context

**Module:** `autodash/charts.py`

---

### MVP.6: Renderer (matplotlib Sandbox)

**Why early:** Every other plotlint component depends on being able to execute chart code and get a rendered figure object back.

**Scope:**
- Execute a user-provided Python script in a subprocess with timeout
- Force matplotlib Agg backend (non-interactive, no display needed)
- Capture the rendered `Figure` object (via pickle or by hooking `savefig`)
- Return both the figure object (for Inspector) and a PNG screenshot (for Critic later)
- Standardize: fixed DPI (100), fixed backend (Agg)
- Error handling: syntax errors, runtime errors, import errors, timeout

**Key design decisions:**
- Subprocess isolation: user code runs in a child process, not in the plotlint process
- Communication: parent sends script path, child returns pickled figure + PNG bytes
- No Docker yet (subprocess + timeout; Docker is a hardening concern for PL-3)

**Risk validation embedded:** If matplotlib's artist tree is not accessible after `savefig()` in the Agg backend, we discover it here. Mitigation: test `fig.get_children()` traversal immediately after render.

**Module:** `plotlint/renderer.py`

### MVP.7: Inspector Foundation (Bounding Box Extraction + Label Overlap)

**Why this is the crown jewel validation:** The entire plotlint thesis depends on being able to extract element positions programmatically. This item validates or invalidates that thesis.

**Scope:**
- Walk the matplotlib artist tree: `fig → axes → xaxis/yaxis → ticklabels, title, legend, patches, lines`
- Call `get_window_extent(renderer)` on each artist to get pixel-space bounding boxes
- Convert all extents to a uniform coordinate system (pixels from top-left)
- Implement `BoundingBox` dataclass with overlap detection methods
- Implement TWO defect checks: `label_overlap` and `element_cutoff`
- Output: structured `InspectionResult` with issues list and numeric score

**Key technical detail:**
```python
renderer = fig.canvas.get_renderer()
for label in ax.xaxis.get_ticklabels():
    bbox = label.get_window_extent(renderer)
    # bbox.x0, bbox.y0, bbox.x1, bbox.y1 in pixels
```
Must be called *after* `fig.canvas.draw()` to ensure layout is finalized.

**Module:** `plotlint/inspector.py`, `plotlint/geometry.py`

### MVP.8: Patcher (LLM Fix Generation)

**Scope:**
- Accept: current source code + issue list (from Inspector) + fix history
- Prompt Claude text API with structured context
- Constraint: fix highest-severity issue only (one per iteration)
- Parse output: extract the modified Python code
- Validate: output is syntactically valid Python (`ast.parse`)
- Return: patched source code string

**Module:** `plotlint/patcher.py`

### MVP.9: Output

**Scope:**
- Render final fixed chart to PNG
- Simple output: save to file

**Module:** `autodash/output.py`

### MVP.10: Docker Packaging

**Scope:**
- Dockerfile with Python, matplotlib, and all dependencies pre-installed
- `docker run autodash sales.csv "questions"` works without local installation
- Mount data files via volume binding
- Output PNG written to mounted output directory

**Module:** `Dockerfile`, `docker-compose.yml`

### MVP Hero Demo

- **"Fix this broken bar chart"**: Feed the monthly sales example. Inspector detects label overlap. Patcher adds rotation. Second render passes.
- **"Inspect only"**: Run Inspector on a chart with overlapping labels. Get structured JSON with issue type, severity, suggestion.
- **"Clean chart passes"**: Run Inspector on a clean chart. Score = 1.0, empty issues list.

### MVP Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `get_window_extent()` returns (0,0,0,0) for some elements | Medium | Test across chart types in MVP.7; document unreliable elements |
| Subprocess communication overhead too high | Low | Profile; switch to in-process `exec()` if needed |
| LLM Patcher generates invalid code | Medium | `ast.parse` validation + retry once on failure |

### MVP Success Metrics

| Metric | Target |
|--------|--------|
| End-to-end runs | `autodash` produces a polished chart PNG from CSV + questions |
| Bbox extraction works | `get_window_extent()` returns valid bboxes for bar, line, scatter |
| Label overlap detected | Inspector catches overlapping labels in 4/5 test cases |
| Patcher fixes overlap | Loop converges (score improves) in 3/5 test cases |
| Pipeline completes | < 60 seconds for a typical single-chart run |

**Deliverable:** Working render-inspect-patch loop for matplotlib label overlap + element cutoff. Validated bbox extraction across bar, line, scatter chart types. Single polished chart from data + questions.

---

# PL: plotlint

Three sub-phases deepening the visual compliance engine. Each is independently demoable.

---

## PL-1: Core Engine

**Goal:** Make plotlint credible as a standalone tool. Expand to 5 defect types, add convergence robustness, ship the CLI, and generate the convergence GIF — the hero demo artifact.

**Estimated effort:** 3-3.5 weeks total

| Item | Effort | Dependencies |
|------|--------|-------|
| PL-1.1 Inspector: legend_occlusion + empty_plot + unnecessary_legend | 2-3 days | MVP Inspector infra |
| PL-1.2 Convergence Robustness | 2-3 days | MVP loop |
| PL-1.3 Critic (LLM vision, selective) | 1-2 days | MVP loop |
| PL-1.4 plotlint CLI | 1-2 days | MVP loop |
| PL-1.5 Convergence GIF Generator | 1 day | MVP renderer |
| PL-1.6 Plotly Renderer + Inspector | 1-2 weeks | MVP Inspector infra |
| PL-1.7 Smoke Test Suite | 1-2 days | Benefits from PL-1.1 |

**Within PL-1:** Items 1-6 are independent of each other. Item 7 benefits from 1 being done first.

### PL-1.1: Inspector Expansion (3 New Defect Types)

**legend_occlusion:** Legend bbox vs data area bbox. Overlap area with data elements → severity.

**empty_plot:** Check if axes contains any data elements (`ax.lines`, `ax.patches`, `ax.collections`). All empty = high severity.

**unnecessary_legend:** Count legend entries. Exactly 1 entry = low severity (adds no info).

**Module:** `plotlint/checks/` (one file per check category)

### PL-1.2: Convergence Robustness

- **Rollback:** If new score < previous score → revert to previous code, add failed fix to history (LangGraph conditional edge)
- **Fix deduplication:** Hash each fix attempt; skip duplicates (state field)
- **Score-drop detection:** Two consecutive drops → stop (oscillation) (conditional edge)
- Track "best code seen" separately from "current code" (state field)

**Module:** `plotlint/loop.py` (extend LangGraph state + edges)

### PL-1.3: Critic (Selective LLM Vision)

Invoked once when Inspector returns 0 issues. Sends rendered PNG to Claude vision API. Checks: chart type appropriateness, color meaning, title/annotation quality. When a `ChartSpec` is provided (e.g., from the AutoDash pipeline), also checks spec conformance — verifying the rendered chart matches the intended chart type, data mapping, and layout. Max 1 Critic call per invocation (cost control).

**Module:** `plotlint/critic.py`

### PL-1.4: plotlint CLI

- `plotlint script.py` — full fix loop, outputs fixed code
- `plotlint inspect script.py` — inspect only, no fixing
- `plotlint inspect script.py --format json` — machine-readable output
- `plotlint script.py --max-iterations 3` — configurable
- `plotlint script.py --no-critic` — skip LLM vision

**Module:** `plotlint/cli.py`

### PL-1.5: Convergence GIF Generator

The hero demo artifact. After each render, save annotated PNG with semi-transparent bounding box overlays (red for issues, green for passing). Assemble into animated GIF with 1.5s per frame. Overlay: iteration number, score, issue summary.

**Module:** `plotlint/viz.py`

### PL-1.6: Plotly Renderer + Inspector

Plotly produces nicer charts than matplotlib by default. Adding Plotly support early makes both plotlint and auto-dash output significantly more impressive.

**Scope:**
1. Render Plotly charts in headless Playwright
2. Inject JavaScript to query the rendered SVG/Canvas: `getBoundingClientRect()` on tick labels, legend, title, plot area
3. Return JSON of all element positions
4. Run same collision detection logic (shared `geometry.py`)
5. Update Patcher prompts to handle Plotly code (different API than matplotlib)

**Key design:** The Inspector's defect checks (`label_overlap`, `element_cutoff`, etc.) operate on `BoundingBox` dataclasses — renderer-agnostic. Only the extraction step differs between matplotlib (artist tree walk) and Plotly (Playwright JS injection).

**Module:** `plotlint/renderers/plotly.py`

### PL-1.7: Smoke Test Suite

**Scope:**
- 5-8 hand-written matplotlib and Plotly scripts with known label overlap, cutoff, legend, and empty plot defects
- Varying severity: 4 categories, 8 categories, 12 categories, long text labels
- pytest-based: run full loop, verify Inspector detects the issue, verify Patcher fixes it

**Module:** `tests/smoke/`, `tests/fixtures/`

### PL-1 Hero Demos

- **Convergence GIF**: Monthly sales example, 3-4 frames broken → fixed with visible bounding boxes
- **Multi-defect chart**: label overlap + legend occlusion + element cutoff. Loop fixes all three. Score: 0.20 → 0.55 → 0.78 → 1.00
- **CLI in terminal**: `plotlint broken_chart.py --output fixed.py` works end-to-end
- **Rollback demo**: Fixing one issue breaks another. Loop detects score drop, rolls back, tries alternative

### PL-1 Success Metrics

| Metric | Target |
|--------|--------|
| Defect types | 5 (label_overlap, element_cutoff, legend_occlusion, empty_plot, unnecessary_legend) |
| Convergence GIF | Clearly shows improvement across frames |
| Rollback | Triggers correctly when score drops |
| CLI | `plotlint script.py` runs end-to-end |
| Smoke tests | 6/8 pass (detection + convergence) |

**Deliverable:** 5 defect types. Convergence with rollback. Working CLI. Convergence GIF generator. Plotly renderer + inspector.

---

## PL-2: Full Taxonomy + Proof

**Goal:** Complete defect taxonomy, build benchmark suite for quantitative claims, generate before/after gallery. After PL-2, plotlint is portfolio-ready as a standalone tool.

**Estimated effort:** 1.5-2 weeks total

| Item | Effort | Dependencies |
|------|--------|-------|
| PL-2.1 Inspector: readability + y_axis_formatting | 2-3 days | MVP Inspector infra |
| PL-2.2 Inspector: color_contrast + colorblind_safety | 3-4 days | MVP Inspector infra + new color_utils |
| PL-2.3 Benchmark Suite | 3-4 days | Benefits from PL-2.1 + PL-2.2 |
| PL-2.4 Before/After Gallery Generator | 1-2 days | PL-2.3 |
| PL-2.5 Quantitative Report | 1-2 days | PL-2.3 |

**Within PL-2:** Items 1-2 are independent. Items 3-5 form a chain.

### PL-2.1: Inspector — Readability + Formatting

**readability:** Font size relative to figure dimensions. Below threshold = defect.

**y_axis_formatting:** Detect raw large numbers (>1000) without comma/K/M formatting.

### PL-2.2: Inspector — Color Checks (WCAG + Colorblind)

**color_contrast:** Extract fg/bg colors, compute WCAG contrast ratio. Below AA (4.5:1) = high. Below AAA (7:1) = medium.

**colorblind_safety:** Simulate deuteranopia/protanopia (matrix transform on RGB). Check if palette colors become indistinguishable (deltaE < threshold).

**Module:** `plotlint/checks/color.py`, `plotlint/color_utils.py`

### PL-2.3: Benchmark Suite

30-50 matplotlib and Plotly scripts with known defects:

| Category | Count | Examples |
|----------|-------|---------|
| Label overlap | 8-10 | Rotation needed, too many categories, long text, datetime labels |
| Legend problems | 5-7 | Occlusion, unnecessary single-entry, poor placement |
| Element cutoff | 5-7 | Title clipped, labels extend beyond figure |
| Color issues | 5-7 | Low contrast, colorblind-unsafe, WCAG failures |
| Readability / formatting | 3-5 | Tiny fonts, unformatted axes |
| Multi-defect | 4-6 | Combinations of the above |

Metadata per script: expected defects, severity, chart type.

**Module:** `benchmarks/scripts/`, `benchmarks/manifest.json`

### PL-2.4: Before/After Gallery

Run plotlint on each benchmark script. Capture before/after PNGs. Generate HTML gallery with side-by-side views and issue summaries. Portfolio artifact.

### PL-2.5: Quantitative Report

- Detection accuracy (% of known defects identified)
- False positive rate
- Convergence rate (% reaching score 1.0 or threshold)
- Average iterations to converge
- Regression rate (patches that break prior fixes) — target <5%
- Time per iteration (mean, p95)

### PL-2 Hero Demos

- **Full benchmark run**: `python -m benchmarks.runner` processes 30-50 scripts, outputs report
- **Gallery artifact**: `gallery.html` with 30+ before/after pairs
- **Quantitative claim**: "plotlint detects X/50 defects, fixes Y/50 autonomously"

### PL-2 Success Metrics

| Metric | Target |
|--------|--------|
| Detection accuracy | >90% on benchmark suite |
| Convergence rate | >80% of defects fixed |
| False positive rate | <10% |
| Average iterations | <5 |
| Gallery | 30+ before/after pairs |

**Deliverable:** Complete defect taxonomy (9 types). Benchmark suite. Gallery. Quantitative report. plotlint is portfolio-ready.

---

## PL-3: Hardening & Integration

**Goal:** Production readiness and broader ecosystem support. Items are independent — prioritize by need.

**Estimated effort:** 2-3 weeks (selective)

| Item | Effort | Dependencies |
|------|--------|-------|
| PL-3.1 Docker Code Sandbox | 2-3 days | MVP.6 (renderer subprocess) |
| PL-3.2 CI gate (`plotlint inspect --json`) | 1 day | PL-1.4 (CLI) |
| PL-3.3 Jupyter cell magic | 2-3 days | PL-1.4 (CLI) |

### PL-3.1: Docker Code Sandbox

Run user-provided chart scripts inside a Docker container instead of a bare subprocess. Provides OS-level isolation for untrusted code. The `Renderer` protocol is unchanged — `MatplotlibRenderer` is configured to use `docker exec` instead of `subprocess.run`.

**Module:** `plotlint/core/sandbox.py` (extend), `plotlint/renderer.py` (configure)

### PL-3.2: CI Gate

`plotlint inspect script.py --format json` as a CI/CD gate. Non-zero exit on issues above a configurable severity threshold. Machine-readable JSON output.

### PL-3.3: Jupyter Cell Magic

`%%plotlint` cell magic for notebook users. Wraps the convergence loop around cell output.

---

# DI: Dashboard Intelligence

Four sub-phases building multi-chart dashboard capabilities. Each is independently demoable.

---

## DI-1: Multi-Chart Foundation

**Goal:** Scale from single chart to multi-chart dashboard. The pipeline becomes truly useful.

**Estimated effort:** 1.5-2 weeks total

| Item | Effort | Dependencies |
|------|--------|-------|
| DI-1.1 Multi-step Analysis Planning | 1-2 days | MVP analysis planning |
| DI-1.2 Agent-Loop Data Exploration | 2-3 days | MVP exploration |
| DI-1.3 Multi-Chart Planning | 1-2 days | MVP chart planning |
| DI-1.4 Simple Grid Layout | 1-2 days | MVP renderer |
| DI-1.5 Dashboard Output + CLI | 1-2 days | MVP output |

**Within DI-1:** Items are loosely sequential (1→2→3→4→5) since each builds on the previous, but 4 and 5 are independent of 1-3.

### DI-1.1: Multi-step Analysis Planning

LLM receives `DataProfile` + user questions. Outputs list of `AnalysisStep` (what to compute, which columns, what aggregation). Validate: each step references columns that exist.

**Module:** `autodash/planner.py` (extend from MVP.3)

### DI-1.2: Agent-Loop Data Exploration

Agent loop: for each step, LLM generates pandas code → execute in sandbox → review result → iterate if needed (max 3 attempts per step). Output: list of `InsightResult` (dataframe + text summary).

**Module:** `autodash/explorer.py` (extend from MVP.4)

### DI-1.3: Multi-Chart Planning

Given analysis results + original questions, determine: which insights need visualization, what chart type for each, visual hierarchy (primary vs supporting), how many charts (3-6 typical).

Output: List of `ChartSpec` with priority field.

**Module:** `autodash/charts.py` (extend from MVP.5)

### DI-1.4: Simple Grid Layout

Accept `List[ChartSpec]` with priorities, produce a `matplotlib.gridspec` layout. Primary chart gets 2x area. No compliance checks yet.

**Module:** `plotlint/dashboard/layout.py`

### DI-1.5: Dashboard Output + CLI

Render composed dashboard to PNG. `autodash data.csv "questions"` CLI entry point.

**Module:** `autodash/output.py` (extend), `autodash/cli.py`

### DI-1 Hero Demo

**End-to-end**: `autodash sales.csv "What drove Q4 revenue? How do regions compare?"` → 4-chart dashboard PNG.

### DI-1 Success Metrics

| Metric | Target |
|--------|--------|
| Multi-chart output | 3-6 charts in a single dashboard from one command |
| Agent loop | Completes analysis steps with < 3 retries per step |
| End-to-end | < 120 seconds for a typical 4-chart run |

**Deliverable:** `autodash` produces multi-chart dashboard PNG from CSV + questions.

---

## DI-2: Layout & Style

**Goal:** Make dashboards look professionally composed. Replace simple grid with intelligent layout.

**Estimated effort:** 1.5-2 weeks total

| Item | Effort | Dependencies |
|------|--------|-------|
| DI-2.1 Layout Engine Full | 3-4 days | DI-1.4 (replaces simple grid) |
| DI-2.2 Style Harmonizer | 3-4 days | DI-1 rendered charts |
| DI-2.3 HTML Output | 1-2 days | DI-1.5 output module |

**Within DI-2:** All items are independent.

### DI-2.1: Layout Engine (Full)

- Accept List of `ChartSpec` with priority (primary, secondary, supporting)
- Primary chart gets 2x area, secondary 1x, supporting 0.5x
- Respect aspect ratios and minimum sizes
- Output: `LayoutSpec` (grid positions and sizes)

**Module:** `plotlint/dashboard/layout.py`

### DI-2.2: Style Harmonizer

- Palette normalization across charts
- Typography normalization (font family, sizes)
- Theme application (light/dark, color scheme presets)

**Module:** `plotlint/dashboard/style.py`

### DI-2.3: HTML Output

Render dashboard to interactive HTML with linked PNG fallback. Extends DI-1's PNG-only output.

### DI-2 Hero Demo

4 charts with different fonts, colors, styles → after harmonization: consistent, cohesive look. Priority chart spans two columns.

### DI-2 Success Metrics

| Metric | Target |
|--------|--------|
| Layout engine | Reasonable grid for 3-6 charts with mixed priorities |
| Style harmonizer | Produces visually consistent dashboards |
| HTML output | Interactive version loads in browser |

**Deliverable:** Professional grid layout with consistent styling. HTML output.

---

## DI-3: Compliance

**Goal:** Automated quality checking for multi-chart dashboards. The second novel capability.

**Estimated effort:** 1.5-2 weeks total

| Item | Effort | Dependencies |
|------|--------|-------|
| DI-3.1 Dashboard Inspector | 3-4 days | MVP renderer + Inspector infra |
| DI-3.2 Dashboard Convergence Loop | 1-2 days | DI-3.1 |
| DI-3.3 Dashboard Benchmark Suite | 2-3 days | DI-3.1 |

**Within DI-3:** Item 1 first, then 2 and 3 (independent of each other).

### DI-3.1: Dashboard Inspector

Extend Inspector paradigm from single charts to multi-chart layouts.

| Check | Method |
|-------|--------|
| Chart overlap | Bounding box collision between chart containers |
| Spacing consistency | Measure inter-chart gaps; flag inconsistent margins |
| Alignment | Charts should align to grid; detect drift |
| Color collision | Same color for different meanings across charts |
| Style drift | Inconsistent font sizes, title styles across charts |
| Hierarchy clarity | Primary chart should have largest area |

**Module:** `plotlint/dashboard/renderer.py`, `plotlint/dashboard/checks/`

### DI-3.2: Dashboard Convergence Loop

Render dashboard → Inspect layout → Patch (resize/reposition) → Re-render. Separate LangGraph graph reusing the same pattern as the single-chart loop.

### DI-3.3: Dashboard Benchmark Suite

Multi-chart layout test cases with known defects. Verify Dashboard Inspector detects spacing, alignment, and style issues.

### DI-3 Hero Demo

4 charts with inconsistent spacing and one overlapping another. Dashboard Inspector flags all issues. Convergence loop fixes them.

### DI-3 Success Metrics

| Metric | Target |
|--------|--------|
| Dashboard checks | 6 types detected across 3+ multi-chart layouts |
| Convergence | Dashboard loop fixes spacing/alignment issues |

**Deliverable:** Dashboard-level defect detection (6 check types). Dashboard convergence loop. Dashboard benchmarks.

---

## DI-4: Pipeline Polish

**Goal:** Refinements to the full auto-dash experience. Items are independent — prioritize by need.

**Estimated effort:** 1-1.5 weeks (selective)

| Item | Effort | Dependencies |
|------|--------|-------|
| DI-4.1 PDF Output | 1-2 days | DI-1.5 output module |
| DI-4.2 Human-in-the-loop Checkpoints | 2-3 days | MVP pipeline |
| DI-4.3 JSON Output | 1 day | DI-1.5 output module |

### DI-4.1: PDF Output

Render dashboard to PDF via matplotlib.

### DI-4.2: Human-in-the-loop Checkpoints

Optional pause points at key decision stages using LangGraph's `interrupt()` + checkpointing:
1. **After analysis plan:** "Also look at customer segments" / "Looks good, proceed"
2. **After data exploration:** "Dig deeper into enterprise" / "Move on to charts"
3. **After chart planning:** "Skip the pie, add a heatmap instead"
4. **After dashboard draft:** "Make the trend chart bigger" / "Approved"

Each checkpoint is optional (user can auto-approve all). plotlint's visual compliance runs without human intervention. LangGraph's built-in persistence allows resuming a paused pipeline across sessions.

Pipeline prompt refinement is ongoing work folded into regular development, not a separate item.

### DI-4.3: JSON Output

Serialize chart specs + data mappings to JSON for embedding in other tools. Enables programmatic consumption of auto-dash output by downstream applications.

**Module:** `autodash/output.py`

---

## Module Mapping

| Component | Module |
|-----------|--------|
| **plotlint/core/ (foundation)** | |
| LLM client abstraction | `plotlint/core/llm.py` |
| Code execution sandbox | `plotlint/core/sandbox.py` |
| LLM response parsing | `plotlint/core/parsing.py` |
| Error hierarchy | `plotlint/core/errors.py` |
| Shared config (LLM, sandbox) | `plotlint/core/config.py` |
| **plotlint (compliance engine)** | |
| Renderer (matplotlib sandbox) | `plotlint/renderer.py` |
| Inspector (bbox extraction) | `plotlint/inspector.py` |
| Geometry utilities | `plotlint/geometry.py` |
| Element map + Extractor protocol | `plotlint/elements.py` |
| Individual defect checks | `plotlint/checks/` |
| Matplotlib extractor | `plotlint/extractors/matplotlib.py` |
| Color utilities (WCAG, CVD) | `plotlint/color_utils.py` |
| Patcher (LLM fix generation) | `plotlint/patcher.py` |
| Critic (LLM vision) | `plotlint/critic.py` |
| Convergence loop (LangGraph) | `plotlint/loop.py` |
| Convergence config | `plotlint/config.py` |
| Scoring | `plotlint/scoring.py` |
| GIF generator | `plotlint/viz.py` |
| Gallery generator | `plotlint/gallery.py` |
| plotlint CLI | `plotlint/cli.py` |
| **plotlint/dashboard/ (dashboard compliance)** | |
| Dashboard renderer | `plotlint/dashboard/renderer.py` |
| Dashboard checks | `plotlint/dashboard/checks/` |
| Layout engine (simple grid → full) | `plotlint/dashboard/layout.py` |
| Style harmonizer | `plotlint/dashboard/style.py` |
| **autodash (pipeline)** | |
| Data intelligence | `autodash/data.py` |
| Analysis planner | `autodash/planner.py` |
| Data explorer | `autodash/explorer.py` |
| Chart planner + code gen | `autodash/charts.py` |
| Output generation | `autodash/output.py` |
| AutoDash pipeline (LangGraph) | `autodash/pipeline.py` |
| Pipeline config | `autodash/config.py` |
| AutoDash CLI | `autodash/cli.py` |
| **Benchmarks** | |
| Benchmark scripts | `benchmarks/scripts/` |
| Benchmark runner + report | `benchmarks/runner.py`, `benchmarks/report.py` |

---

## Testing Strategy

### Levels of Testing

| Level | What | How | When |
|-------|------|-----|------|
| **Unit tests** | Individual components (geometry, color_utils, checks) | pytest with fixture charts | Throughout |
| **Smoke tests** | Minimal loop validation | 5-8 hand-written scripts | PL-1 |
| **Integration tests** | Full render-inspect-patch loop | Benchmark scripts as fixtures | PL-2 onwards |
| **Benchmark harness** | Quantitative detection + convergence measurement | 30-50 scripts with ground truth | PL-2 onwards |
| **Hero demos** | Manual validation + portfolio artifacts | Specific scripts per milestone | Every phase |

### Testing Progression

| Phase | Testing Approach |
|-------|------------------|
| **MVP** | Manual validation. End-to-end run produces a chart. Inspector detects known issues. |
| **PL-1** | Smoke tests (5-8 scripts). CLI integration tests. Hero demos (convergence GIF). |
| **PL-2** | Full benchmark harness. Quantitative report. Gallery generation. |
| **DI-1** | Manual validation. Multi-chart pipeline produces dashboard. |
| **DI-2/3** | Run benchmark after each milestone. Extend for dashboard checks. |

### Self-Validation

The convergence GIF is itself a test artifact. If the GIF shows improvement across frames, the loop works. If it shows thrashing, there's a bug. The artifact is both the demo and the test.

---

## Overall Success Metrics

| Milestone | Metric |
|-----------|--------|
| MVP | Label overlap + cutoff detected in 4/5 test scripts; loop converges for 3/5; bbox extraction works for bar + line + scatter |
| PL-1 | 5 defect types detected; convergence GIF shows clear improvement; CLI runs end-to-end; rollback triggers correctly |
| PL-2 | Detection accuracy >90% on benchmark; convergence rate >80%; false positive rate <10%; gallery with 30+ pairs |
| DI-1 | `autodash` produces 3-6 chart dashboard from a single command |
| DI-2 | Layout engine produces reasonable grid; style harmonizer produces consistent output |
| DI-3 | Dashboard-level defects detected across 3+ layouts |

---

## What We're NOT Doing

| Feature | Rationale |
|---------|-----------|
| Plotly support in MVP | matplotlib first for MVP; Plotly added in PL-1 with Playwright. |
| Web UI / frontend | CLI-first. Web is a future concern. |
| Custom chart libraries (altair, bokeh, seaborn) | matplotlib covers 90%. Modular renderer design allows future additions. |
| Fine-tuned models | General-purpose Claude text + vision is sufficient. |
| Docker sandboxing for user code | subprocess + timeout for dev. Docker for code isolation is separate from Docker for packaging. |
| Jupyter cell magic in MVP | Future integration point (PL-3). |
| Multi-language support | Python only. Chart code is Python. |

---

## Orchestration Decision (Resolved: LangGraph)

**Question:** Should AutoDash use a plain while loop + state dict, or a framework like LangGraph?

**Two orchestration surfaces:**
1. **plotlint convergence loop** (render → inspect → patch → repeat): State machine with rollback, branching, score tracking
2. **AutoDash pipeline** (data → analysis → exploration → charts → output): Sequential with retries, checkpoints, human-in-the-loop

**Option A: Plain Python (while loop + state dict)**

| Pro | Con |
|-----|-----|
| Simple, no framework overhead | Manual state management |
| Full control, easy to debug | No built-in persistence/checkpointing |
| No learning curve | Rollback/branching logic is hand-rolled |
| "I understand this deeply" portfolio signal | Less impressive on resume/portfolio |

**Option B: LangGraph** (chosen)

| Pro | Con |
|-----|-----|
| Built-in state machines, branching, persistence | Framework dependency and learning curve |
| Checkpointing and replay for debugging | May be over-engineering for the loop complexity |
| Human-in-the-loop support built in | Hides the orchestration logic behind framework abstractions |
| Portfolio signal: "I know the AI agent ecosystem" | Risk of "used a framework where a for loop suffices" signal |

**Why LangGraph wins:** Both orchestration surfaces are genuine state machines — the convergence loop needs rollback, branching, and score tracking; the pipeline needs HITL checkpoints with persistence (resume after user pauses). Building checkpointing and interrupt/resume from scratch would be reinventing what LangGraph provides. The HITL argument is especially compelling: `interrupt()` + checkpointing is a real feature (DI-4.2), not a nice-to-have. The "over-engineering" risk is mitigated by the actual complexity of the problem.
