# AutoDash: End-to-End Data → Dashboard

## Executive Summary

LLMs can generate chart code. Multimodal models can even look at the rendered output. But **seeing is not measuring** — a vision model says "the labels look crowded" while the actual problem is "labels 3 and 4 overlap by 12px." The standard fix loop (screenshot → vision model → guess a fix → re-render → repeat) is expensive (~$0.03/image), slow (2-5s per call), imprecise, and non-deterministic.

**AutoDash** is an end-to-end pipeline: data tables + natural language questions → polished, multi-chart dashboards.

Its core engine, **plotlint**, closes the visual feedback loop through **programmatic spatial analysis** — extracting bounding boxes directly from the rendering engine and running collision detection — not screenshot guessing. The Inspector gives pixel-precise, deterministic measurements for free, in milliseconds. LLM vision is reserved for the 20% of issues that require semantic judgment (color meaning, chart type fit). This hybrid is what makes it work.

```
$ autodash sales.csv "What drove Q4 revenue? How do regions compare?"
→ Loads & profiles data
→ Plans analysis, explores data, identifies insights
→ Generates 4 charts, each polished by plotlint (visual compliance engine)
→ Composes dashboard with layout intelligence
→ Outputs: dashboard.html (interactive) + dashboard.png (static)
```

```
$ plotlint broken_chart.py
→ Renders chart, inspects for visual defects
→ Fixes: label overlap, legend occlusion, y-axis formatting
→ Converges in 3 iterations (score: 0.30 → 0.72 → 0.88 → 1.00)
→ Outputs: fixed code + convergence GIF
```

---

## The Problem

LLMs generate chart code that is syntactically correct but visually broken: overlapping labels, legends covering data, titles cut off, colors indistinguishable in colorblind mode. The code runs. It just looks terrible.

Multimodal models can now *see* rendered charts. But there are three layers to this problem:

### Layer 1: The Feedback Loop is Still Manual

Most LLM coding tools (Code Interpreter, Claude artifacts, Copilot) send the rendered image to the user, not back into the model's context. The user screenshots, pastes back, says "fix the labels," gets new code, re-runs, finds the legend is now broken, pastes again. **The human is still the feedback loop.**

### Layer 2: Vision Models Can See But Can't Measure

Even when a vision model does see the chart, it operates on vibes, not geometry:

| What you need | What vision models give you |
|---------------|---------------------------|
| "Labels 3 and 4 overlap by 12px" | "The labels seem a bit crowded" |
| "Legend covers 23% of the data area" | "The legend might be in the way" |
| "WCAG contrast ratio is 2.8:1 (min 4.5:1)" | "The colors could be more distinct" |
| Deterministic, reproducible | Different answer each call |
| Free, <100ms | ~$0.03/image, 2-5 seconds |

The naive agentic approach (render → screenshot → vision model → guess fix → re-render) works sometimes but is **expensive, slow, imprecise, and non-deterministic**. It's what most "AI visualization" tools will build because it's obvious. It's also mediocre.

### Layer 3: Fixes Interact

Visual layout problems are emergent and fixes cascade. Rotating labels changes the figure's effective height, which pushes the legend, which now overlaps the title. A linear "fix one thing" approach doesn't work — you need a convergence loop that re-inspects the *entire* chart after each fix, detects regressions, and rolls back when things get worse.

And even when individual charts look fine, assembling them into a coherent dashboard is another manual process: deciding layout, ensuring consistent styling, creating visual hierarchy.

### plotlint's Approach

**Measure, don't guess.** The Inspector extracts bounding boxes directly from the rendering engine's internal representation (matplotlib's artist tree, plotly's DOM) and runs geometric collision detection. This gives exact, actionable measurements — for free, in milliseconds, deterministically. LLM vision is reserved for the ~20% of issues that genuinely require semantic judgment.

**AutoDash automates both loops.** plotlint handles per-chart visual compliance. The Dashboard Compliance Agent handles multi-chart composition.

### Why This Requires Agents, Not Scripts

| Capability | Script / Linter | plotlint |
|------------|-----------------|----------|
| **Detection** | Rule-based checks (font size > 8?) | Sees that labels visually overlap at render time |
| **Diagnosis** | One check at a time | Reasons about interactions (fixing overlap might push legend off-canvas) |
| **Repair** | Cannot modify code | Rewrites the specific matplotlib/plotly call to fix the issue |
| **Verification** | Static analysis only | Re-renders and confirms the fix actually worked |
| **Convergence** | Single pass | Loops until all issues resolved or max retries hit |

The core insight: **visual layout problems are emergent**. You cannot statically analyze Python code and know that labels will overlap — it depends on the data, the figure size, the font, the DPI, and how matplotlib's layout engine distributes space. You must render to know.

---

## Architecture Overview

```
AutoDash
├── Pipeline Orchestration (LangGraph — state graph with HITL checkpoints)
│   ├── Data Intelligence: load, profile, detect types
│   ├── Analysis Planning: LLM converts questions → analysis plan
│   ├── Data Exploration: agent loop executes pandas code, finds insights
│   ├── Chart Planning: which insights → which chart types
│   └── Output Generation: PNG/PDF, interactive HTML, JSON export
│
└── plotlint: Visual Compliance Engine (deep component — the novel work)
    ├── Inspector ← THE CROWN JEWEL
    │   Programmatic bounding-box extraction + collision detection
    ├── Renderer
    │   Sandboxed execution + screenshot capture
    ├── Critic
    │   Selective LLM vision for semantic issues only
    ├── Patcher
    │   LLM-powered surgical code fixes
    ├── Convergence Loop (LangGraph state graph)
    │   Score tracking, rollback, fix deduplication
    └── Dashboard Compliance
        Multi-chart layout, style harmony, spacing checks
```

**What's novel vs plumbing:**
- Pipeline stages (data loading, LLM planning, exploration) = plumbing. Every LLM agent framework does this.
- plotlint (visual compliance through programmatic spatial analysis) = **novel**. Nobody does this.
- Dashboard Compliance (automated multi-chart layout checking) = **novel**. Nobody does this either.

### Modular Design

Every component communicates via well-defined interfaces. plotlint works standalone (`plotlint script.py`) or inside the AutoDash pipeline. Chart and dashboard specifications are renderer-agnostic — the same spec can target matplotlib, plotly, or future renderers.

| Component | Input | Output | Standalone? |
|-----------|-------|--------|-------------|
| Data Intelligence | Raw data | Schema + profile | Yes |
| Chart Planner | Insights + questions | List[ChartSpec] | No |
| **plotlint** | Chart code | Fixed chart code | **Yes** |
| Layout Engine | List[ChartSpec] | LayoutSpec | Yes |
| **Dashboard Compliance** | Dashboard render | Fixed layout | **Yes** |
| Dashboard Renderer | DashboardSpec | HTML/PNG | Yes |

---

## plotlint: Visual Compliance Engine

This is where the engineering depth lives.

### The Core Loop

```
┌──────────────┐
│ Render Code  │  Execute in sandbox, capture screenshot
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Inspect      │  Programmatic: bounding boxes, collision detection
└──────┬───────┘
       │
       ▼
┌──────────────┐         ┌──────────────┐
│ Issues = 0?  │── yes ──│ Critique     │  LLM vision (semantic validation)
└──────┬───────┘         └──────┬───────┘
       │ no                     │
       │                        ▼
       │                 ┌──────────────┐
       │                 │ Pass?        │── yes ──── DONE (return fixed code)
       │                 └──────┬───────┘
       │                        │ no (semantic issues found)
       │                        │
       ├────────────────────────┘
       ▼
┌──────────────┐
│ Patch Code   │  LLM generates targeted fix (one issue per iteration)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Max retries? │──── yes ──── STOP (return best attempt + report)
└──────┬───────┘
       │ no
       └──────── back to Render
```

### The Inspector (Crown Jewel)

The Inspector is the key technical insight that makes plotlint work better than a naive "screenshot → GPT-4o → fix" approach.

**The problem with vision models for spatial analysis:** They can say "something looks off" but not "label A's right edge is at pixel 342, label B's left edge is at pixel 338, overlap = 4px." The Inspector gives exact, actionable measurements — for free, in milliseconds.

**How it works:** After rendering, the Inspector extracts element positions directly from the rendering engine's internal representation, then runs geometric checks.

#### Extracting Element Positions from matplotlib

matplotlib's Agg backend renders to a pixel buffer, but the Artist objects retain their bounding boxes in figure coordinates.

```
Extraction: Walk the artist tree after fig.savefig()

fig
 +-- axes[0]
 |    +-- xaxis
 |    |    +-- ticklabels[] ──→ get_window_extent() for each
 |    +-- yaxis
 |    |    +-- ticklabels[] ──→ get_window_extent()
 |    +-- title             ──→ get_window_extent()
 |    +-- legend            ──→ get_window_extent()
 |    +-- patches[]         ──→ bar positions
 |    +-- lines[]           ──→ line positions
 +-- suptitle               ──→ get_window_extent()

Convert all extents to pixel coordinates → run collision detection
```

#### Extracting from plotly/HTML Charts

For browser-rendered charts (plotly, D3):

```
1. Render in headless Playwright
2. Inject JavaScript to query the rendered SVG/Canvas:
   - document.querySelectorAll('.xtick text')  → getBoundingClientRect()
   - document.querySelectorAll('.legend')       → getBoundingClientRect()
   - document.querySelector('.plot-container')  → getBoundingClientRect()
3. Return JSON of all element positions
4. Run same collision detection logic
```

#### Collision Detection

```
For two bounding boxes A and B:

  A overlaps B if:
    A.left < B.right AND A.right > B.left AND
    A.top < B.bottom AND A.bottom > B.top

  Overlap area = max(0, min(A.right, B.right) - max(A.left, B.left))
               × max(0, min(A.bottom, B.bottom) - max(A.top, B.top))

  Severity = overlap_area / min(A.area, B.area)
    > 0.5 = high (element substantially hidden)
    > 0.1 = medium (partially obscured)
    > 0.0 = low (touching/minor)
```

#### Defect Taxonomy

The Inspector checks for these defects programmatically (no LLM, no API cost):

| Defect | Detection Method | Severity Heuristic |
|--------|-----------------|-------------------|
| **Label overlap** | Bounding box intersection of tick labels | % of labels colliding |
| **Element cutoff** | Element bounds vs figure bounds | Area outside figure / total area |
| **Legend occlusion** | Legend bbox vs data area bbox | Overlap area with data elements |
| **Empty plot** | Check if plot area contains rendered data elements | Binary |
| **Readability** | Font size relative to figure dimensions | Below threshold = high |
| **Color contrast** | Extract fg/bg colors, compute WCAG contrast ratio | Below AA = high, below AAA = medium |
| **Colorblind safety** | Simulate deuteranopia/protanopia, check palette distinguishability | Confusable pairs = medium |
| **Y-axis formatting** | Detect raw large numbers without formatting | Numbers > 1000 unformatted = low |
| **Unnecessary legend** | Legend with single entry | Always low |

Inspector output example:
```json
{
  "issues": [
    {
      "type": "label_overlap",
      "severity": "high",
      "details": "X-axis labels overlap: 10 of 12 labels collide",
      "suggestion": "Rotate labels 45-90 degrees or reduce label count"
    },
    {
      "type": "color_contrast",
      "severity": "medium",
      "details": "WCAG AA contrast ratio 2.8:1 (minimum 4.5:1) for bar fill vs background",
      "suggestion": "Darken bar color or lighten background"
    }
  ],
  "score": 0.35
}
```

#### Why This Beats Naive LLM Vision

| Aspect | Inspector (programmatic) | GPT-4o vision |
|--------|------------------------|---------------|
| **Precision** | "Labels 3 and 4 overlap by 12px" | "The labels seem crowded" |
| **Cost** | Free | ~$0.01-0.03 per image |
| **Speed** | <100ms | 2-5 seconds |
| **Determinism** | Same input → same output | Varies between calls |
| **Spatial accuracy** | Pixel-perfect | Approximate |
| **Semantic understanding** | None | "Red for profit is misleading" |

**The hybrid:** Inspector handles spatial/geometric issues (80% of defects). The Critic (LLM vision) handles semantic issues that require judgment. Critic is expensive, so it only runs when the Inspector finds nothing left to fix, or when a reference spec is provided.

### The Critic (Selective LLM Vision)

Invoked selectively — not every iteration. Handles issues the Inspector can't catch:

| Check | Why LLM Needed |
|-------|---------------|
| Semantic clarity | "Is this chart communicating the insight well?" |
| Color meaning | "Red for profit and green for loss is misleading" |
| Chart type fit | "Pie chart with 25 slices → suggest bar chart" |
| Annotation quality | "Title says 'Chart' — not descriptive" |
| Spec conformance | "User asked for grouped bar, this is stacked" |

### The Patcher

LLM generates a targeted code fix given:
- Current source code
- Issue list (Inspector + Critic)
- History of previous attempts (to avoid repeating failed fixes)

Constraints:
- Fix **one issue per iteration** (prevents cascading breakage)
- Priority order: highest severity first
- Preserve data and intent — only modify presentation
- Surgical: change specific lines/parameters, don't rewrite entire script

### Convergence Strategy

The loop must converge (terminate) and improve (not thrash).

**Stop conditions:**

| Condition | Action |
|-----------|--------|
| Score = 1.0 (no issues) | Return fixed code |
| Max iterations (default: 3) | Return best attempt + remaining issues |
| Score stagnant for 2 iterations | Stop — stuck in local minimum |
| Patch introduced new issue | Revert, try alternative fix |

**Avoiding thrash:**
1. Fix highest-severity issue first
2. After each fix, re-run full inspection (not just the targeted issue)
3. Track score history — if score drops, revert and try alternative
4. Include fix history in Patcher context so LLM doesn't repeat failures

**State between iterations:**
```json
{
  "iteration": 3,
  "current_code": "...",
  "current_score": 0.62,
  "score_history": [0.35, 0.55, 0.62],
  "fixes_attempted": [
    {"issue": "label_overlap", "fix": "rotation=45", "result": "resolved"},
    {"issue": "legend_occlusion", "fix": "loc='upper left'", "result": "resolved"}
  ]
}
```

### Example Walkthrough

**Input:** LLM-generated code for "Bar chart of monthly sales, Jan-Dec"

```python
import matplotlib.pyplot as plt
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
sales = [45000, 52000, 48000, 61000, 55000, 67000,
         72000, 69000, 58000, 63000, 71000, 84000]
plt.figure(figsize=(8, 4))
plt.bar(months, sales, color='blue')
plt.title('Monthly Sales 2025')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.legend(['Sales'])
plt.savefig('chart.png')
```

**Iteration 1** — Inspector finds 3 issues:
```
[HIGH]   label_overlap       10 of 12 x-axis labels collide
[MEDIUM] legend_unnecessary  legend has one entry, adds no info
[LOW]    y_format            raw numbers (45000), not formatted ($45K)
Score: 0.30
```
Patch: `plt.xticks(rotation=45, ha='right')` + `plt.tight_layout()` → Score: 0.72

**Iteration 2** — Patch: remove `plt.legend(['Sales'])` → Score: 0.88

**Iteration 3** — Patch: add `FuncFormatter(lambda x, p: f'${x/1000:.0f}K')` → Score: 1.00

**Converged in 3 iterations.** Output: fixed code + convergence GIF.

### Dashboard Compliance Agent

Same architecture as per-chart compliance, but for multi-chart layouts.

| Check | Method |
|-------|--------|
| Chart overlap | Bounding box collision between chart containers |
| Spacing consistency | Measure gaps, flag inconsistent margins |
| Alignment | Charts should align to grid; detect misalignment |
| Hierarchy clarity | Primary chart should be visually dominant (size, position) |
| Color collision | Different charts using same color for different meanings |
| Style drift | Font sizes, title styles inconsistent across charts |

Convergence loop: Render dashboard → Inspect layout → Patch (resize/reposition) → Re-render.

---

## AutoDash Pipeline

The pipeline wraps plotlint with data intelligence and chart planning stages. These stages are plumbing — they need to work, but the novel engineering is in plotlint.

### Stage 1: Data Intelligence
Load data (CSV/Excel/Parquet), profile it (types, distributions, nulls, cardinality), produce a schema summary. Standard pandas profiling.

### Stage 2: Analysis Planning
LLM receives data profile + user questions. Outputs an analysis plan: what to compute, which columns, what aggregations.

### Stage 3: Data Exploration
Agent loop: LLM generates pandas code → execute in sandbox → review results → iterate if needed. Output: aggregated dataframes and insights.

### Stage 4: Chart Planning
Given analysis results + original questions, determine: which insights need visualization, what chart type for each, visual hierarchy (primary vs supporting), how many charts (3-6 typical).

Output: List of chart specifications:
```python
ChartSpec = {
    "type": "bar",
    "data": {"source": "aggregated_df", "x": "region", "y": "revenue", "color": "quarter"},
    "title": "Q4 Revenue by Region",
    "priority": "primary",  # primary, secondary, supporting
}
```

### Stage 5-6: plotlint + Dashboard Compliance
Each chart generated and polished by plotlint. Then composed into a dashboard with layout intelligence and checked by the Dashboard Compliance Agent.

### Stage 7: Output Generation
Static (PNG/PDF via matplotlib), Interactive (HTML+JS via plotly), Structured (JSON for embedding).

### Human-in-the-Loop Checkpoints

The pipeline is not fully autonomous. Users can intervene at key decision points:

1. **After analysis plan:** "Also look at customer segments" / "Looks good, proceed"
2. **After data exploration:** "Dig deeper into enterprise" / "Move on to charts"
3. **After chart planning:** "Skip the pie, add a heatmap instead"
4. **After dashboard draft:** "Make the trend chart bigger" / "Approved"

Each checkpoint is optional (user can auto-approve all). plotlint's visual compliance runs without human intervention — that's mechanical.

---

## Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python | matplotlib is Python; single ecosystem |
| Orchestration | LangGraph | Both loops are genuine state machines (convergence: rollback, branching, score tracking; pipeline: HITL checkpoints, persistence). Built-in `interrupt()` + checkpointing for human-in-the-loop. |
| Rendering (matplotlib) | Agg backend | Direct access to figure/artist objects for element extraction |
| Rendering (browser) | Playwright | Headless browser, can inject JS for element positions |
| Inspector | Custom Python module | Bounding box extraction, collision detection, contrast checks |
| Critic | Claude API (vision) | Only invoked selectively; not every iteration |
| Patcher | Claude API (text) | Given issue list + code, generates surgical fix |
| Sandboxing | subprocess / Docker | User code runs in isolation |
| Chart specs | Renderer-agnostic (dataclasses → JSON) | Type-safe in Python, portable for transport |

### What This Omits (and Why)

| Omitted | Why |
|---------|-----|
| LangChain | LangGraph is used for orchestration; LangChain's abstraction layer is not needed |
| Vector database | No retrieval needed; all context fits in prompt |
| Fine-tuned models | General-purpose vision + code models are sufficient |
| Frontend framework | CLI-first; web UI is a future concern |

---

## Demo & Proof

### Demo Artifacts

1. **Convergence GIF** (hero demo): 4 frames — broken chart → fix 1 → fix 2 → clean chart. Each frame overlaid with semi-transparent bounding boxes showing what the Inspector detected (red for overlaps, green for passing). The viewer immediately sees the Inspector's spatial reasoning.

2. **Before/after gallery**: 8-10 common defects, each showing the broken chart, the Inspector's findings, and the fixed result.

3. **Full pipeline demo**: Terminal recording of `autodash sales.csv "What drove Q4?"` producing a complete dashboard.

### Benchmark Suite

30-50 matplotlib/plotly scripts with known visual defects, categorized by type:
- Label overlap (various: rotation needed, too many categories, long text)
- Legend problems (occlusion, unnecessary single-entry, poor placement)
- Element cutoff (title, labels, annotations extending beyond figure)
- Color issues (low contrast, colorblind-unsafe palettes, WCAG failures)
- Chart type misuse (pie with too many slices, line for categorical data)

Used for: unit tests (Inspector detection accuracy), integration tests (full loop convergence), quantitative claims ("plotlint detects N/50 defects, fixes M/50 autonomously").

### Success Metrics

| Metric | Target |
|--------|--------|
| Convergence rate | >80% of benchmark defects fixed |
| Iterations to converge | <5 average |
| Inspector false positive rate | <10% |
| Regression rate (patch breaks prior fix) | <5% |
| Time per iteration | <10s |

---

## Competitive Landscape

### What Exists

| Tool | Strengths | Gap plotlint Fills |
|------|-----------|-------------------|
| **Tableau / Power BI** | Powerful, polished, enterprise | Manual: user must know what to build. No automation. |
| **ChatGPT Code Interpreter** | Conversational chart generation | Single charts, no visual QA, no dashboard composition |
| **Julius.ai** | Chat-to-chart, nice UX | One chart at a time, no layout intelligence |
| **Percy / Chromatic** | Visual regression testing | Reports to humans, doesn't fix anything |
| **matplotlib tight_layout()** | Basic overlap prevention | Dumb heuristic, fails on complex layouts |

### What Will Be Built (the obvious approach)

Agentic loops that screenshot a chart, send it to a vision model, get feedback, and regenerate. This is the natural next step for every AI coding tool. It will work *okay* but hit the precision/cost/speed ceiling described in the Problem section.

### What plotlint Does Differently

- **Programmatic element extraction** from the rendering engine's internals — not screenshots
- **Pixel-precise geometric measurement** — not vision model approximation
- **Deterministic, free, instant** detection for 80% of defects
- **Convergence loop** with score tracking, rollback, and regression detection
- **Automated multi-chart dashboard composition** with visual compliance

| Capability | Tableau | ChatGPT | Julius | **AutoDash** |
|------------|---------|---------|--------|---------------|
| Question → Insight | No | Partial | Partial | Yes |
| Multi-chart Dashboard | Yes (manual) | No | No | Yes (auto) |
| Visual Compliance | No | No | No | Yes (plotlint) |
| Layout Intelligence | Yes (manual) | No | No | Yes (auto) |
| Standalone Components | No | No | No | Yes |

**Positioning:** AutoDash is not competing with Tableau. Tableau is for analysts who know what they want. AutoDash is for people who have data and questions. plotlint is for anyone generating charts programmatically who wants them to not look broken.

---

## Target Users & Integration

### plotlint (standalone)

| User | Pain Point | Entry Point |
|------|-----------|-------------|
| Data scientists | LLM-generated chart code has visual bugs | `plotlint script.py` |
| Report pipelines | Programmatic charts never get visual review | `plotlint inspect --json` as CI/CD gate |
| LLM tool builders | Chart generation outputs are visually broken | plotlint as post-processing step |
| Jupyter users | Screenshot → paste → fix → repeat cycle | `%%plotlint` cell magic (future) |

### AutoDash (full pipeline)

| User | Pain Point | Entry Point |
|------|-----------|-------------|
| Analysts with unfamiliar data | Don't know where to start exploring | `autodash data.csv "questions"` |
| Teams generating recurring reports | Manual dashboard assembly is tedious | AutoDash in report generation pipeline |

### Integration Points

Designed from the start, implemented in phases:

- `plotlint script.py` — CLI (primary interface)
- `plotlint inspect script.py --format json` — inspection only, no LLM, no fixing. For CI/CD.
- `from plotlint import fix_chart, inspect_chart` — Python API
- `%%plotlint` — Jupyter cell magic (future)
- `autodash data.csv "questions"` — full pipeline CLI

---

## Technical Challenges

| Challenge | Mitigation |
|-----------|------------|
| matplotlib artist tree is under-documented | Reverse-engineer from source; tree structure is stable across versions |
| Bounding boxes change with DPI/backend | Standardize: always render at fixed DPI (100) with Agg backend |
| LLM patches break unrelated code | One issue per iteration; full re-render catches regressions |
| Convergence thrashing | Score history + rollback + fix deduplication |
| plotly SVG structure varies by chart type | Build extractor per chart type; start with bar/line/scatter |
| Sandboxing user code | subprocess with timeout; Docker for production |
| Vision API cost per iteration | Inspector (free) handles 80% of issues; Critic (paid) only for semantic checks |
