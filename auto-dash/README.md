# AutoDash

### Keywords
*Extracted by GitHub Copilot*

- **Language:** `Python`
- **Architecture & Patterns:** `LangGraph StateGraph` · `TypedDict State Machines` · `Protocol-Based DI` · `Closure Factory (Conditional Edges)` · `Subprocess Sandbox` · `Frozen Dataclasses` · `Dual-Package Monorepo` · `Stub-to-Real Node Replacement` · `Layered Config Composition`
- **LLM & AI:** `Anthropic Claude API` · `Google Gemini API` · `LLM Code Generation` · `LLM Vision (Critic)` · `Prompt Engineering` · `Hybrid AI (Programmatic + LLM)` · `Convergence Loop`
- **Data & Visualization:** `pandas` · `matplotlib` · `Plotly` · `Bounding Box Extraction` · `Collision Detection` · `Spatial Analysis` · `Chart Code Generation` · `Dashboard Composition`
- **Visual Compliance:** `Programmatic Inspection` · `Defect Taxonomy` · `Label Overlap Detection` · `Element Cutoff Detection` · `WCAG Color Contrast` · `Convergence Scoring`
- **Pipeline:** `Data Profiling` · `Semantic Type Detection` · `Analysis Planning` · `Data Exploration` · `Chart Planning` · `Visual Compliance` · `Output Generation`

---

An end-to-end pipeline: data tables + natural language questions → polished, multi-chart dashboards. Its core engine, **plotlint**, closes the visual feedback loop through programmatic spatial analysis — extracting bounding boxes directly from the rendering engine and running collision detection — not screenshot guessing.

- [Project Vision](vision.md)
- [Architecture](architecture.md)

## Usage

```
$ autodash sales.csv "What drove Q4 revenue? How do regions compare?"
→ Loads & profiles data
→ Plans analysis, explores data, identifies insights
→ Generates charts, each polished by plotlint (visual compliance engine)
→ Outputs: dashboard.png
```

```
$ plotlint broken_chart.py
→ Renders chart, inspects for visual defects
→ Fixes: label overlap, legend occlusion, y-axis formatting
→ Converges in 3 iterations (score: 0.30 → 0.72 → 0.88 → 1.00)
→ Outputs: fixed code + convergence GIF
```

## Development Roadmap

- [Development Log](development_log.md)
- [Development Plan](development_plan.md)

| Version | Date | Focus |
|---------|------|-------|
| **MVP.1** | 2026-02-08 | Foundation + LangGraph scaffold |
| **MVP.2** | | Data intelligence (load, profile, type detection) |
| **MVP.3** | | Analysis planning (LLM: questions + profile → steps) |
| **MVP.4** | | Data exploration (LLM pandas code → sandbox execute) |
| **MVP.5** | | Chart planning + code generation |
| **MVP.6** | | Renderer (matplotlib Agg subprocess sandbox) |
| **MVP.7** | | Inspector (bbox extraction + label_overlap + element_cutoff) |
| **MVP.8** | | Patcher (LLM fix generation) |
| **MVP.9** | | Output (chart PNG) |
| **MVP.10** | | Docker packaging |

Post-MVP tracks run in parallel:
- **PL** (plotlint): Core engine depth — +7 defect types, Critic, CLI, Plotly, benchmarks
- **DI** (Dashboard Intelligence): Multi-chart, layout, style, dashboard compliance

## Purpose

- To practice and demonstrate familiarity with
  - AI agent orchestration (LangGraph)
  - Multi-step LLM pipelines
  - Multimodal AI (vision + text)
  - Programmatic visual analysis
- To explore the thesis that **deterministic measurement beats blind guessing** for chart quality

## Tech Stack

- **Pipeline Orchestration**: LangGraph (StateGraph, conditional edges)
- **LLM**: Anthropic Claude, Google Gemini (vendor-swappable via `LLMClient` protocol)
- **Data**: pandas
- **Visualization**: matplotlib, Plotly (future)
- **Testing**: pytest, pytest-asyncio

## Guiding Mantra

> **"Declarative, Modular, SoC"**

| Principle | Meaning |
|-----------|---------|
| **Declarative** | Describe *what*, not *how*. Config over code. Data-driven behavior. |
| **Modular** | Components are self-contained, swappable, and independently testable. plotlint works standalone or inside AutoDash. |
| **SoC** | Each module has ONE job. No god objects. Inspector detects. Patcher fixes. Loop orchestrates. |

## Architecture

Two packages, one monorepo. `plotlint` has ZERO imports from `autodash`.

| Package | Single Responsibility |
|---------|----------------------|
| `plotlint/core/` | Foundation utilities — LLM client, sandbox, parsing, errors, config |
| `plotlint/` | Visual compliance engine — convergence loop, models, renderer, inspector, patcher |
| `autodash/` | End-to-end pipeline — data loading, planning, exploration, chart gen, output |

```
plotlint (standalone)                    autodash (pipeline)
  Convergence Loop                         Pipeline Graph
  render → inspect → decide                load → plan → explore →
     ▲         ├── patch ──┘               chart → comply → output
     │         └── stop → END
     └──── loop back ◄────┘                comply invokes plotlint
                                           per chart
```

## Current State (MVP.1 Complete)

- LangGraph graph skeletons: convergence loop (render→inspect→patch) and pipeline (6-node linear)
- All model types defined upfront as shared contracts (MVP.2-9 types ready)
- Real `plotlint/core/` implementations: subprocess sandbox, LLM response parsing, error hierarchy, config
- `should_continue` closure with 4 stop conditions (perfect score, max iterations, render error, stagnation)
- 69 tests, all passing

## Known Limitations

- **All nodes are stubs**: Graph skeletons compile and route correctly, but nodes return `{}`. Real implementations come in MVP.2-9.
- **No rendering yet**: matplotlib bbox extraction (the core thesis) is validated in MVP.7.
- **Single chart only**: Multi-chart and dashboard layout are DI-1+.
- **No CLI**: `autodash` and `plotlint` CLI entry points come in later phases.
