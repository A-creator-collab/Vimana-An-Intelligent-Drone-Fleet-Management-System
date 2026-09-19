# Vimana — Intelligent Drone Fleet Management System

> *Vimana (Sanskrit: विमान) — "flying palace." Ancient word for flight, modern answer for fleets.*

A simulation-only drone fleet management platform that unifies **classical data structures**, **machine learning**, and **rule-based safety** into a single end-to-end pipeline. Pure Python · NumPy · scikit-learn · matplotlib. Runs on CPU in ~15 seconds.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules in Detail](#modules-in-detail)
- [Algorithms & Complexity](#algorithms--complexity)
- [Machine Learning Models](#machine-learning-models)
- [Safety-First Design](#safety-first-design)
- [Outputs](#outputs)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

**Vimana** simulates an autonomous drone delivery fleet operating over a synthetic 100-node city grid. Each run:

1. Builds a connected city graph with charging hubs and time-windowed no-fly zones.
2. Generates a fleet of 24 drones and a stream of 110 delivery tasks.
3. Trains ML models to ground at-risk drones, detect telemetry anomalies, and forecast demand.
4. Executes a 260-tick simulation where every dispatch decision is validated by a rule-based safety layer.
5. Produces four dashboards and a consolidated Markdown report in `output/`.

Everything is **synthetic, seeded, and reproducible** — identical inputs give identical outputs.

---

## Features

- **DSA from scratch** — Dijkstra, A*, Floyd-Warshall, Hungarian, Held-Karp DP, branch-and-bound routing, AVL tree, KD-tree, Trie, Union-Find, heap task queue, Welsh-Powell colouring, merge sort, 3-way quicksort.
- **Machine learning** — RandomForest predictive maintenance, IsolationForest anomaly detection, LinearRegression demand forecasting.
- **NLP command parser** — Regex-based, no LLM, no API. Parses commands like *"send 2 drones to sector B and return when battery hits 20%"*.
- **Rule-based safety validator** — Every proposed action (ML, optimiser, NLP) is checked for payload, battery, corridors, no-fly zones, and airspace reservations before approval.
- **Full simulation engine** — Tick-based movement, dynamic re-planning around no-fly zones, charging housekeeping, proximity checks, and per-tick metrics.
- **Empirical benchmarks** — Every algorithm's Big-O claim is verified with real timing tables and log-log plots.
- **Consolidated report** — Each run writes `output/final_report.md` (with inline images) and `output/final_report.txt`.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │  data_generator.py      │  synthetic city, fleet, tasks
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
      ┌────────────┐      ┌────────────┐      ┌────────────┐
      │ dsa/       │      │ ai/        │      │ models.py  │
      │ algorithms │      │ ML + NLP   │      │ dataclasses│
      └─────┬──────┘      └─────┬──────┘      └──────┬─────┘
            │                   │                    │
            └───────────┬───────┴────────────────────┘
                        ▼
              ┌──────────────────────┐
              │ safety_validator.py  │  ← every proposal passes through here
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ simulation_engine.py │  ← tick-by-tick execution
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ dashboard/           │  plots.py + report.py
              └──────────────────────┘
```

**Key principle:** AI components and optimisers only **propose**. Nothing reaches the simulator until the deterministic `SafetyValidator` approves it.

---

## Project Structure

```
vimana/
├── main.py                    # end-to-end demo
├── benchmark.py               # empirical complexity benchmarks
├── data_generator.py          # seeded synthetic data (city, fleet, tasks, ML sets)
├── models.py                  # core dataclasses (Drone, Task, Node, NoFlyZone)
├── safety_validator.py        # rule-based approve/reject layer
├── simulation_engine.py       # tick-based fleet simulation
├── test_core.py               # correctness tests (pytest)
├── dsa/
│   ├── assignment.py          # greedy + Hungarian (Kuhn-Munkres)
│   ├── dp_routing.py          # Held-Karp, brute force, NN, branch-and-bound
│   ├── extras.py              # merge sort, quicksort, Trie, Welsh-Powell, sliding window
│   ├── pathfinding.py         # Graph, Dijkstra, A*, Floyd-Warshall
│   ├── scheduling.py          # heap TaskQueue, AVL tree, reservation table
│   ├── spatial_index.py       # 2-D KD-tree
│   └── union_find.py          # DSU with path compression
├── ai/
│   ├── anomaly_detection.py   # IsolationForest + streaming z-score
│   ├── demand_forecasting.py  # LinearRegression + drone allocation
│   ├── nlp_command_parser.py  # regex-based command parser
│   └── predictive_maintenance.py  # RandomForest maintenance classifier
└── dashboard/
    ├── plots.py               # 4 matplotlib dashboards
    └── report.py              # consolidated Markdown + text report
```

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install numpy scikit-learn matplotlib pytest
```

On Google Colab, all dependencies are pre-installed.

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This executes all nine sections (data → DSA → ML → NLP → simulation → fleet analysis → benchmarks → plots → report) and writes outputs to `output/`.

### View the report

```bash
cat output/final_report.txt          # plain text
# or open output/final_report.md in VS Code / browser for inline plots
```

### Run tests only

```bash
pytest -q
# or:
python -m test_core
```

### Run benchmarks only

```python
from benchmark import run_all
run_all()
```

---

## Modules in Detail

| Module | Purpose |
|---|---|
| `data_generator.py` | Builds the city graph (Union-Find spanning tree + extras), fleet, task stream, no-fly events, and three ML datasets — all seeded. |
| `models.py` | `Drone`, `Task`, `Node`, `NoFlyZone` dataclasses + battery/energy physics constants. |
| `safety_validator.py` | Approves or rejects missions and commands. Tracks a `Counter` of rejection reasons. |
| `simulation_engine.py` | The tick loop: admit tasks → update zones → run commands → dispatch → move → charge → proximity → record. |
| `benchmark.py` | Four benchmark suites: pathfinding, assignment, TSP, spatial index. |
| `dashboard/plots.py` | `plot_city`, `plot_sim_compare`, `plot_benchmarks`, `plot_ml`. |
| `dashboard/report.py` | `build_report(ctx)` + `write_report(ctx)` → Markdown + text. |

---

## Algorithms & Complexity

| Algorithm | File | Complexity |
|---|---|---|
| Dijkstra | `pathfinding.py` | O((V + E) log V) |
| A* (Euclidean heuristic) | `pathfinding.py` | O((V + E) log V), fewer expansions than Dijkstra |
| Floyd-Warshall | `pathfinding.py` | O(V³) |
| Hungarian (Kuhn-Munkres) | `assignment.py` | O(n² m) |
| Greedy assignment | `assignment.py` | O(n m) |
| Held-Karp DP | `dp_routing.py` | O(2ᵏ · k²) |
| Brute-force TSP | `dp_routing.py` | O(k!) |
| Nearest-neighbour TSP | `dp_routing.py` | O(k²) |
| Branch-and-bound routing | `dp_routing.py` | Worst-case exponential, pruned by budget + deadline |
| AVL tree | `scheduling.py` | O(log n) insert/delete/range query |
| Heap task queue | `scheduling.py` | O(log n) push/pop |
| KD-tree | `spatial_index.py` | O(n log² n) build, O(log n) average query |
| Union-Find | `union_find.py` | Near-O(1) amortised |
| Welsh-Powell colouring | `extras.py` | O(V² + E) |
| Merge sort | `extras.py` | O(n log n), stable |
| 3-way quicksort | `extras.py` | O(n log n) average |

All claims are validated in `benchmark.py` and visualised in `output/benchmarks.png`.

---

## Machine Learning Models

| Model | Task | Metric (synthetic) |
|---|---|---|
| RandomForest (60 trees, depth 8) | Predict maintenance need | Accuracy ~0.9, F1 ~0.8 |
| IsolationForest (100 estimators) | Multivariate telemetry anomalies | Precision/recall vs injected labels |
| Sliding-window z-score | Streaming anomaly detection | O(1) per sample |
| LinearRegression on lag features | Hourly demand per zone | MAE vs naive lag-1 / lag-24 |

Drones flagged with P(maintenance) > 0.85 are **grounded before the simulation starts** — a direct ML → safety → simulation pipeline.

---

## Safety-First Design

Every mission passes through `SafetyValidator.validate_mission`, which checks:

- **Payload** ≤ drone capacity
- **Corridors** exist between consecutive path nodes
- **No-fly zones** — no blocked node in the path
- **Battery reserve** ≥ 15% after pickup + dropoff + return to nearest hub
- **Airspace reservations** — time-windowed corridor slots

Every NLP command passes through `validate_command`, which checks drone counts, sector names, return-battery ranges, priorities, and payload limits. Rejections are logged with reasons and surfaced in `report()["rejection_breakdown"]`.

---

## Outputs

After `python main.py`, the `output/` folder contains:

```
output/
├── city_map.png           # graph, hubs, no-fly zones, drone trails
├── sim_comparison.png     # Hungarian vs Greedy: tasks, queue, battery
├── benchmarks.png         # 4 empirical Big-O plots
├── ml_results.png         # feature importances, forecast, anomalies
├── final_report.md        # complete report with inline images
└── final_report.txt       # plain-text copy
```

The Markdown report covers all nine sections: data, DSA showcase, ML results, NLP log, simulation comparison table, fleet analysis, benchmark tables, plot gallery, and file list.

---

## Testing

`test_core.py` contains 11 correctness tests:

- Dijkstra / A* / Floyd-Warshall agree on every random pair
- Blocked nodes are respected
- Hungarian matches brute-force optimum on rectangular matrices
- Held-Karp matches brute force (open + closed tours)
- Constrained routing finds / rejects solutions correctly
- KD-tree matches brute-force nearest and radius queries
- Union-Find merge semantics
- AVL tree stays balanced and sorted after 500 inserts and 250 deletes
- Task queue priority order + reservation table overlap
- City graph is connected and sectors partition all nodes
- NLP parser round-trips the flagship command

```bash
pytest -q
```

---

## Reproducibility

All randomness is seeded:

- `SEED = 42` in `data_generator.py`
- Local RNGs (`random.Random(seed)`, `np.random.default_rng(seed)`) inside every generator
- `set_seeds(42)` called at the top of `main()`

Two runs on the same machine produce byte-identical reports.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Numerics | NumPy |
| ML | scikit-learn |
| Visualisation | matplotlib |
| Testing | pytest |
| Reporting | Markdown + plain text |

No GPUs, no external APIs, no LLM services — runs end-to-end on a Colab CPU in ~15 seconds.

---

## License

MIT License — see `LICENSE` for details.

---

*Built as a demonstration of how classical algorithms, machine learning, and deterministic safety can be composed into a single coherent system.*
