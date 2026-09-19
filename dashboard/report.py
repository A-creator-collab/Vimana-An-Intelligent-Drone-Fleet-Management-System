"""dashboard/report.py - End-of-run report writer.

Collects every section of the pipeline (data, DSA showcase, ML, NLP,
simulation, fleet analysis, benchmarks, plots) into one Markdown report
(plus a plain-text copy) written to output/.
"""
import os
from datetime import datetime
from typing import Any, Dict, List

OUT = "output"


# --------------------------------------------------------------- helpers
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:                     # NaN
            return "n/a"
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _md_table(rows: List[Dict]) -> str:
    if not rows:
        return "_(no data)_\n"
    keys = list(rows[0])
    header = "| " + " | ".join(keys) + " |"
    sep = "|" + "|".join(["---"] * len(keys)) + "|"
    body = ["| " + " | ".join(_fmt(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join([header, sep] + body) + "\n"


def _section(title: str, body: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n\n{body.strip()}\n"


# --------------------------------------------------------------- builder
def build_report(ctx: Dict) -> str:
    parts: List[str] = []
    parts.append("# Intelligent Drone Fleet Management System - Final Report\n")
    parts.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    parts.append(
        f"_Runtime: **{ctx['runtime_s']:.1f} s** | Seed: `{ctx['seed']}` | "
        f"Drones: **{ctx['n_drones']}** | Tasks: **{ctx['n_tasks']}** | "
        f"Horizon: {ctx['horizon']} ticks | Simulation ticks: {ctx['ticks']} | "
        f"No-fly events: {ctx['n_no_fly']}_\n"
    )

    # ---------------------------------------------------------- 1. data
    parts.append(_section("1. Synthetic Data", (
        f"- City graph: **{ctx['city']['nodes']}** waypoints, "
        f"**{ctx['city']['corridors']}** corridors\n"
        f"- Charging hubs: `{ctx['city']['hubs']}`\n"
        f"- Fleet: {ctx['n_drones']} drones | Tasks: {ctx['n_tasks']} over {ctx['horizon']} ticks\n"
        f"- Floyd-Warshall all-pairs table: O(n^3) precomputed for O(1) dispatch cost lookups\n"
    )))

    # ---------------------------------------------------------- 2. DSA
    d = ctx["dsa"]
    parts.append(_section("2. DSA Showcase", (
        f"**Pathfinding (node 0 -> 99)**\n"
        f"- Dijkstra: cost `{_fmt(d['dijkstra_cost'])}`, expanded `{d['dijkstra_expanded']}` nodes\n"
        f"- A*: cost `{_fmt(d['astar_cost'])}`, expanded `{d['astar_expanded']}` nodes\n"
        f"- Floyd-Warshall: cost `{_fmt(d['fw_cost'])}` - all three agree: **{d['optimum_match']}**\n"
        f"- A* path: `{d['astar_path']}`\n"
        f"- A* re-route with 3 blocked nodes: `{d['astar_reroute']}`\n\n"
        f"**Spatial index**\n"
        f"- KD-tree 3-NN to (33, 47): `{d['kd_3nn']}`\n\n"
        f"**Assignment (toy 4x4)**\n"
        f"- Hungarian: `{d['hungarian_toy']}`\n"
        f"- Greedy: `{d['greedy_toy']}`\n\n"
        f"**Multi-stop routing (6 stops, start=0)**\n"
        f"- Held-Karp DP: cost `{_fmt(d['tsp_dp_cost'])}`, order `{d['tsp_order']}`\n"
        f"- Brute force: cost `{_fmt(d['tsp_brute_cost'])}`\n"
        f"- Nearest-neighbour: cost `{_fmt(d['tsp_nn_cost'])}`\n"
        f"- Constrained (battery + deadline): cost `{_fmt(d['constrained_cost'])}`, "
        f"order `{d['constrained_order']}`, explored `{d['constrained_explored']}` states\n\n"
        f"**Other structures**\n"
        f"- Heap task queue pop order (priority, deadline): `{d['task_queue']}`\n"
        f"- AVL tree: height {d['avl_height']} | keys >= 60% battery: `{d['avl_range']}`\n"
        f"- Reservation table: corridor free at t=12? `{d['resv_t12']}` | at t=15? `{d['resv_t15']}`\n"
        f"- Union-Find groups: `{d['uf_groups']}`\n"
        f"- Trie prefix 'd': `{d['trie_d']}`\n"
        f"- Merge sort by -x: `{d['merge_sort']}`\n"
    )))

    # ---------------------------------------------------------- 3. ML
    m = ctx["ml"]
    parts.append(_section("3. Machine Learning", (
        f"**Predictive maintenance (RandomForest, 3000 synthetic flights)**\n"
        f"- accuracy `{_fmt(m['maint_accuracy'])}` | F1 `{_fmt(m['maint_f1'])}`\n"
        f"- feature importances: `{m['maint_importances']}`\n"
        f"- drones with P(maintenance) > 0.85 -> grounded: `{m['grounded']}`\n\n"
        f"**Anomaly detection (2000 telemetry ticks)**\n"
        f"- IsolationForest: precision `{_fmt(m['iso_precision'])}` | recall `{_fmt(m['iso_recall'])}`\n"
        f"- Sliding-window z-score: precision `{_fmt(m['sw_precision'])}` | recall `{_fmt(m['sw_recall'])}`\n\n"
        f"**Demand forecast (LinearRegression on lags)**\n"
        f"- MAE `{_fmt(m['fc_mae'])}` | RMSE `{_fmt(m['fc_rmse'])}`\n"
        f"- naive lag-1 MAE `{_fmt(m['fc_naive1'])}` | naive lag-24 MAE `{_fmt(m['fc_naive24'])}`\n"
        f"- next-hour predicted demand per zone: `{m['fc_next']}`\n"
        f"- pre-positioned drones: `{m['fc_alloc']}`\n"
    )))

    # ---------------------------------------------------------- 4. NLP
    nlp_lines = "\n".join(f"- `{s}`\n    -> `{r}`" for s, r in ctx["nlp"])
    parts.append(_section("4. NLP Command Parser", nlp_lines))

    # ---------------------------------------------------------- 5. simulation
    sims = ctx["sims"]
    reports = [s.report() for s in sims]
    keys = ["tasks_total", "completed", "completion_rate_%", "late",
            "avg_turnaround_ticks", "avg_battery_reserve_%", "fleet_distance",
            "distance_per_task", "validator_rejections", "replans",
            "hold_ticks", "crashed", "proximity_events",
            "astar_nodes_expanded", "dispatch_time_s"]
    table = ["| metric | " + " | ".join(r["strategy"] for r in reports) + " |",
             "|" + "|".join(["---"] * (len(reports) + 1)) + "|"]
    for k in keys:
        table.append("| `" + k + "` | " + " | ".join(str(r[k]) for r in reports) + " |")
    cmd_log = "\n".join(f"  - {line}" for line in sims[0].command_log) or "  _(none)_"
    parts.append(_section("5. Simulation (Hungarian vs Greedy)", (
        "**Strategy comparison (identical scenario, seeded)**\n\n"
        + "\n".join(table) + "\n\n"
        f"**Rejection breakdown (Hungarian):** `{reports[0]['rejection_breakdown']}`\n\n"
        f"**NLP commands executed during the Hungarian run:**\n{cmd_log}\n"
    )))

    # ---------------------------------------------------------- 6. fleet
    a = ctx["fleet"]
    parts.append(_section("6. Fleet Analysis (end of Hungarian run)", (
        f"- Swarm clusters (Union-Find, radius 25): `{a['swarm_clusters']}`\n"
        f"- Radio channels needed (Welsh-Powell colouring): `{a['channels_used']}`\n"
        f"- Merged no-fly regions: `{a['no_fly_regions']}`\n"
        f"- Top drones (id, tasks done, battery): `{a['top_drones']}`\n"
    )))

    # ---------------------------------------------------------- 7. benchmarks
    b = ctx["benchmarks"]
    parts.append(_section("7. Benchmarks (Big-O verified empirically)", (
        "### Pathfinding - Dijkstra vs A*\n\n" + _md_table(b["pathfinding"]) +
        "\n### Assignment - Greedy vs Hungarian\n\n" + _md_table(b["assignment"]) +
        "\n### Multi-stop routing - Held-Karp vs brute force vs nearest-neighbour\n\n" + _md_table(b["tsp"]) +
        "\n### Spatial index - KD-tree vs brute force\n\n" + _md_table(b["spatial"])
    )))

    # ---------------------------------------------------------- 8. plots
    gallery = []
    for p in ctx["plot_files"]:
        base = os.path.basename(p)
        gallery.append(f"### {base}\n\n![{base}]({base})\n")
    parts.append(_section("8. Generated Plots", "\n".join(gallery)))

    # ---------------------------------------------------------- 9. files
    parts.append(_section("9. Output Files", "\n".join(
        [f"- `{p}`" for p in ctx["plot_files"]] +
        [f"- `{OUT}/final_report.md` (this file)",
         f"- `{OUT}/final_report.txt` (plain-text copy)"]
    )))

    return "\n".join(parts)


# --------------------------------------------------------------- writer
def write_report(ctx: Dict, name: str = "final_report") -> List[str]:
    """Write both Markdown and plain-text copies to output/. Returns the file paths."""
    os.makedirs(OUT, exist_ok=True)
    md = build_report(ctx)
    md_path = f"{OUT}/{name}.md"
    txt_path = f"{OUT}/{name}.txt"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    # crude markdown strip for the plain-text copy
    plain_lines = []
    for line in md.splitlines():
        s = line.replace("**", "").replace("`", "").rstrip("_").lstrip("_")
        if s.startswith("#"):
            s = s.lstrip("# ").upper()
        plain_lines.append(s)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(plain_lines))
    return [md_path, txt_path]
