"""dashboard/plots.py - matplotlib figures (static dashboard). Saved to output/ and shown in Colab."""
import os
import sys

import matplotlib
if "google.colab" not in sys.modules and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")           # headless when run as a plain script
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = "output"


def _finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=110)
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)
    return f"{OUT}/{name}.png"


def plot_city(sim, name="city_map"):
    """City graph, hubs, no-fly zones and the flight trails of the first drones."""
    g = sim.g
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for u, nb in g.adj.items():
        for v in nb:
            if u < v:
                a, b = g.nodes[u], g.nodes[v]
                ax.plot([a.x, b.x], [a.y, b.y], color="#cccccc", lw=0.8, zorder=1)
    ax.scatter([n.x for n in g.nodes.values() if not n.is_hub], [n.y for n in g.nodes.values() if not n.is_hub], s=6, c="#888888", zorder=2)
    ax.scatter([g.nodes[h].x for h in sim.hubs], [g.nodes[h].y for h in sim.hubs], s=110, marker="s", c="tab:green", zorder=3, label="hub / charger")
    for z in sim.zones:
        ax.add_patch(Circle((z.cx, z.cy), z.radius, color="red", alpha=0.18, zorder=1))
    for did, tr in sim.trails.items():
        xs, ys = zip(*tr)
        ax.plot(xs, ys, lw=1.4, alpha=0.8, zorder=4, label=f"drone {did}")
    ax.set_title(f"City graph, no-fly zones (red) and drone trails - {sim.strategy}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1)); ax.set_aspect("equal")
    return _finish(fig, name)


def plot_sim_compare(sims, name="sim_comparison"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for s in sims:
        axes[0].plot(s.hist["done"], label=s.strategy)
        axes[1].plot(s.hist["pending"], label=s.strategy)
        axes[2].plot(s.hist["avg_battery"], label=s.strategy)
    for ax, t in zip(axes, ["Cumulative tasks completed", "Tasks waiting in queue", "Fleet avg battery (%)"]):
        ax.set_title(t); ax.set_xlabel("tick"); ax.legend(); ax.grid(alpha=.3)
    return _finish(fig, name)


def plot_benchmarks(res, name="benchmarks"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    p = res["pathfinding"]
    ax[0, 0].plot([r["nodes"] for r in p], [r["dij_expanded"] for r in p], "o-", label="Dijkstra")
    ax[0, 0].plot([r["nodes"] for r in p], [r["astar_expanded"] for r in p], "s-", label="A*")
    ax[0, 0].set(title="Nodes expanded per query", xlabel="graph size (nodes)")
    a = res["assignment"]
    ax[0, 1].loglog([r["n"] for r in a], [r["greedy_ms"] for r in a], "o-", label="Greedy")
    ax[0, 1].loglog([r["n"] for r in a], [r["hung_ms"] for r in a], "s-", label="Hungarian")
    ax[0, 1].set(title="Assignment time (ms)", xlabel="drones = tasks")
    t = res["tsp"]
    ax[1, 0].semilogy([r["stops"] for r in t], [r["dp_ms"] for r in t], "o-", label="Held-Karp DP")
    ax[1, 0].semilogy([r["stops"] for r in t], [r["nn_ms"] for r in t], "s-", label="Nearest-neighbour")
    ax[1, 0].semilogy([r["stops"] for r in t if r["brute_ms"] == r["brute_ms"]], [r["brute_ms"] for r in t if r["brute_ms"] == r["brute_ms"]], "^-", label="Brute force")
    ax[1, 0].set(title="Multi-stop routing time (ms)", xlabel="stops")
    s = res["spatial"]
    ax[1, 1].loglog([r["drones"] for r in s], [r["kd_query_us"] for r in s], "o-", label="KD-tree")
    ax[1, 1].loglog([r["drones"] for r in s], [r["brute_query_us"] for r in s], "s-", label="Brute force")
    ax[1, 1].set(title="Nearest-drone query (us)", xlabel="drones")
    for x in ax.flat:
        x.legend(); x.grid(alpha=.3)
    return _finish(fig, name)


def plot_ml(maint_imp, forecast, tel, iso_flags, name="ml_results"):
    import numpy as np
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].barh(list(maint_imp), list(maint_imp.values()), color="tab:blue")
    ax[0].set_title("Predictive maintenance: RF feature importance")
    ax[1].plot(forecast["test_true"][:96], label="actual"); ax[1].plot(forecast["test_pred"][:96], label="LinReg forecast")
    ax[1].set_title("Demand forecast (zone A, last 4 days)"); ax[1].set_xlabel("hour"); ax[1].legend()
    n = len(tel["drain"]); idx = np.arange(n)
    ax[2].scatter(idx, tel["drain"], s=4, c="#999999", label="normal")
    ax[2].scatter(idx[iso_flags == 1], tel["drain"][iso_flags == 1], s=14, c="red", label="IsolationForest flag")
    ax[2].set_title("Telemetry: battery drain anomalies"); ax[2].set_xlabel("tick"); ax[2].legend()
    return _finish(fig, name)
