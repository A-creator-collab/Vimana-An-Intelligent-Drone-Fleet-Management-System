"""data_generator.py - Every piece of data in the project is synthetic and seeded.

City graph, fleet, task stream, no-fly events, ML training sets (maintenance,
telemetry anomalies, hourly demand). Fixed seeds -> fully reproducible runs.
"""
import math
import random
from typing import Dict, List, Tuple

import numpy as np

from dsa.pathfinding import Graph
from dsa.union_find import UnionFind
from models import Drone, NoFlyZone, Node, Task

SEED = 42
ZONES = ["A", "B", "C", "D"]          # A=NW  B=NE  C=SW  D=SE


def set_seeds(seed: int = SEED) -> None:
    """Seed Python's and NumPy's global RNGs (local generators are seeded too)."""
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------------------------------------------ city
def make_city(rows: int = 10, cols: int = 10, spacing: float = 10.0,
              keep_extra: float = 0.85, n_hubs: int = 5, seed: int = SEED) -> Tuple[Graph, List[int]]:
    """Grid city. A random spanning tree (built with Union-Find) guarantees
    connectivity; every other grid edge survives with prob `keep_extra`.
    Edge weight = length * (1 + wind/risk), risk ~ U(0, 0.4)."""
    rng = random.Random(seed)
    corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1, (rows // 2) * cols + cols // 2]
    extra = [i for i in range(rows * cols) if i not in corners]
    rng.shuffle(extra)
    hubs = set((corners + extra)[:n_hubs])
    g = Graph()
    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            g.add_node(Node(nid, c * spacing, r * spacing, nid in hubs))
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols: edges.append((u, u + 1))
            if r + 1 < rows: edges.append((u, u + cols))
    rng.shuffle(edges)
    uf = UnionFind(rows * cols)
    for u, v in edges:
        merged = uf.union(u, v)
        if merged or rng.random() < keep_extra:
            g.add_edge(u, v, risk=rng.uniform(0.0, 0.4))
    assert uf.components == 1
    return g, sorted(hubs)


def city_extent(g: Graph) -> Tuple[float, float]:
    return max(n.x for n in g.nodes.values()), max(n.y for n in g.nodes.values())


def sector_of(x: float, y: float, extent: Tuple[float, float]) -> str:
    """A=NW, B=NE, C=SW, D=SE (y grows northwards)."""
    east, north = x >= extent[0] / 2, y >= extent[1] / 2
    return {(False, True): "A", (True, True): "B", (False, False): "C", (True, False): "D"}[(east, north)]


def sector_nodes(g: Graph, sector: str) -> List[int]:
    ext = city_extent(g)
    return sorted(n.id for n in g.nodes.values() if sector_of(n.x, n.y, ext) == sector)


def sector_center(g: Graph, sector: str) -> Tuple[float, float]:
    ext = city_extent(g)
    cx = {"A": .25, "C": .25, "B": .75, "D": .75}[sector] * ext[0]
    cy = {"A": .75, "B": .75, "C": .25, "D": .25}[sector] * ext[1]
    return cx, cy


# ------------------------------------------------------------- fleet/tasks
def sample_maintenance_features(n: int, rng: np.random.Generator) -> np.ndarray:
    """Columns: flight_hours, battery_cycles, vibration(g rms), motor_temp(C)."""
    hours = rng.uniform(0, 500, n)
    cycles = rng.uniform(0, 800, n)
    vib = np.clip(rng.normal(0.6, 0.3, n), 0.1, 1.6)
    temp = rng.normal(65, 10, n)
    return np.column_stack([hours, cycles, vib, temp])


def make_fleet(n: int, g: Graph, hubs: List[int], seed: int = SEED) -> List[Drone]:
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    feats = sample_maintenance_features(n, nrng)
    drones = []
    for i in range(n):
        node = rng.choice(hubs) if rng.random() < 0.5 else rng.randrange(g.n)
        nd = g.nodes[node]
        drones.append(Drone(
            id=i, node=node, x=nd.x, y=nd.y, last_node=node,
            battery=rng.uniform(60, 100), speed=rng.choice([4.0, 5.0, 6.0]),
            capacity=rng.choice([3.0, 5.0, 8.0]),
            flight_hours=float(feats[i, 0]), battery_cycles=int(feats[i, 1]),
            vibration=float(feats[i, 2]), motor_temp=float(feats[i, 3])))
    return drones


def make_task_stream(n: int, horizon: int, g: Graph, seed: int = SEED) -> List[Task]:
    """Tasks arrive over [0, 0.85*horizon]; priorities skew low; pickup/drop >= 25 units apart."""
    rng = random.Random(seed)
    times = sorted(rng.uniform(0, 0.85 * horizon) for _ in range(n))
    tasks = []
    for i, t in enumerate(times):
        while True:
            p, d = rng.randrange(g.n), rng.randrange(g.n)
            if g.euclid(p, d) >= 25:
                break
        created = int(t)
        tasks.append(Task(id=i, pickup=p, dropoff=d,
                          priority=rng.choices([1, 2, 3, 4, 5], [.3, .25, .2, .15, .1])[0],
                          weight=round(rng.uniform(0.5, 4.5), 1), created=created,
                          deadline=created + rng.randint(45, 90)))
    return tasks


def make_no_fly_events(k: int, horizon: int, g: Graph, seed: int = SEED) -> List[NoFlyZone]:
    rng = random.Random(seed)
    ex, ey = city_extent(g)
    zones = []
    for i in range(k):
        start = rng.randint(10, max(11, horizon - 50))
        zones.append(NoFlyZone(i, rng.uniform(15, ex - 15), rng.uniform(15, ey - 15),
                               rng.uniform(9, 14), start, start + rng.randint(15, 30)))
    return zones


# ------------------------------------------------------------- ML datasets
def make_maintenance_dataset(n: int = 3000, seed: int = SEED):
    """Synthetic flight logs. Label 'needs maintenance' = top-25% of a noisy wear score."""
    rng = np.random.default_rng(seed)
    X = sample_maintenance_features(n, rng)
    z = 0.006 * X[:, 0] + 0.004 * X[:, 1] + 2.2 * X[:, 2] + 0.045 * (X[:, 3] - 60) + rng.normal(0, 0.4, n)
    y = (z > np.quantile(z, 0.75)).astype(int)
    return X, y


def make_telemetry_stream(n: int = 2000, anomaly_rate: float = 0.04, seed: int = SEED) -> Dict[str, np.ndarray]:
    """Per-tick telemetry: battery drain (%/tick), off-course error (m), motor temp (C).
    Anomalies perturb a random non-empty subset of the three channels."""
    rng = np.random.default_rng(seed)
    drain = rng.normal(0.8, 0.1, n)
    off = np.abs(rng.normal(0.0, 0.6, n))
    temp = rng.normal(60, 4, n)
    label = (rng.random(n) < anomaly_rate).astype(int)
    label_drain = np.zeros(n, dtype=int)
    for i in np.where(label)[0]:
        mask = rng.random(3) < 0.6
        if not mask.any():
            mask[rng.integers(3)] = True
        if mask[0]: drain[i] = rng.uniform(1.5, 2.5); label_drain[i] = 1
        if mask[1]: off[i] = rng.uniform(2.5, 5.0)
        if mask[2]: temp[i] = rng.normal(85, 4)
    return {"drain": drain, "off_course": off, "temp": temp, "label": label, "label_drain": label_drain}


def make_demand_history(days: int = 21, seed: int = SEED) -> np.ndarray:
    """Hourly delivery demand per zone, shape (4, days*24): daily sinusoid + zone base + Poisson noise."""
    rng = np.random.default_rng(seed)
    base = np.array([6.0, 4.0, 3.0, 5.0])
    phase = np.array([8, 12, 17, 10])          # each zone peaks at a different hour
    hrs = np.arange(days * 24)
    rate = base[:, None] * (1 + 0.7 * np.sin(2 * np.pi * (hrs[None, :] - phase[:, None]) / 24 + np.pi / 2))
    return rng.poisson(np.clip(rate, 0.3, None)).astype(float)
