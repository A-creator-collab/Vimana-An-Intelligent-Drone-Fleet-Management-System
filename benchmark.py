"""benchmark.py - Empirical complexity benchmarks (kept to ~15 s total on Colab CPU)."""
import itertools
import random
import time
from typing import Dict, List

from data_generator import make_city
from dsa.assignment import greedy_assign, hungarian
from dsa.dp_routing import brute_force_tsp, held_karp, nearest_neighbor_tsp
from dsa.pathfinding import astar, dijkstra
from dsa.spatial_index import KDTree


def _time(fn, *a):
    t = time.perf_counter(); out = fn(*a); return out, time.perf_counter() - t


def bench_pathfinding(sizes=(10, 20, 40, 60), queries: int = 40) -> List[Dict]:
    """Dijkstra vs A* on progressively larger grid cities: time, nodes expanded, path-cost equality."""
    rows = []
    for s in sizes:
        g, _ = make_city(s, s, seed=1)
        rng = random.Random(7)
        pairs = [(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(queries)]
        res = {}
        for name, fn in (("dijkstra", dijkstra), ("astar", astar)):
            t0 = time.perf_counter(); exp = 0; cost = 0.0
            for a, b in pairs:
                _, c, e = fn(g, a, b); exp += e; cost += c
            res[name] = (time.perf_counter() - t0, exp / queries, cost)
        assert abs(res["dijkstra"][2] - res["astar"][2]) < 1e-6, "A* must match Dijkstra's optimum"
        rows.append({"nodes": g.n, "dij_ms": 1000 * res["dijkstra"][0] / queries, "astar_ms": 1000 * res["astar"][0] / queries,
                     "dij_expanded": res["dijkstra"][1], "astar_expanded": res["astar"][1]})
    return rows


def bench_assignment(sizes=(10, 50, 100, 300, 1000)) -> List[Dict]:
    """Greedy vs Hungarian on random n x n cost matrices (n drones = n tasks)."""
    rng = random.Random(3)
    rows = []
    for n in sizes:
        cost = [[rng.uniform(1, 100) for _ in range(n)] for _ in range(n)]
        (gp, gc), gt = _time(greedy_assign, cost)
        (hp, hc), ht = _time(hungarian, cost)          # worst case O(n^3), typically far less
        rows.append({"n": n, "greedy_cost": gc, "hung_cost": hc, "greedy_ms": 1000 * gt, "hung_ms": 1000 * ht})
    return rows


def bench_tsp(ks=(5, 6, 7, 8, 10, 12)) -> List[Dict]:
    """Held-Karp DP vs brute force vs nearest-neighbour on random Euclidean instances."""
    rng = random.Random(5)
    rows = []
    for k in ks:
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(k)]
        d = [[((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** .5 for b in pts] for a in pts]
        (hc, _), ht = _time(held_karp, d)
        (nc, _), nt = _time(nearest_neighbor_tsp, d)
        if k <= 8:
            (bc, _), bt = _time(brute_force_tsp, d)
            assert abs(bc - hc) < 1e-9
        else:
            bt = float("nan")
        rows.append({"stops": k, "dp_ms": 1000 * ht, "brute_ms": 1000 * bt, "nn_ms": 1000 * nt,
                     "dp_cost": hc, "nn_cost": nc, "nn_gap_%": 100 * (nc / hc - 1)})
    return rows


def bench_spatial(sizes=(100, 1000, 5000), queries: int = 200) -> List[Dict]:
    """'Nearest drone' queries: brute-force O(n) scan vs KD-tree O(log n)."""
    rng = random.Random(9)
    rows = []
    for n in sizes:
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000), i) for i in range(n)]
        qs = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(queries)]
        tree, bt = _time(KDTree, pts)
        t0 = time.perf_counter()
        a = [tree.nearest(x, y)[0][1] for x, y in qs]
        kt = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = [min(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[2] for x, y in qs]
        brt = time.perf_counter() - t0
        assert a == b
        rows.append({"drones": n, "build_ms": 1000 * bt, "kd_query_us": 1e6 * kt / queries, "brute_query_us": 1e6 * brt / queries})
    return rows


def _table(title, rows):
    print(f"\n--- {title} ---")
    keys = list(rows[0])
    print("  ".join(f"{k:>14}" for k in keys))
    for r in rows:
        print("  ".join(f"{(f'{v:.2f}' if isinstance(v, float) else v):>14}" for v in r.values()))


def run_all(verbose: bool = True) -> Dict[str, List[Dict]]:
    out = {"pathfinding": bench_pathfinding(), "assignment": bench_assignment(),
           "tsp": bench_tsp(), "spatial": bench_spatial()}
    if verbose:
        for k, v in out.items():
            _table(k, v)
    return out


if __name__ == "__main__":
    run_all()
