"""Correctness tests. Run with `pytest -q` or `python -m tests.test_core`."""
import itertools
import math
import random

from data_generator import make_city, sector_nodes
from dsa.assignment import hungarian
from dsa.dp_routing import brute_force_tsp, constrained_route, held_karp
from dsa.pathfinding import astar, dijkstra, floyd_warshall, fw_path
from dsa.scheduling import AVLTree, ReservationTable, TaskQueue
from dsa.spatial_index import KDTree
from dsa.union_find import UnionFind
from ai.nlp_command_parser import NLPCommandParser
from models import Task


def test_astar_dijkstra_floyd_agree():
    g, _ = make_city(8, 8, seed=3)
    apsp, nxt = floyd_warshall(g)
    rng = random.Random(1)
    for _ in range(30):
        a, b = rng.randrange(g.n), rng.randrange(g.n)
        _, c1, _ = dijkstra(g, a, b); p2, c2, _ = astar(g, a, b)
        assert math.isclose(c1, c2) and math.isclose(c1, apsp[a][b])
        assert fw_path(nxt, a, b)[0] == a and p2[-1] == b


def test_blocked_nodes_respected():
    g, _ = make_city(8, 8, seed=3)
    p, _, _ = astar(g, 0, 63)
    p2, _, _ = astar(g, 0, 63, frozenset(p[2:4]))
    assert p2 is None or not set(p[2:4]) & set(p2)


def test_hungarian_is_optimal():
    rng = random.Random(0)
    for n, m in [(4, 4), (5, 5), (3, 6), (6, 3)]:
        c = [[rng.randint(1, 30) for _ in range(m)] for _ in range(n)]
        _, tot = hungarian(c)
        if n <= m:
            best = min(sum(c[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        else:
            best = min(sum(c[p[j]][j] for j in range(m)) for p in itertools.permutations(range(n), m))
        assert tot == best


def test_held_karp_matches_brute_force_and_constraints():
    rng = random.Random(2)
    pts = [(rng.random() * 50, rng.random() * 50) for _ in range(7)]
    d = [[math.dist(a, b) for b in pts] for a in pts]
    assert math.isclose(held_karp(d)[0], brute_force_tsp(d)[0])
    assert math.isclose(held_karp(d, True)[0], brute_force_tsp(d, True)[0])
    best = held_karp(d)[0]
    assert constrained_route(d, best * 1.01)[1] is not None
    assert constrained_route(d, best * 0.5)[1] is None


def test_kdtree_matches_brute_force():
    rng = random.Random(4)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100), i) for i in range(300)]
    kd = KDTree(pts)
    for _ in range(50):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        want = sorted(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)[:3]
        assert [i for _, i in kd.nearest(x, y, 3)] == [p[2] for p in want]
        assert sorted(kd.range_query(x, y, 15)) == sorted(p[2] for p in pts if math.hypot(p[0] - x, p[1] - y) <= 15)


def test_union_find():
    uf = UnionFind(5)
    assert uf.union(0, 1) and not uf.union(1, 0) and uf.union(3, 4)
    assert uf.connected(0, 1) and not uf.connected(1, 3) and uf.components == 3


def test_avl_stays_balanced_and_sorted():
    t = AVLTree(); keys = list(range(500)); random.Random(1).shuffle(keys)
    for k in keys: t.insert(k)
    assert t.height() <= 1.45 * math.log2(501) + 2
    for k in keys[:250]: t.delete(k)
    assert [k for k, _ in t.inorder()] == sorted(keys[250:]) and len(t) == 250


def test_task_queue_order_and_reservations():
    q = TaskQueue()
    q.push(Task(1, 0, 1, 2, 1, 0, 50)); q.push(Task(2, 0, 1, 5, 1, 0, 90)); q.push(Task(3, 0, 1, 5, 1, 0, 60))
    assert [q.pop().id for _ in range(3)] == [3, 2, 1]
    r = ReservationTable(capacity=1); r.reserve("e", 5, 9)
    assert not r.is_free("e", 8, 10) and r.is_free("e", 10, 12)


def test_city_is_connected_and_sectors_partition():
    g, _ = make_city()
    assert dijkstra(g, 0, g.n - 1)[0] is not None
    assert sum(len(sector_nodes(g, s)) for s in "ABCD") == g.n


def test_nlp():
    p = NLPCommandParser()
    c = p.parse("send 2 drones to sector B and return when battery hits 20%")
    assert (c.action, c.params) == ("dispatch", {"count": 2, "sector": "B", "return_battery": 20})
    assert p.parse("deliver from node 3 to sector C priority 2").params["to"] == ("sector", "C")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
