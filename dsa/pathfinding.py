"""pathfinding.py - Weighted graph, Dijkstra, A*, Floyd-Warshall (all from scratch).

heapq is used only as a binary-heap primitive. Edge weight = length * (1 + risk),
so weight >= Euclidean length and the Euclidean heuristic is admissible AND
consistent -> A* is optimal with a closed set.
"""
import heapq
import math
from typing import Dict, FrozenSet, List, Optional, Tuple

from models import Node

INF = float("inf")


class Graph:
    """Undirected weighted graph stored as adjacency dicts (hash maps)."""

    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.adj: Dict[int, Dict[int, float]] = {}

    @property
    def n(self) -> int:
        return len(self.nodes)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        self.adj.setdefault(node.id, {})

    def euclid(self, u: int, v: int) -> float:
        a, b = self.nodes[u], self.nodes[v]
        return math.hypot(a.x - b.x, a.y - b.y)

    def add_edge(self, u: int, v: int, risk: float = 0.0) -> None:
        w = self.euclid(u, v) * (1.0 + risk)
        self.adj[u][v] = w
        self.adj[v][u] = w

    def path_length(self, path: List[int]) -> float:
        """Physical (Euclidean) length of a node path - what drains the battery."""
        return sum(self.euclid(a, b) for a, b in zip(path, path[1:]))


def _reconstruct(prev: Dict[int, int], src: int, dst: int) -> List[int]:
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def dijkstra(g: Graph, src: int, dst: int,
             blocked: FrozenSet[int] = frozenset()) -> Tuple[Optional[List[int]], float, int]:
    """Shortest path. Returns (path|None, cost, nodes_expanded). O((V+E) log V)."""
    if dst in blocked:
        return None, INF, 0
    dist = {src: 0.0}
    prev: Dict[int, int] = {}
    pq = [(0.0, src)]
    done = set()
    expanded = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        expanded += 1
        if u == dst:
            return _reconstruct(prev, src, dst), d, expanded
        for v, w in g.adj[u].items():
            if v in blocked or v in done:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None, INF, expanded


def astar(g: Graph, src: int, dst: int,
          blocked: FrozenSet[int] = frozenset()) -> Tuple[Optional[List[int]], float, int]:
    """A* with Euclidean heuristic. Same contract as dijkstra()."""
    if dst in blocked:
        return None, INF, 0
    gscore = {src: 0.0}
    prev: Dict[int, int] = {}
    pq = [(g.euclid(src, dst), src)]
    done = set()
    expanded = 0
    while pq:
        _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        expanded += 1
        if u == dst:
            return _reconstruct(prev, src, dst), gscore[u], expanded
        for v, w in g.adj[u].items():
            if v in blocked or v in done:
                continue
            ng = gscore[u] + w
            if ng < gscore.get(v, INF):
                gscore[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + g.euclid(v, dst), v))
    return None, INF, expanded


def floyd_warshall(g: Graph) -> Tuple[List[List[float]], List[List[int]]]:
    """All-pairs shortest paths, O(V^3). Node ids must be 0..n-1.
    Returns (dist, next_hop) matrices for O(1) cost lookup / path rebuild."""
    n = g.n
    dist = [[INF] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
        nxt[i][i] = i
    for u, nbrs in g.adj.items():
        for v, w in nbrs.items():
            dist[u][v] = w
            nxt[u][v] = v
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di, ni = dist[i], nxt[i]
            nik = ni[k]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
                    ni[j] = nik
    return dist, nxt


def fw_path(nxt: List[List[int]], u: int, v: int) -> List[int]:
    """Rebuild a path from the Floyd-Warshall next-hop matrix."""
    if nxt[u][v] == -1:
        return []
    path = [u]
    while u != v:
        u = nxt[u][v]
        path.append(u)
    return path
