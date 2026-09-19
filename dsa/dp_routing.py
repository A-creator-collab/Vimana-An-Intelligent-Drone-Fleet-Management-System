"""dp_routing.py - Multi-stop routing: Held-Karp DP, brute force, nearest-neighbour,
and a backtracking / branch-and-bound planner with battery + time-window constraints."""
import itertools
from typing import List, Optional, Sequence, Tuple

INF = float("inf")


def held_karp(d: Sequence[Sequence[float]], closed: bool = False) -> Tuple[float, List[int]]:
    """Exact TSP-style DP, O(2^k * k^2). Index 0 is the start (drone/hub).
    closed=False -> open path (drone need not come back); True -> round trip."""
    k = len(d)
    if k <= 1:
        return 0.0, [0]
    m = k - 1
    full = 1 << m
    dp = [[INF] * m for _ in range(full)]
    par = [[-1] * m for _ in range(full)]
    for j in range(m):
        dp[1 << j][j] = d[0][j + 1]
    for mask in range(1, full):
        row = dp[mask]
        for j in range(m):
            cur = row[j]
            if cur == INF or not (mask >> j) & 1:
                continue
            dj = d[j + 1]
            for nx in range(m):
                if (mask >> nx) & 1:
                    continue
                nm = mask | (1 << nx)
                c = cur + dj[nx + 1]
                if c < dp[nm][nx]:
                    dp[nm][nx] = c
                    par[nm][nx] = j
    best, bj = INF, -1
    for j in range(m):
        c = dp[full - 1][j] + (d[j + 1][0] if closed else 0.0)
        if c < best:
            best, bj = c, j
    order, mask, j = [], full - 1, bj
    while j != -1:
        order.append(j + 1)
        pj = par[mask][j]
        mask ^= 1 << j
        j = pj
    order.append(0)
    order.reverse()
    return best, order


def brute_force_tsp(d, closed: bool = False) -> Tuple[float, List[int]]:
    """O(k!) reference implementation (used to verify Held-Karp)."""
    k = len(d)
    best, bo = INF, []
    for perm in itertools.permutations(range(1, k)):
        c = d[0][perm[0]] if perm else 0.0
        for a, b in zip(perm, perm[1:]):
            c += d[a][b]
        if closed and perm:
            c += d[perm[-1]][0]
        if c < best:
            best, bo = c, [0] + list(perm)
    return best, bo


def nearest_neighbor_tsp(d, closed: bool = False) -> Tuple[float, List[int]]:
    """Greedy O(k^2) heuristic - fast but not optimal."""
    k = len(d)
    order, left, cost = [0], set(range(1, k)), 0.0
    while left:
        nxt = min(left, key=lambda j: d[order[-1]][j])
        cost += d[order[-1]][nxt]
        order.append(nxt)
        left.remove(nxt)
    if closed:
        cost += d[order[-1]][0]
    return cost, order


def constrained_route(d: Sequence[Sequence[float]], max_distance: float,
                      speed: float = 1.0,
                      deadlines: Optional[Sequence[float]] = None) -> Tuple[float, Optional[List[int]], int]:
    """Backtracking + branch-and-bound. Visit ALL stops 1..k-1 starting from 0 such that
    total distance <= max_distance (battery) and each stop i is reached by deadlines[i]
    (time window). Prunes on: budget, deadline, and current-best cost.
    Returns (best_cost, order|None, nodes_explored)."""
    k = len(d)
    best = [INF, None]
    explored = [0]

    def dfs(cur, visited, dist, order):
        explored[0] += 1
        if dist >= best[0]:
            return                                  # bound
        if len(order) == k:
            best[0], best[1] = dist, order[:]
            return
        for nx in range(1, k):
            if nx in visited:
                continue
            nd = dist + d[cur][nx]
            if nd > max_distance:                   # battery constraint
                continue
            if deadlines is not None and nd / speed > deadlines[nx]:   # time window
                continue
            visited.add(nx); order.append(nx)
            dfs(nx, visited, nd, order)
            visited.discard(nx); order.pop()

    dfs(0, {0}, 0.0, [0])
    return best[0], best[1], explored[0]
