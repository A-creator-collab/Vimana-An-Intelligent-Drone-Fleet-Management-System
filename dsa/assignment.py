"""assignment.py - Greedy and Hungarian (Kuhn-Munkres, O(n^2 m)) assignment."""
from typing import List, Optional, Sequence, Tuple

BIG = 1e9   # cost used for infeasible pairs


def greedy_assign(cost: Sequence[Sequence[float]],
                  order: Optional[Sequence[int]] = None) -> Tuple[List[Tuple[int, int]], float]:
    """Each row (task), in `order`, grabs the cheapest still-free column (drone).
    O(n*m). Fast but can be globally sub-optimal. Requires rows <= cols."""
    n = len(cost)
    m = len(cost[0]) if n else 0
    used = [False] * m
    pairs, total = [], 0.0
    for r in (order if order is not None else range(n)):
        best, bc = -1, float("inf")
        row = cost[r]
        for c in range(m):
            if not used[c] and row[c] < bc:
                best, bc = c, row[c]
        if best >= 0:
            used[best] = True
            pairs.append((r, best))
            total += bc
    return pairs, total


def hungarian(cost: Sequence[Sequence[float]]) -> Tuple[List[Tuple[int, int]], float]:
    """Optimal min-cost assignment via the potentials (shortest augmenting path)
    formulation of the Hungarian algorithm. Works on rectangular matrices
    (every row of the smaller side is matched). Returns ([(row, col)], total)."""
    n, m = len(cost), len(cost[0])
    flip = n > m
    if flip:
        cost = [[cost[i][j] for i in range(n)] for j in range(m)]
        n, m = m, n
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)       # p[j] = row matched to column j
    way = [0] * (m + 1)
    inf = float("inf")
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            row = cost[i0 - 1]
            ui0 = u[i0]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui0 - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:                       # augment along the alternating path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs, total = [], 0.0
    for j in range(1, m + 1):
        if p[j]:
            r, c = p[j] - 1, j - 1
            pairs.append((c, r) if flip else (r, c))
            total += cost[r][c]
    return pairs, total
