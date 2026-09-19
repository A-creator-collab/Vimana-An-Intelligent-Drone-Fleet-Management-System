"""union_find.py - Disjoint Set Union with path compression + union by rank."""
from collections import defaultdict
from typing import Dict, List


class UnionFind:
    """Near-O(1) amortised find/union. Used for (a) guaranteeing city-graph
    connectivity, (b) clustering drones into swarm groups, (c) merging
    overlapping no-fly zones into connected regions."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:                 # iterative path compression
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge sets; returns True if they were previously separate."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> List[List[int]]:
        g: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return list(g.values())
