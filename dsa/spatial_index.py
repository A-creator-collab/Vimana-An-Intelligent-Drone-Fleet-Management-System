"""spatial_index.py - 2-D KD-tree (nearest / k-nearest / radius queries), from scratch."""
import heapq
import math
from typing import Callable, List, Optional, Sequence, Tuple


class _KDNode:
    __slots__ = ("x", "y", "id", "axis", "left", "right")

    def __init__(self, x, y, pid, axis):
        self.x, self.y, self.id, self.axis = x, y, pid, axis
        self.left = self.right = None


class KDTree:
    """Static 2-D KD-tree over (x, y, id) points.
    Build O(n log^2 n) (median split by sorting); query O(log n) average."""

    def __init__(self, points: Sequence[Tuple[float, float, int]]):
        self.size = len(points)
        self.root = self._build(list(points), 0)

    def _build(self, pts, depth):
        if not pts:
            return None
        axis = depth % 2
        pts.sort(key=lambda p: p[axis])
        m = len(pts) // 2
        node = _KDNode(pts[m][0], pts[m][1], pts[m][2], axis)
        node.left = self._build(pts[:m], depth + 1)
        node.right = self._build(pts[m + 1:], depth + 1)
        return node

    def nearest(self, x: float, y: float, k: int = 1,
                predicate: Optional[Callable[[int], bool]] = None) -> List[Tuple[float, int]]:
        """k nearest ids (optionally only those passing `predicate`).
        Returns [(distance, id), ...] sorted ascending."""
        heap: list = []          # max-heap of (-dist^2, id)

        def search(node):
            if node is None:
                return
            if predicate is None or predicate(node.id):
                d2 = (node.x - x) ** 2 + (node.y - y) ** 2
                if len(heap) < k:
                    heapq.heappush(heap, (-d2, node.id))
                elif d2 < -heap[0][0]:
                    heapq.heapreplace(heap, (-d2, node.id))
            diff = (x, y)[node.axis] - (node.x, node.y)[node.axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            search(near)
            if len(heap) < k or diff * diff < -heap[0][0]:   # prune far side
                search(far)

        search(self.root)
        return sorted((math.sqrt(-d2), pid) for d2, pid in heap)

    def range_query(self, x: float, y: float, r: float) -> List[int]:
        """All ids within Euclidean radius r of (x, y)."""
        out: List[int] = []
        r2 = r * r

        def search(node):
            if node is None:
                return
            if (node.x - x) ** 2 + (node.y - y) ** 2 <= r2:
                out.append(node.id)
            diff = (x, y)[node.axis] - (node.x, node.y)[node.axis]
            if diff - r <= 0:
                search(node.left)
            if diff + r >= 0:
                search(node.right)

        search(self.root)
        return out
