"""scheduling.py - Heap task queue, AVL tree, and interval-based airspace reservations."""
import bisect
import heapq
from typing import Dict, Hashable, List, Optional, Tuple

from models import Task


class TaskQueue:
    """Min-heap ordered by (-priority, deadline, created, id). O(log n) push/pop.
    Uses lazy deletion so remove() is O(1)."""

    def __init__(self):
        self._heap: list = []
        self._alive: Dict[int, Task] = {}

    def push(self, task: Task) -> None:
        key = (-task.priority, task.deadline, task.created, task.id)
        heapq.heappush(self._heap, (key, task.id))
        self._alive[task.id] = task

    def pop(self) -> Optional[Task]:
        while self._heap:
            _, tid = heapq.heappop(self._heap)
            t = self._alive.pop(tid, None)
            if t is not None:
                return t
        return None

    def pop_many(self, k: int) -> List[Task]:
        out = []
        while len(out) < k:
            t = self.pop()
            if t is None:
                break
            out.append(t)
        return out

    def remove(self, tid: int) -> None:
        self._alive.pop(tid, None)

    def __len__(self) -> int:
        return len(self._alive)


# ---------------------------------------------------------------- AVL tree
class _AVLNode:
    __slots__ = ("key", "val", "h", "l", "r")

    def __init__(self, key, val):
        self.key, self.val, self.h, self.l, self.r = key, val, 1, None, None


def _h(n): return n.h if n else 0
def _upd(n): n.h = 1 + max(_h(n.l), _h(n.r))
def _bal(n): return _h(n.l) - _h(n.r)


def _rot_right(y):
    x = y.l
    y.l, x.r = x.r, y
    _upd(y); _upd(x)
    return x


def _rot_left(x):
    y = x.r
    x.r, y.l = y.l, x
    _upd(x); _upd(y)
    return y


def _rebalance(n):
    _upd(n)
    b = _bal(n)
    if b > 1:
        if _bal(n.l) < 0:
            n.l = _rot_left(n.l)
        return _rot_right(n)
    if b < -1:
        if _bal(n.r) > 0:
            n.r = _rot_right(n.r)
        return _rot_left(n)
    return n


class AVLTree:
    """Self-balancing BST keyed by e.g. (battery, drone_id): O(log n) insert/delete,
    O(log n + k) range query -> 'all idle drones with battery >= X'."""

    def __init__(self):
        self.root = None
        self._n = 0

    def __len__(self): return self._n
    def height(self): return _h(self.root)

    def insert(self, key, val=None):
        def ins(n):
            if n is None:
                self._n += 1
                return _AVLNode(key, val)
            if key < n.key: n.l = ins(n.l)
            elif key > n.key: n.r = ins(n.r)
            else: n.val = val
            return _rebalance(n)
        self.root = ins(self.root)

    def delete(self, key):
        def mn(n):
            while n.l: n = n.l
            return n

        def dele(n, k):
            if n is None:
                return None
            if k < n.key: n.l = dele(n.l, k)
            elif k > n.key: n.r = dele(n.r, k)
            else:
                if n.l is None or n.r is None:
                    self._n -= 1
                    return n.l or n.r
                s = mn(n.r)
                n.key, n.val = s.key, s.val
                n.r = dele(n.r, s.key)
            return _rebalance(n)
        self.root = dele(self.root, key)

    def inorder(self):
        out = []
        def walk(n):
            if n:
                walk(n.l); out.append((n.key, n.val)); walk(n.r)
        walk(self.root)
        return out

    def max(self):
        n = self.root
        while n and n.r: n = n.r
        return (n.key, n.val) if n else None

    def range_query(self, lo, hi=None):
        """All (key, val) with lo <= key (<= hi), ascending."""
        out = []
        def walk(n):
            if n is None: return
            if n.key > lo: walk(n.l)
            if n.key >= lo and (hi is None or n.key <= hi): out.append((n.key, n.val))
            if hi is None or n.key < hi: walk(n.r)
        walk(self.root)
        return out


# ---------------------------------------------- interval-based reservations
class ReservationTable:
    """Time-windowed airspace reservations (interval-tree role).
    Each resource (e.g. an undirected corridor) keeps its intervals sorted by
    start time; bisect + a max-length bound gives O(log n + k) overlap counts.
    A reservation is refused if `capacity` drones already hold an overlapping slot."""

    def __init__(self, capacity: int = 1):
        self.capacity = capacity
        self._starts: Dict[Hashable, List[int]] = {}
        self._ivals: Dict[Hashable, List[Tuple[int, int]]] = {}
        self._maxlen: Dict[Hashable, int] = {}

    def overlaps(self, res: Hashable, s: int, e: int) -> int:
        starts = self._starts.get(res)
        if not starts:
            return 0
        ivals = self._ivals[res]
        i = bisect.bisect_left(starts, s - self._maxlen[res])
        cnt = 0
        while i < len(ivals) and ivals[i][0] <= e:
            if ivals[i][1] >= s:
                cnt += 1
            i += 1
        return cnt

    def is_free(self, res: Hashable, s: int, e: int) -> bool:
        return self.overlaps(res, s, e) < self.capacity

    def reserve(self, res: Hashable, s: int, e: int) -> None:
        starts = self._starts.setdefault(res, [])
        ivals = self._ivals.setdefault(res, [])
        i = bisect.bisect_left(starts, s)
        starts.insert(i, s)
        ivals.insert(i, (s, e))
        self._maxlen[res] = max(self._maxlen.get(res, 0), e - s)
