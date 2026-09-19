"""extras.py - Remaining DSA topics: merge sort / quicksort with custom comparators,
Trie, Welsh-Powell graph colouring, sliding-window statistics."""
from collections import deque
from typing import Callable, Dict, Iterable, List, Sequence, Set


def merge_sort(items: Sequence, key: Callable = lambda x: x, reverse: bool = False) -> List:
    """Stable O(n log n) merge sort with a key function (multi-criteria ranking via tuple keys)."""
    a = list(items)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    l, r = merge_sort(a[:mid], key, reverse), merge_sort(a[mid:], key, reverse)
    out, i, j = [], 0, 0
    while i < len(l) and j < len(r):
        take_left = key(l[i]) >= key(r[j]) if reverse else key(l[i]) <= key(r[j])
        if take_left:
            out.append(l[i]); i += 1
        else:
            out.append(r[j]); j += 1
    out.extend(l[i:]); out.extend(r[j:])
    return out


def quick_sort(items: Sequence, key: Callable = lambda x: x) -> List:
    """3-way quicksort, average O(n log n)."""
    a = list(items)
    if len(a) <= 1:
        return a
    pivot = key(a[len(a) // 2])
    less = [x for x in a if key(x) < pivot]
    eq = [x for x in a if key(x) == pivot]
    more = [x for x in a if key(x) > pivot]
    return quick_sort(less, key) + eq + quick_sort(more, key)


class Trie:
    """Prefix tree for dashboard search over drone IDs / location names."""

    def __init__(self):
        self.root: Dict = {}

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word.lower():
            node = node.setdefault(ch, {})
        node["$"] = True

    def starts_with(self, prefix: str) -> List[str]:
        node = self.root
        for ch in prefix.lower():
            if ch not in node:
                return []
            node = node[ch]
        out: List[str] = []
        def walk(n, acc):
            for ch, child in sorted(n.items()):
                if ch == "$": out.append(acc)
                else: walk(child, acc + ch)
        walk(node, prefix.lower())
        return out


def welsh_powell(adj: Dict[int, Set[int]]) -> Dict[int, int]:
    """Greedy graph colouring, highest-degree first. Colours = radio channels /
    time-slots so that nearby (adjacent) drones never share one."""
    order = merge_sort(list(adj), key=lambda v: len(adj[v]), reverse=True)
    color: Dict[int, int] = {}
    for v in order:
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


class SlidingWindowStats:
    """Fixed-size window with O(1) running mean / std (streaming telemetry)."""

    def __init__(self, size: int = 20):
        self.size = size
        self.buf: deque = deque()
        self.s = 0.0
        self.s2 = 0.0

    def push(self, x: float) -> None:
        self.buf.append(x); self.s += x; self.s2 += x * x
        if len(self.buf) > self.size:
            old = self.buf.popleft(); self.s -= old; self.s2 -= old * old

    @property
    def mean(self) -> float:
        return self.s / len(self.buf) if self.buf else 0.0

    @property
    def std(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        return max(self.s2 / n - (self.s / n) ** 2, 0.0) ** 0.5
