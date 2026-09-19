"""simulation_engine.py - Tick-based fleet simulation.

Each tick: admit tasks -> update no-fly zones -> run NLP commands -> dispatch
(Hungarian or greedy, validated) -> move drones (battery drain, dynamic re-planning
around zones) -> charging housekeeping -> proximity check -> record metrics.
"""
import math
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional

from ai.nlp_command_parser import CommandParseError, NLPCommandParser
from data_generator import sector_center, sector_nodes
from dsa.assignment import BIG, greedy_assign, hungarian
from dsa.extras import merge_sort, welsh_powell
from dsa.pathfinding import astar
from dsa.scheduling import AVLTree, TaskQueue
from dsa.spatial_index import KDTree
from dsa.union_find import UnionFind
from models import (CHARGE_RATE, DEFAULT_RETURN_THRESHOLD, Drone, NoFlyZone, Task, energy_cost)
from safety_validator import SafetyValidator

MOVING = ("flying", "to_hub")


class SimulationEngine:
    def __init__(self, graph, drones: List[Drone], tasks: List[Task], hubs: List[int],
                 zones: List[NoFlyZone], apsp, strategy: str = "hungarian"):
        self.g, self.hubs, self.hub_set = graph, hubs, set(hubs)
        self.drones: Dict[int, Drone] = {d.id: d for d in drones}
        self.stream = sorted(tasks, key=lambda t: (t.created, t.id))
        self.ptr = 0
        self.zones, self.strategy, self.apsp = zones, strategy, apsp
        self.queue = TaskQueue()
        self.all_tasks: List[Task] = []
        self.blocked = frozenset()
        self.t = 0
        # spatial indexes (static): hubs and all waypoints
        self.hub_kd = KDTree([(graph.nodes[h].x, graph.nodes[h].y, h) for h in hubs])
        self.node_kd = KDTree([(n.x, n.y, n.id) for n in graph.nodes.values()])
        hub_dist = [min(apsp[n][h] for h in hubs) for n in range(graph.n)]
        self.validator = SafetyValidator(graph, hub_dist)
        self.hub_dist = hub_dist
        self.parser = NLPCommandParser()
        self.commands: List = []            # (tick, text)
        self.command_log: List[str] = []
        self.next_task_id = 100000
        self.stats = Counter()
        self.dispatch_time = 0.0
        self.battery_at_completion: List[float] = []
        self.hist = {"avg_battery": [], "flying": [], "pending": [], "blocked": [], "done": []}
        self.trails = defaultdict(list)

    # ------------------------------------------------------------ public API
    def ground(self, drone_ids) -> None:
        """Take drones out of service (e.g. flagged by the maintenance model)."""
        for i in drone_ids:
            self.drones[i].status = "grounded"

    def schedule_command(self, tick: int, text: str) -> None:
        self.commands.append((tick, text))

    def run(self, ticks: int) -> Dict:
        for _ in range(ticks):
            self.step()
        return self.report()

    def step(self) -> None:
        t = self.t
        self._admit_tasks(t)
        self._update_zones(t)
        self._run_commands(t)
        t0 = time.perf_counter()
        self._dispatch(t)
        self.dispatch_time += time.perf_counter() - t0
        for d in self.drones.values():
            if d.status in MOVING:
                self._advance(d, t)
            elif d.status == "holding":
                d.hold_ticks += 1
                self.stats["hold_ticks"] += 1
                self._replan(d, d.hold_goals, resume=True)
        self._housekeeping()
        self._proximity()
        self._record()
        self.t += 1

    # ------------------------------------------------------------- tick parts
    def _admit_tasks(self, t):
        while self.ptr < len(self.stream) and self.stream[self.ptr].created <= t:
            tk = self.stream[self.ptr]
            self.queue.push(tk); self.all_tasks.append(tk)
            self.ptr += 1

    def _update_zones(self, t):
        nb = set()
        for z in self.zones:
            if z.active(t):
                nb.update(self.node_kd.range_query(z.cx, z.cy, z.radius))
        nb = frozenset(nb)
        if nb != self.blocked:
            self.stats["zone_changes"] += 1
        self.blocked = nb

    def _est_cost(self, d: Drone, tk: Task) -> float:
        """O(1) cost estimate from the Floyd-Warshall matrix; BIG if infeasible."""
        if tk.weight > d.capacity:
            return BIG
        a, b = self.apsp[d.node][tk.pickup], self.apsp[tk.pickup][tk.dropoff]
        need = energy_cost(a) + energy_cost(b, tk.weight) + energy_cost(self.hub_dist[tk.dropoff])
        if d.battery - need < self.validator.min_reserve or a + b >= BIG:
            return BIG
        return a + b

    def _dispatch(self, t):
        if not len(self.queue):
            return
        # AVL tree of idle drones keyed by (battery, id): quickly drop drones too weak for any job
        avl = AVLTree()
        for d in self.drones.values():
            if d.status == "idle":
                avl.insert((d.battery, d.id), d)
        idle = [v for _, v in avl.range_query((self.validator.min_reserve + 5, -1))]
        if not idle:
            return
        tasks = self.queue.pop_many(len(idle))
        cost = [[self._est_cost(d, tk) for d in idle] for tk in tasks]
        pairs, _ = hungarian(cost) if self.strategy == "hungarian" else greedy_assign(cost)
        matched = set()
        for r, c in pairs:
            if cost[r][c] < BIG / 2 and self._launch(idle[c], tasks[r], t):
                matched.add(r)
        for r, tk in enumerate(tasks):
            if r not in matched:
                tk.retries += 1
                self.queue.push(tk)

    def _launch(self, d: Drone, tk: Task, t: int) -> bool:
        p1, _, e1 = astar(self.g, d.node, tk.pickup, self.blocked)
        p2, _, e2 = astar(self.g, tk.pickup, tk.dropoff, self.blocked) if p1 else (None, 0, 0)
        self.stats["nodes_expanded"] += e1 + e2
        if not p1 or not p2:
            self.validator.rejections["no_path"] += 1
            return False
        res = self.validator.validate_mission(d, tk, p1, p2, self.blocked, t)
        if not res.ok:
            return False
        nodes = p1 + p2[1:]
        acts: Dict[int, List[str]] = defaultdict(list)
        acts[len(p1) - 1].append("pickup")
        acts[len(nodes) - 1].append("dropoff")
        d.waypoints = deque((n, acts.get(i, [])) for i, n in enumerate(nodes))
        d.status, d.task, d.carrying = "flying", tk, False
        d.return_threshold = tk.return_threshold or DEFAULT_RETURN_THRESHOLD
        tk.status, tk.assigned_to = "assigned", d.id
        return True

    # --------------------------------------------------------------- movement
    def _at(self, d, node):
        n = self.g.nodes[node]
        return math.hypot(d.x - n.x, d.y - n.y) < 1e-6

    def _advance(self, d: Drone, t: int):
        # a zone popped up ahead of us: turn around (mid-edge) or re-plan (at a node)
        if d.waypoints and d.waypoints[0][0] in self.blocked and d.waypoints[0][0] != d.last_node:
            if self._at(d, d.last_node):
                goals = [w for w in d.waypoints if w[1]]
                if not self._replan(d, goals):
                    return
            else:
                d.waypoints.appendleft((d.last_node, []))
        remaining = d.speed
        while remaining > 1e-9 and d.waypoints and d.status in MOVING:
            node, acts = d.waypoints[0]
            tn = self.g.nodes[node]
            dist = math.hypot(tn.x - d.x, tn.y - d.y)
            move = min(remaining, dist)
            d.battery -= energy_cost(move, d.task.weight if d.carrying and d.task else 0.0)
            d.distance_flown += move
            d.flight_hours += move / max(d.speed, 1e-9) / 60.0
            self.stats["distance"] += move
            if d.battery <= 0:
                d.status = "dead"
                if d.task:
                    d.task.status = "failed"
                self.stats["crashed"] += 1
                return
            if dist <= remaining + 1e-9:
                d.x, d.y = tn.x, tn.y
                remaining -= dist
                d.waypoints.popleft()
                self._arrive(d, node, acts, t)
            else:
                d.x += (tn.x - d.x) / dist * remaining
                d.y += (tn.y - d.y) / dist * remaining
                remaining = 0.0

    def _arrive(self, d: Drone, node: int, acts: List[str], t: int):
        d.node = d.last_node = node
        for a in acts:
            if a == "pickup":
                d.carrying = True
                d.task.status = "in_flight"
            elif a == "dropoff":
                tk = d.task
                tk.status, tk.completed_at = "done", t
                d.carrying, d.task, d.status = False, None, "idle"
                d.tasks_done += 1
                self.battery_at_completion.append(d.battery)
            elif a == "charge":
                d.status = "charging"
                d.waypoints.clear()
        if d.status in MOVING and any(w[0] in self.blocked for w in d.waypoints):
            self._replan(d, [w for w in d.waypoints if w[1]])

    def _replan(self, d: Drone, goals, resume: bool = False) -> bool:
        """A* re-plan from the drone's node through the remaining goals (pickup/dropoff/charge).
        On failure the drone hovers ('holding') until the zone clears."""
        cur, new = d.last_node, []
        for node, acts in goals:
            p, _, e = astar(self.g, cur, node, self.blocked)
            self.stats["nodes_expanded"] += e
            if p is None:
                if not resume:
                    d.prev_status, d.status = d.status, "holding"
                    d.hold_goals, d.waypoints = list(goals), deque()
                    self.stats["holds"] += 1
                return False
            new.extend((x, []) for x in p[1:-1])
            new.append((node, acts))
            cur = node
        d.waypoints = deque(new)
        if resume:
            d.status = d.prev_status
        self.stats["replans"] += 1
        return True

    def _housekeeping(self):
        for d in self.drones.values():
            if d.status == "charging":
                d.battery = min(100.0, d.battery + CHARGE_RATE)
                if d.battery >= 95:
                    d.status = "idle"
                    d.battery_cycles += 1
            elif d.status == "idle":
                at_hub = d.node in self.hub_set
                if at_hub and d.battery < 90:
                    d.status = "charging"
                elif not at_hub and d.battery < d.return_threshold:
                    self._send_to_hub(d)

    def _send_to_hub(self, d: Drone):
        near = self.hub_kd.nearest(d.x, d.y, 1, predicate=lambda h: h not in self.blocked)
        if not near:
            return
        p, _, _ = astar(self.g, d.node, near[0][1], self.blocked)
        if not p:
            return
        d.waypoints = deque([(x, []) for x in p[1:-1]] + [(p[-1], ["charge"])])
        d.status = "to_hub"
        if len(p) == 1:
            d.status, d.waypoints = "charging", deque()

    def _proximity(self):
        """Count pairs of airborne drones closer than 1.0 unit (KD-tree range query)."""
        fly = [d for d in self.drones.values() if d.status in MOVING]
        if len(fly) < 2:
            return
        kd = KDTree([(d.x, d.y, d.id) for d in fly])
        pairs = set()
        for d in fly:
            for j in kd.range_query(d.x, d.y, 1.0):
                if j != d.id:
                    pairs.add((min(j, d.id), max(j, d.id)))
        self.stats["proximity_events"] += len(pairs)

    def _record(self):
        ds = list(self.drones.values())
        self.hist["avg_battery"].append(sum(d.battery for d in ds) / len(ds))
        self.hist["flying"].append(sum(d.status in MOVING for d in ds))
        self.hist["pending"].append(len(self.queue))
        self.hist["blocked"].append(len(self.blocked))
        self.hist["done"].append(sum(1 for tk in self.all_tasks if tk.status == "done"))
        for d in ds[:6]:
            self.trails[d.id].append((d.x, d.y))

    # ------------------------------------------------- NLP command execution
    def _run_commands(self, t):
        for tick, text in [c for c in self.commands if c[0] == t]:
            self.command_log.append(f"[t={t}] '{text}' -> " + self.execute_command(text, t))

    def _resolve(self, loc):
        kind, val = loc
        if kind == "node":
            return val if val in self.g.nodes else None
        cx, cy = sector_center(self.g, val)
        return self.node_kd.nearest(cx, cy, 1)[0][1]

    def execute_command(self, text: str, t: int) -> str:
        try:
            cmd = self.parser.parse(text)
        except CommandParseError as e:
            self.validator.rejections["cmd:unparseable"] += 1
            return f"REJECTED (parse error: {e})"
        res = self.validator.validate_command(cmd, len(self.drones))
        if not res.ok:
            return f"REJECTED by safety validator {res.reasons}  [{cmd.action} {cmd.params}]"
        p = cmd.params
        if cmd.action == "dispatch":
            nodes = sector_nodes(self.g, p["sector"])
            for i in range(p["count"]):
                n = nodes[i * len(nodes) // p["count"]]
                tk = Task(self.next_task_id, n, n, 5, 0.0, t, t + 40, "patrol",
                          return_threshold=p.get("return_battery"))
                self.next_task_id += 1
                self.queue.push(tk); self.all_tasks.append(tk)
            return f"APPROVED: {p['count']} patrol task(s) queued for sector {p['sector']}, return threshold {p.get('return_battery', 'default')}%"
        if cmd.action == "deliver":
            a, b = self._resolve(p["from"]), self._resolve(p["to"])
            if a is None or b is None:
                return "REJECTED (unknown node)"
            tk = Task(self.next_task_id, a, b, p["priority"], p["weight"], t, t + 60)
            self.next_task_id += 1
            self.queue.push(tk); self.all_tasks.append(tk)
            return f"APPROVED: delivery {a}->{b} priority {p['priority']} queued"
        if cmd.action == "nofly":
            cx, cy = sector_center(self.g, p["sector"])
            self.zones.append(NoFlyZone(900 + t, cx, cy, 14.0, t, t + p["duration"]))
            return f"APPROVED: no-fly zone over sector {p['sector']} for {p['duration']} ticks"
        if cmd.action == "status":
            d = self.drones[p["drone_id"]]
            return f"drone {d.id}: {d.status}, battery {d.battery:.0f}%, node {d.node}, done {d.tasks_done}"
        if cmd.action == "recall":
            ids = [p["drone_id"]] if p["drone_id"] is not None else list(self.drones)
            n = 0
            for i in ids:
                d = self.drones[i]
                if d.status in ("idle", "flying", "to_hub", "holding"):
                    if d.task:                      # put the job back on the queue
                        d.task.status = "pending"; self.queue.push(d.task); d.task = None
                    d.carrying = False
                    d.status = "idle"
                    d.waypoints.clear()
                    self._send_to_hub(d)
                    n += 1
            return f"APPROVED: {n} drone(s) recalled to nearest hub"
        return "REJECTED"

    # ------------------------------------------------------------- analysis
    def analyze_fleet(self, radius: float = 25.0) -> Dict:
        """Union-Find swarm clusters, graph-colouring radio channels, no-fly region merging, ranking."""
        ds = list(self.drones.values())
        idx = {d.id: i for i, d in enumerate(ds)}
        kd = KDTree([(d.x, d.y, d.id) for d in ds])
        uf, adj = UnionFind(len(ds)), {d.id: set() for d in ds}
        for d in ds:
            for j in kd.range_query(d.x, d.y, radius):
                if j != d.id:
                    uf.union(idx[d.id], idx[j]); adj[d.id].add(j)
        channels = welsh_powell(adj)
        zuf = UnionFind(len(self.zones)) if self.zones else None
        if zuf:
            for i, a in enumerate(self.zones):
                for j in range(i + 1, len(self.zones)):
                    b = self.zones[j]
                    if math.hypot(a.cx - b.cx, a.cy - b.cy) <= a.radius + b.radius:
                        zuf.union(i, j)
        ranked = merge_sort(ds, key=lambda d: (d.tasks_done, d.battery), reverse=True)
        return {"swarm_clusters": [sorted(ds[i].id for i in grp) for grp in uf.groups()],
                "channels_used": (max(channels.values()) + 1) if channels else 0,
                "no_fly_regions": zuf.components if zuf else 0,
                "top_drones": [(d.id, d.tasks_done, round(d.battery)) for d in ranked[:3]]}

    def report(self) -> Dict:
        total = len(self.all_tasks)
        done = [tk for tk in self.all_tasks if tk.status == "done"]
        late = sum(1 for tk in done if tk.completed_at > tk.deadline)
        waits = [tk.completed_at - tk.created for tk in done]
        return {
            "strategy": self.strategy, "tasks_total": total, "completed": len(done),
            "completion_rate_%": round(100 * len(done) / max(total, 1), 1), "late": late,
            "avg_turnaround_ticks": round(sum(waits) / max(len(waits), 1), 1),
            "avg_battery_reserve_%": round(sum(self.battery_at_completion) / max(len(self.battery_at_completion), 1), 1),
            "fleet_distance": round(self.stats["distance"]),
            "distance_per_task": round(self.stats["distance"] / max(len(done), 1), 1), "validator_rejections": sum(self.validator.rejections.values()),
            "rejection_breakdown": dict(self.validator.rejections), "replans": self.stats["replans"],
            "hold_ticks": self.stats["hold_ticks"], "crashed": self.stats["crashed"],
            "proximity_events": self.stats["proximity_events"], "astar_nodes_expanded": self.stats["nodes_expanded"],
            "dispatch_time_s": round(self.dispatch_time, 3)}
