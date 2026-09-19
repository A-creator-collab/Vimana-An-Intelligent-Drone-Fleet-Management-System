"""safety_validator.py - Deterministic approve/reject layer.

Design principle (borrowed from the dronesphere reference): AI components and
optimisers only PROPOSE. Nothing reaches the simulation engine until these
rule-based checks approve it.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Sequence

from dsa.scheduling import ReservationTable
from models import Drone, Task, energy_cost


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)


class SafetyValidator:
    def __init__(self, graph, hub_dist: Sequence[float], min_reserve: float = 15.0,
                 reservation_capacity: int = 3):
        """hub_dist[node] = weighted distance from node to its nearest charging hub."""
        self.g = graph
        self.hub_dist = hub_dist
        self.min_reserve = min_reserve
        self.reservations = ReservationTable(capacity=reservation_capacity)
        self.rejections: Counter = Counter()
        self.approvals = 0

    # ------------------------------------------------------------ missions
    def validate_mission(self, drone: Drone, task: Task, p1: List[int], p2: List[int],
                         blocked, t: int) -> ValidationResult:
        reasons = []
        nodes = p1 + p2[1:]
        if task.weight > drone.capacity:
            reasons.append("payload_exceeds_capacity")
        if any(b not in self.g.adj[a] for a, b in zip(nodes, nodes[1:])):
            reasons.append("invalid_corridor")
        if any(n in blocked for n in nodes[1:]):
            reasons.append("no_fly_zone")
        need = (energy_cost(self.g.path_length(p1)) + energy_cost(self.g.path_length(p2), task.weight)
                + energy_cost(self.hub_dist[task.dropoff]))
        if drone.battery - need < self.min_reserve:
            reasons.append("insufficient_battery")
        # time-windowed airspace reservation (only checked when everything else passes)
        windows = []
        if not reasons:
            dist_so_far = 0.0
            for a, b in zip(nodes, nodes[1:]):
                s = t + int(dist_so_far / drone.speed)
                dist_so_far += self.g.euclid(a, b)
                e = t + int(dist_so_far / drone.speed) + 1
                windows.append(((min(a, b), max(a, b)), s, e))
            if any(not self.reservations.is_free(res, s, e) for res, s, e in windows):
                reasons.append("airspace_conflict")
        if reasons:
            for r in reasons:
                self.rejections[r] += 1
            return ValidationResult(False, reasons)
        for res, s, e in windows:
            self.reservations.reserve(res, s, e)
        self.approvals += 1
        return ValidationResult(True)

    # ------------------------------------------------------------ commands
    def validate_command(self, cmd, fleet_size: int) -> ValidationResult:
        """Checks applied to NLP-proposed commands before they are queued."""
        p, reasons = cmd.params, []
        if cmd.action == "dispatch":
            if not 1 <= p.get("count", 1) <= fleet_size: reasons.append("bad_drone_count")
            if p.get("sector") not in ("A", "B", "C", "D"): reasons.append("unknown_sector")
            rb = p.get("return_battery")
            if rb is not None and not 10 <= rb <= 50: reasons.append("return_battery_out_of_range")
        elif cmd.action == "deliver":
            if not 1 <= p.get("priority", 3) <= 5: reasons.append("bad_priority")
            if p.get("weight", 1.0) > 8.0: reasons.append("payload_exceeds_fleet_max")
        elif cmd.action in ("status", "recall"):
            did = p.get("drone_id")
            if did is not None and not 0 <= did < fleet_size: reasons.append("unknown_drone")
        elif cmd.action == "nofly":
            if p.get("sector") not in ("A", "B", "C", "D"): reasons.append("unknown_sector")
            if not 1 <= p.get("duration", 20) <= 120: reasons.append("bad_duration")
        for r in reasons:
            self.rejections["cmd:" + r] += 1
        return ValidationResult(not reasons, reasons)
