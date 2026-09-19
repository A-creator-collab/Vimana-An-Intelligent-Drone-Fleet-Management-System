"""models.py - Core dataclasses shared by every module (Drone, Task, Node, NoFlyZone)."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple
import math

# ---- Physical constants of the simulated world ------------------------------
BATTERY_PER_UNIT = 0.2      # % battery burned per unit distance (empty drone)
PAYLOAD_FACTOR = 0.05       # extra drain per kg carried
CHARGE_RATE = 12.0          # % battery gained per tick while charging
DEFAULT_RETURN_THRESHOLD = 35.0


def energy_cost(distance: float, payload: float = 0.0) -> float:
    """Battery % needed to fly `distance` units while carrying `payload` kg."""
    return distance * BATTERY_PER_UNIT * (1.0 + PAYLOAD_FACTOR * payload)


@dataclass(frozen=True)
class Node:
    """A waypoint in the city graph. Hubs double as charging stations."""
    id: int
    x: float
    y: float
    is_hub: bool = False


@dataclass
class Task:
    """A delivery / patrol job. Patrol tasks have pickup == dropoff."""
    id: int
    pickup: int
    dropoff: int
    priority: int            # 1 (low) .. 5 (urgent)
    weight: float
    created: int
    deadline: int
    kind: str = "delivery"
    status: str = "pending"  # pending|assigned|in_flight|done|failed
    assigned_to: Optional[int] = None
    completed_at: Optional[int] = None
    retries: int = 0
    return_threshold: Optional[float] = None   # set by NLP "return when battery hits X%"


@dataclass
class NoFlyZone:
    """Circular temporary no-fly zone active during [start, end)."""
    id: int
    cx: float
    cy: float
    radius: float
    start: int
    end: int

    def active(self, t: int) -> bool:
        return self.start <= t < self.end

    def contains(self, x: float, y: float) -> bool:
        return math.hypot(x - self.cx, y - self.cy) <= self.radius


@dataclass
class Drone:
    """A simulated drone. All state is plain data - no hardware anywhere."""
    id: int
    node: int
    x: float
    y: float
    battery: float = 100.0
    speed: float = 5.0
    capacity: float = 5.0
    status: str = "idle"   # idle|flying|to_hub|charging|holding|grounded|dead
    task: Optional[Task] = None
    carrying: bool = False
    waypoints: Deque[Tuple[int, List[str]]] = field(default_factory=deque)
    last_node: int = 0
    prev_status: str = "idle"
    hold_goals: list = field(default_factory=list)
    return_threshold: float = DEFAULT_RETURN_THRESHOLD
    distance_flown: float = 0.0
    tasks_done: int = 0
    hold_ticks: int = 0
    # maintenance features (used by the ML model)
    flight_hours: float = 0.0
    battery_cycles: int = 0
    vibration: float = 0.5
    motor_temp: float = 60.0
