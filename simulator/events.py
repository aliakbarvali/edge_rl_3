"""
simulator/events.py

"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    REQUEST_ARRIVAL = auto()
    REQUEST_ROUTED = auto()
    DECISION_TICK = auto()         
    SERVER_BOOT_DONE = auto()       # BOOTING -> ACTIVE
    SERVER_DRAIN_DONE = auto()      # DRAINING -> OFF
    REPLICA_READY = auto()          # STARTING -> READY
    REPLICA_TERMINATED = auto()     # DRAINING -> TERMINATED
    ENERGY_RESYNC = auto()         


@dataclass(order=True)
class Event:
    time: float
    seq: int                      
    type: EventType = field(compare=False)
    payload: Any = field(default=None, compare=False)