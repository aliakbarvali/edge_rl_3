"""
common/models.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Dict, Optional


class ServerState(Enum):
    OFF = auto()
    BOOTING = auto()
    ACTIVE = auto()
    DRAINING = auto()


class ReplicaState(Enum):
    STARTING = auto()
    READY = auto()
    DRAINING = auto()
    TERMINATED = auto()


class RequestStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()
    REJECTED_QUEUE_FULL = auto()
    REJECTED_NO_REPLICA = auto()


@dataclass
class Replica:
    service_id: int
    server_id: int
    queue_len: int
    exec_time: float
    state: ReplicaState = ReplicaState.STARTING
    created_at: float = 0.0
    ready_since: Optional[float] = None
    drain_started_at: Optional[float] = None

    available_at: float = 0.0           
    departures: deque = field(default_factory=deque)  

    def is_selectable(self) -> bool:
        return self.state == ReplicaState.READY

    def queue_occupancy(self, now: float) -> int:
        
        while self.departures and self.departures[0] <= now:
            self.departures.popleft()
        return len(self.departures)

    def try_admit(self, arrival_time: float, cold_start_extra: float = 0.0):
       
        occ = self.queue_occupancy(arrival_time)
        if occ >= self.queue_len:
            return None
        start = max(arrival_time, self.available_at)
        service_time = self.exec_time + cold_start_extra
        finish = start + service_time
        self.available_at = finish
        self.departures.append(finish)
        return {
            "queue_enter_time": arrival_time,
            "service_start_time": start,
            "service_end_time": finish,
            "wait_time_sec": start - arrival_time,
        }

    def is_idle(self, now: float) -> bool:
        return self.queue_occupancy(now) == 0


@dataclass
class Server:
    id: int
    profile: str
    lat: float
    long: float
    capacity: int 
    p_idle: float
    p_max: float
    state: ServerState = ServerState.OFF
    hosted_replicas: Dict[int, Replica] = field(default_factory=dict) 
    boot_started_at: Optional[float] = None
    drain_started_at: Optional[float] = None
    last_transition_time: float = -1e18 
    cumulative_energy_joule: float = 0.0
    cumulative_busy_cpu_seconds: float = 0.0  
    num_boots: int = 0
    num_shutdowns: int = 0

    def _speed_factor(self) -> float:
        from common.config import CFG, REFERENCE_MIPS_PER_CORE
        return CFG.server_profiles[self.profile]["mips_per_core"] / REFERENCE_MIPS_PER_CORE

    def used_cpu(self) -> int:
        # مجموع effective_mips واقعیِ رزروشده روی *این* میزبان مشخص
        # (نه resource_mips خام که نسبت به سرور مرجع تعریف شده)
        return sum(self._cpu_of(r) for r in self.hosted_replicas.values())

    def _cpu_of(self, replica: Replica) -> int:
        from common.config import CFG
        resource_mips = CFG.services_info[replica.service_id]["resource_mips"]
        return round(resource_mips * self._speed_factor())

    def free_capacity(self) -> int:
        return self.capacity - self.used_cpu()

    def can_host(self, service_id: int, cpu_demand: int) -> bool:
        # cpu_demand طبق قرارداد پروژه، resource_mips خام (نسبت به سرور مرجع)
        # است - همان‌طور که همه‌ی الگوریتم‌ها صدایش می‌زنند؛ اینجا آن را به
        # effective_mips واقعیِ *این* میزبان تبدیل می‌کنیم تا با واحد
        # capacity (که خودش واقعی است) قابل‌مقایسه شود.
        if service_id in self.hosted_replicas:
            return False
        real_demand = round(cpu_demand * self._speed_factor())
        return self.free_capacity() >= real_demand
    



    def can_host(self, service_id: int, cpu_demand: int) -> bool:
        from common.config import CFG, is_sla_feasible
        if service_id in self.hosted_replicas:
            return False
        if self.free_capacity() < cpu_demand:
            return False
        return is_sla_feasible(service_id, self.lat, self.long,
                                CFG.server_profiles[self.profile]["mips_per_core"])


    def in_cooldown(self, now: float, cooldown_sec: float) -> bool:
        return (now - self.last_transition_time) < cooldown_sec


    
    def instantaneous_utilization(self, now: float) -> float:
      
        busy_cpu = 0
        for r in self.hosted_replicas.values():
            if r.state in (ReplicaState.READY, ReplicaState.DRAINING) and not r.is_idle(now):
                busy_cpu += self._cpu_of(r)
        return busy_cpu / self.capacity if self.capacity > 0 else 0.0
    
    def instantaneous_power_w(self, now: float) -> float:
        if self.state == ServerState.OFF:
            return 0.0
        if self.state == ServerState.BOOTING:
            return self.p_idle
        util = self.instantaneous_utilization(now)
        return self.p_idle + (self.p_max - self.p_idle) * util


@dataclass
class Request:
    id: int
    bts_lat: float
    bts_long: float
    service_id: int
    arrival_time: float
    assigned_server_id: Optional[int] = None
    queue_enter_time: Optional[float] = None
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    network_delay_ms: float = 0.0    
    routing_delay_sec: float = 0.0
    wait_time_sec: float = 0.0
    response_time_sec: float = 0.0
    deadline_violated: bool = False
    status: RequestStatus = RequestStatus.PENDING