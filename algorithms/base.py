from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, Replica, Request, ServerState, ReplicaState


class ScaleAction(Enum):
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    NO_CHANGE = auto()


class ProvisionActionType(Enum):
    TURN_ON = auto()
    TURN_OFF = auto()
    NO_CHANGE = auto()


@dataclass
class ProvisionAction:
    action: ProvisionActionType
    server_id: Optional[int] = None


@dataclass
class MigrationStep:
    service_id: int
    target_server_id: int


class AlgorithmBase(ABC):
    name: str = "base"

    def initial_placement(self, servers, active_bts):
        remaining = set(range(len(active_bts)))
        covers = {}
        for sid, srv in servers.items():
            covered = set()
            for i, (lat, lon) in enumerate(active_bts):
                d_km = haversine_km(lat, lon, srv.lat, srv.long)
                delay = CFG.base_latency_ms + CFG.k_ms_per_km * d_km
                if delay <= CFG.l0_ms:
                    covered.add(i)
            covers[sid] = covered

        selected = []
        while remaining:
            best_sid, best_cover = None, set()
            for sid, covered in covers.items():
                if sid in selected:
                    continue
                inter = covered & remaining
                if len(inter) > len(best_cover):
                    best_sid, best_cover = sid, inter
            if best_sid is None or len(best_cover) == 0:
                break
            selected.append(best_sid)
            remaining -= best_cover
            if len(selected) == len(servers):
                break
        if not selected:
            selected = [next(iter(servers))]

        total_cpu_needed = sum(s["resource_mips"] for s in CFG.services_info.values())
        remaining_servers = [sid for sid in servers if sid not in selected]
        remaining_servers.sort(key=lambda sid: min(
            haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
            for s2 in selected) if selected else 0)
        while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining_servers:
            selected.append(remaining_servers.pop(0))

        return selected

    def select_replica(self, request, candidate_replicas, servers, now, admit_fn=None):

        if not candidate_replicas:
            return None
        admit_fn = admit_fn or (lambda r: r.queue_occupancy(now) < r.queue_len)

        # *** توجه: Replica یک dataclass غیر-frozen است و به‌طور پیش‌فرض
        # __hash__ ندارد (unhashable) - پس نباید از خودِ آبجکت به‌عنوان کلید
        # dict استفاده شود. به‌جایش فاصله را در یک لیست از تاپل‌های
        # (distance, replica) نگه می‌داریم.
        dist_pairs = [
            (haversine_km(request.bts_lat, request.bts_long,
                          servers[r.server_id].lat, servers[r.server_id].long), r)
            for r in candidate_replicas
        ]
        min_dist = min(d for d, _ in dist_pairs)

        # همه‌ی سرورهایی که تقریباً هم‌فاصله‌اند (در محدوده‌ی +۵ کیلومتر
        # نزدیک‌ترین) و در حال حاضر پذیرا هستند
        near_pool = [(d, r) for d, r in dist_pairs if d <= min_dist + 5.0 and admit_fn(r)]
        if near_pool:
            # بین هم‌فاصله‌ها، کم‌ترافیک‌ترین (کم‌ترین نسبت اشغال صف) انتخاب می‌شود
            return min(near_pool, key=lambda pair: pair[1].queue_occupancy(now) / max(pair[1].queue_len, 1))[1]

        # اگر در استخر نزدیک هیچ‌کدام صف خالی نداشتند، به ترتیب فاصله ادامه بده
        for d, r in sorted(dist_pairs, key=lambda pair: pair[0]):
            if admit_fn(r):
                return r
        return None
 
    @staticmethod
    def _pick_profile_for_overload(overloaded_servers: List[Server], fallback_capacity: int) -> str:
        _large_threshold = CFG.server_profiles["large"]["capacity_mips"]
        _medium_threshold = CFG.server_profiles["medium"]["capacity_mips"]
        total = sum(s.capacity for s in overloaded_servers) if overloaded_servers else fallback_capacity
        if total >= _large_threshold:
            return "large"
        if total >= _medium_threshold:
            return "medium"
        return "edge_small"

    @staticmethod
    def _filter_by_profile_with_fallback(candidates: List[Server], desired_profile: str) -> List[Server]:
        matching = [s for s in candidates if s.profile == desired_profile]
        return matching if matching else candidates 

    
    def select_scale_down_victim(self, service_id, ready_replicas, servers, now, occupancy_fn=None):
 
        occupancy_fn = occupancy_fn or (lambda r: r.queue_occupancy(now))
        return min(ready_replicas, key=occupancy_fn)

    @staticmethod
    def _capacity_starved_services(metrics_snapshot: dict, servers: Dict[int, Server],
                                    occ_threshold: float = 0.7) -> List[int]:

        starved = []
        for svc_id, sv in metrics_snapshot["services"].items():
            occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
            if not (occ_ratio > occ_threshold or sv["rejection_rate"] > 0.0):
                continue
            
            cpu = CFG.services_info[svc_id]["resource_mips"]
            if not any(s.state in (ServerState.ACTIVE, ServerState.BOOTING) and s.can_host(svc_id, cpu)
                       for s in servers.values()):
                starved.append(svc_id)
        return starved

    @abstractmethod
    def scale_decision(self, service_id, metrics_snapshot):
        ...

    @abstractmethod
    def provision_decision(self, servers, metrics_snapshot, now):
        ...

    @abstractmethod
    def select_placement_server(self, service_id, servers):
        ...

    @abstractmethod
    def migration_decision(self, draining_server, servers):
        ...