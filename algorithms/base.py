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

        total_cpu_needed = sum(s["cpu_demand"] for s in CFG.services_info.values())
        remaining_servers = [sid for sid in servers if sid not in selected]
        remaining_servers.sort(key=lambda sid: min(
            haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
            for s2 in selected) if selected else 0)
        while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining_servers:
            selected.append(remaining_servers.pop(0))

        return selected

    def select_replica(self, request, candidate_replicas, servers, now):
        if not candidate_replicas:
            return None
        ranked = sorted(
            candidate_replicas,
            key=lambda r: haversine_km(request.bts_lat, request.bts_long,
                                        servers[r.server_id].lat, servers[r.server_id].long),
        )
        for r in ranked:
            if r.queue_occupancy(now) < r.queue_len:
                return r
        return None

    # ------------------------------------------------------------------
    # بخش ۶.۱: انتخاب پروفایل سرور خاموش متناسب با میزان اضافه‌بار
    # (پیاده‌سازی مشترک - چون معیارش «میزان اضافه‌بار فعلی سیستم» است، نه
    # یک تصمیم مختص فلسفه‌ی هر الگوریتم؛ Greedy/Voila آن را با اولویت
    # نزدیک‌ترین فاصله ترکیب می‌کنند، HPA به‌عمد بدون فاصله فراخوانی‌اش
    # می‌کند تا location-unaware بماند - نگاه کنید hpa_algorithm.py).
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_profile_for_overload(overloaded_servers: List[Server], fallback_capacity: int) -> str:
        """
        بخش ۶.۱ سند: «با پروفایل ظرفیتی متناسب با میزان اضافه‌بار (تقاضای
        بیشتر -> ترجیح large، تقاضای کم -> ترجیح edge_small)». سند مقدار
        عددی دقیق آستانه را مشخص نکرده (بخش ۱۳: قابل کالیبراسیون)؛ از مجموع
        ظرفیت سرورهای ACTIVه‌ی اشباع‌شده‌ی فعلی به‌عنوان proxy معقول برای
        «میزان اضافه‌بار» استفاده شده - هرچه این مجموع بزرگ‌تر، سرور بزرگ‌تری
        لازم است.
        """
        total = sum(s.capacity for s in overloaded_servers) if overloaded_servers else fallback_capacity
        if total >= 200:
            return "large"
        if total >= 100:
            return "medium"
        return "edge_small"

    @staticmethod
    def _filter_by_profile_with_fallback(candidates: List[Server], desired_profile: str) -> List[Server]:
        matching = [s for s in candidates if s.profile == desired_profile]
        return matching if matching else candidates  # اگر پروفایل دلخواه در دسترس نبود -> همه‌ی کاندیدها

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