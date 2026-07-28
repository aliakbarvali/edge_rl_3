"""
algorithms/base.py
کلاس انتزاعی مشترک بین Greedy/Voila/HPA/PPO، طبق بخش ۱۰ سند معماری.

دو متد (initial_placement و select_replica) یک پیاده‌سازی پیش‌فرض *مشترک*
دارند چون سند صریحاً می‌گوید این دو بخش بین همه‌ی الگوریتم‌ها یکسان است
(بخش ۴: پوشش اولیه به سبک Voila Procedure 3؛ بخش ۵: مسیریابی بر پایه‌ی
فاصله+صف، نه چیزی که هر الگوریتم جدا طراحی کند). بقیه‌ی متدها (scale_decision،
provision_decision، migration_decision) باید توسط هر زیرکلاس پیاده شوند چون
دقیقاً همان‌هایی هستند که الگوریتم‌ها را از هم متمایز می‌کنند.
"""

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

    # ------------------------------------------------------------------
    # بخش ۴: پوشش اولیه (پیاده‌سازی مشترک، پیش‌فرض برای همه‌ی الگوریتم‌ها)
    # ------------------------------------------------------------------
    def initial_placement(self, servers: Dict[int, Server],
                           active_bts: List[tuple]) -> List[int]:
        """
        پوشش حریصانه‌ی Set-Cover-Style (مشابه Voila Procedure 3): حداقل
        زیرمجموعه‌ای از سرورها که تمام BTSهای فعال (active_bts: لیست
        (lat, long)) را در محدوده‌ی l0 پوشش دهد.
        خروجی: لیست server_id هایی که باید روشن شوند.
        """
        remaining = set(range(len(active_bts)))
        covers: Dict[int, set] = {}
        for sid, srv in servers.items():
            covered = set()
            for i, (lat, lon) in enumerate(active_bts):
                d_km = haversine_km(lat, lon, srv.lat, srv.long)
                delay = CFG.base_latency_ms + CFG.k_ms_per_km * d_km
                if delay <= CFG.l0_ms:
                    covered.add(i)
            covers[sid] = covered

        selected: List[int] = []
        while remaining:
            best_sid, best_cover = None, set()
            for sid, covered in covers.items():
                if sid in selected:
                    continue
                inter = covered & remaining
                if len(inter) > len(best_cover):
                    best_sid, best_cover = sid, inter
            if best_sid is None or len(best_cover) == 0:
                break  # هیچ سرور باقی‌مانده‌ای پوشش بیشتری نمی‌دهد
            selected.append(best_sid)
            remaining -= best_cover
            if len(selected) == len(servers):
                break
        if not selected:  # حالت لبه: هیچ BTS فعالی نبود - حداقل یک سرور روشن کن
            selected = [next(iter(servers))]

        # *** گسترش انتخاب تا ظرفیت کل هم کافی باشد (نه فقط پوشش جغرافیایی):
        # مجموع cpu_demand هر ۱۵ سرویس معمولاً از ظرفیت تک‌تک سرورها بیشتر است،
        # پس صرفِ پوشش جغرافیایی کافی نیست - باید سرور کافی برای جاگیری همه‌ی
        # سرویس‌ها هم روشن شود (بخش ۴: «اگر پوشش اولیه کافی نبود، همه‌ی
        # سرورها روشن می‌شوند» - همین قاعده را برای کمبود ظرفیت هم اعمال می‌کنیم).
        total_cpu_needed = sum(s["cpu_demand"] for s in CFG.services_info.values())
        remaining_servers = [sid for sid in servers if sid not in selected]
        remaining_servers.sort(key=lambda sid: min(
            haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
            for s2 in selected) if selected else 0)
        while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining_servers:
            selected.append(remaining_servers.pop(0))

        return selected

    # ------------------------------------------------------------------
    # بخش ۵: مسیریابی/انتخاب نمونه (پیاده‌سازی مشترک، پیش‌فرض برای همه)
    # ------------------------------------------------------------------
    def select_replica(self, request: Request,
                        candidate_replicas: List[Replica],
                        servers: Dict[int, Server], now: float) -> Optional[Replica]:
        """
        candidate_replicas: تمام رپلیکاهای READY همان سرویس (از هر جای سیستم).
        بر اساس فاصله‌ی جغرافیایی صعودی مرتب و اولین رپلیکایی که صفش پر
        نیست انتخاب می‌شود (بخش ۵/۳).
        """
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
        return None  # همه‌ی رپلیکاهای READY صف پر دارند -> REJECTED_QUEUE_FULL

    # ------------------------------------------------------------------
    # متدهایی که هر الگوریتم باید خودش پیاده کند (منطق تمایزدهنده)
    # ------------------------------------------------------------------
    @abstractmethod
    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        ...

    @abstractmethod
    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        ...

    @abstractmethod
    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        """کدام سرور برای رپلیکای *جدید* این سرویس (بعد از SCALE_UP) انتخاب شود؟"""
        ...

    @abstractmethod
    def migration_decision(self, draining_server: Server,
                            servers: Dict[int, Server]) -> List[MigrationStep]:
        ...
