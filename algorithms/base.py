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

    # *** جدید: آیا این الگوریتم مجاز است گیت sustain-tracking پیش‌فرض
    # (SUSTAIN_HIGH_SEC/SUSTAIN_LOW_SEC در simulator/engine.py و
    # k8s_adapter/realtime_dispatcher.py:_apply_provisioning) را برای
    # TURN_ON/TURN_OFF دور بزند؟ پیش‌فرض False برای همه‌ی الگوریتم‌ها
    # (Greedy/HPA/VOILA) - یعنی رفتار قبلی برایشان دقیقاً حفظ می‌شود.
    # این فقط اکشن‌های TURN_ON/TURN_OFF را تحت‌تأثیر قرار می‌دهد؛
    # cooldown و min_active_duration (قیود عملیاتی/سخت‌افزاری، نه بخشی
    # از این آستانه‌ی reactive) صرف‌نظر از این پرچم همچنان برای همه اعمال
    # می‌شوند. ممیزی decision_correctness هم صرف‌نظر از این پرچم دقیقاً
    # طبق همان معیار عینی مستقل ادامه پیدا می‌کند - این پرچم فقط تعیین
    # می‌کند آیا اکشن *اعمال* می‌شود، نه اینکه در ممیزی «درست» شمرده شود.
    bypass_sustain_gate: bool = False

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

    def select_replica(self, request, candidate_replicas, servers, now,
                        admit_fn=None, occupancy_fn=None):

        if not candidate_replicas:
            return None
        # *** admit_fn ممکن است side-effect داشته باشد (مثلاً واقعاً یک
        # اسلات صف را در Redis رزرو کند - نگاه کنید realtime_dispatcher.py).
        # به همین دلیل نباید در حین ساختن/رتبه‌بندی «استخر نزدیک» (near_pool)
        # روی چند کاندید مختلف صدا زده شود، وگرنه هر رپلیکای هم‌فاصله‌ای که
        # فقط برای رتبه‌بندی چک می‌شود هم واقعاً رزرو می‌شود و رزروهای
        # استفاده‌نشده تا انقضای TTL به‌صورت phantom باقی می‌مانند (نشتی صف).
        # پس اینجا فقط از یک تابع فقط-خواندنی (occupancy_fn) برای رتبه‌بندی
        # استفاده می‌کنیم و admit_fn (رزرو واقعی) را دقیقاً یک‌بار، فقط روی
        # کاندیدای نهایی انتخاب‌شده صدا می‌زنیم.
        occupancy_fn = occupancy_fn or (lambda r: r.queue_occupancy(now))
        admit_fn = admit_fn or (lambda r: occupancy_fn(r) < r.queue_len)

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
        # نزدیک‌ترین) و طبق آخرین اطلاعات (بدون رزرو کردن) هنوز جا دارند
        near_pool = [(d, r) for d, r in dist_pairs
                     if d <= min_dist + 5.0 and occupancy_fn(r) < r.queue_len]
        # بین هم‌فاصله‌ها، کم‌ترافیک‌ترین (کم‌ترین نسبت اشغال صف) در اولویت است؛
        # بقیه به‌عنوان fallback اگر رزروِ گزینه‌ی اول به‌خاطر race شکست بخورد
        ordered = sorted(near_pool, key=lambda pair: occupancy_fn(pair[1]) / max(pair[1].queue_len, 1))
        # هر چیزی خارج از near_pool هم به ترتیب فاصله به‌عنوان fallback نهایی
        ordered += sorted([p for p in dist_pairs if p not in near_pool], key=lambda pair: pair[0])

        for _, r in ordered:
            if admit_fn(r):          # رزرو واقعی، دقیقاً یک‌بار برای هر کاندیدا،
                return r              # و فقط تا وقتی که یکی موفق بشود
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
            centroid = sv.get("demand_centroid")
            bts_lat, bts_long = centroid if centroid else (None, None)
            if not any(s.state in (ServerState.ACTIVE, ServerState.BOOTING)
                    and s.can_host(svc_id, cpu, bts_lat=bts_lat, bts_long=bts_long)
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