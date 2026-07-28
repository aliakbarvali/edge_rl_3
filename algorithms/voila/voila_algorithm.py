"""
algorithms/voila/voila_algorithm.py

پیاده‌سازی فاز ۲: الگوریتم مبتنی بر فلسفه‌ی مقاله‌ی Voila (بخش ۱ تا ۷ آن)،
تطبیق‌داده‌شده با معماری این پروژه:

    - initial_placement/select_replica: از پیاده‌سازی مشترک AlgorithmBase
      استفاده می‌شود (خودِ سند در بخش ۴/۵ این دو را بین همه‌ی الگوریتم‌ها
      مشترک تعریف کرده - دقیقاً همان Procedure 3 مقاله).
    - تفاوت اصلی Voila با Greedy اینجاست که placement/migration را بر پایه‌ی
      *مرکز ثقل تقاضای واقعی* هر سرویس (demand_centroid، میانگین متحرک
      موقعیت جغرافیایی درخواست‌های اخیر - نگاه کنید به
      simulator/engine.py:_build_metrics_snapshot) انتخاب می‌کند، نه صرفاً
      نزدیک‌ترین به مرکز سرورهای فعال (که Greedy انجام می‌دهد).
    - scale_decision مشابه Procedure 4 مقاله: ترکیب نقض ظرفیت (اشغال صف) و
      نقض دسترس‌پذیری (rejection_rate) به‌عنوان معیار E سرویس؛ scale-down
      فقط بعد از چند تیک متوالی بدون نقض (safety/patience - بخش V-C مقاله).

*** نکته‌ی طراحی: چون select_placement_server(service_id, servers) در
اینترفیس AlgorithmBase به metrics_snapshot دسترسی ندارد ولی Voila برای
انتخاب مکان به demand_centroid نیاز دارد، scale_decision (که همیشه بلافاصله
قبل از select_placement_server برای همان سرویس در همان تیک صدا زده می‌شود -
نگاه کنید به simulator/engine.py:_apply_scale_decision) آخرین snapshot را
کش می‌کند.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep


class VoilaAlgorithm(AlgorithmBase):
    name = "voila"

    # --- آستانه‌های heuristic اختصاصی Voila (نه قید سخت سیستم؛ در config.py
    # نیستند چون مختص سیاست تصمیم‌گیری این الگوریتم‌اند، نه فیزیک سیستم) ---
    OCC_UP_THRESHOLD = 0.75    # مشابه co مقاله: اشغال صف بالاتر از این = نقض ظرفیت
    OCC_DOWN_THRESHOLD = 0.20  # زیر این = این replica کم‌بار است
    SCALE_DOWN_PATIENCE_TICKS = 3  # طبق بخش V-C مقاله: «۳ چرخه بدون نقض»

    def __init__(self):
        self._good_streak: Dict[int, int] = {}
        self._last_snapshot: Optional[dict] = None

    # ------------------------------------------------------------------
    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        self._last_snapshot = metrics_snapshot  # برای select_placement_server همین تیک
        sv = metrics_snapshot["services"][service_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

        # Procedure 4 مقاله: نقض = یا نقض ظرفیت (co) یا نقض دسترس‌پذیری (عدم پوشش/رد شدن)
        violation = occ_ratio > self.OCC_UP_THRESHOLD or sv["rejection_rate"] > 0.0

        if violation:
            self._good_streak[service_id] = 0
            return ScaleAction.SCALE_UP

        self._good_streak[service_id] = self._good_streak.get(service_id, 0) + 1
        if (self._good_streak[service_id] >= self.SCALE_DOWN_PATIENCE_TICKS and
                occ_ratio < self.OCC_DOWN_THRESHOLD and sv["n_replicas"] > 1):
            self._good_streak[service_id] = 0
            return ScaleAction.SCALE_DOWN

        return ScaleAction.NO_CHANGE

    # ------------------------------------------------------------------
    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None

        centroid = None
        if self._last_snapshot is not None:
            centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
        if centroid is None:
            # هنوز داده‌ی کافی از موقعیت درخواست‌های این سرویس نداریم -> نزدیک‌ترین
            # به مرکز ثقل سرورهای فعال فعلی (fallback معقول، مثل Greedy)
            active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
            clat = sum(s.lat for s in active) / len(active)
            clon = sum(s.long for s in active) / len(active)
            centroid = (clat, clon)

        candidates.sort(key=lambda s: haversine_km(centroid[0], centroid[1], s.lat, s.long))
        return candidates[0].id

    # ------------------------------------------------------------------
    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        overloaded = [s for s in active
                      if metrics_snapshot["servers"][s.id]["utilization"] > CFG.util_scale_up_threshold]
        if overloaded:
            off_servers = [s for s in servers.values() if s.state == ServerState.OFF]
            if off_servers:
                ref = overloaded[0]
                off_servers.sort(key=lambda s: haversine_km(ref.lat, ref.long, s.lat, s.long))
                return ProvisionAction(ProvisionActionType.TURN_ON, off_servers[0].id)

        if active:
            idle = min(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
            if metrics_snapshot["servers"][idle.id]["utilization"] < CFG.util_scale_down_threshold:
                return ProvisionAction(ProvisionActionType.TURN_OFF, idle.id)

        return ProvisionAction(ProvisionActionType.NO_CHANGE)

    # ------------------------------------------------------------------
    def migration_decision(self, draining_server: Server,
                            servers: Dict[int, Server]) -> List[MigrationStep]:
        steps = []
        for service_id, replica in draining_server.hosted_replicas.items():
            if replica.state == ReplicaState.TERMINATED:
                continue
            other_hosts = [s for s in servers.values()
                           if s.id != draining_server.id and service_id in s.hosted_replicas
                           and s.hosted_replicas[service_id].state != ReplicaState.TERMINATED]
            if other_hosts:
                continue
            cpu = CFG.services_info[service_id]["cpu_demand"]
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu)]
            if not candidates:
                continue
            centroid = None
            if self._last_snapshot is not None:
                centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
            ref_lat, ref_lon = centroid if centroid else (draining_server.lat, draining_server.long)
            candidates.sort(key=lambda s: haversine_km(ref_lat, ref_lon, s.lat, s.long))
            steps.append(MigrationStep(service_id=service_id, target_server_id=candidates[0].id))
        return steps
