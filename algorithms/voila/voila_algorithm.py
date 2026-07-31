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

*** پچ (بازبینی): نسخه‌ی قبلی این فایل نقض deadline (`deadline_violation_rate`)
را در محاسبه‌ی `violation` نادیده می‌گرفت - فقط `occ_ratio` (نقض ظرفیت) و
`rejection_rate` چک می‌شدند. این یعنی یک سرویس که به‌خاطر *فاصله‌ی جغرافیایی*
(نه ازدحام صف) دائم deadline نقض می‌کرد - یعنی دقیقاً نقض proximity طبق
Procedure 4 مقاله (Vlo) - می‌توانست occ_ratio پایین داشته باشد،
good_streak بالا برود، و بعد از SCALE_DOWN_PATIENCE_TICKS واقعاً
SCALE_DOWN بخورد؛ درست برعکسِ آنچه سند می‌خواهد (Vlo هم باید نقض حساب
شود، نه فقط Vco). حالا `deadline_violation_rate > 0` هم بخشی از تشخیص
نقض است؛ چون این‌جا AlgorithmBase اکشن مستقیم "Replace" ندارد (برای اصلاح
tail-latency بدون افزودن replica جدید - دقیقاً Procedure 5 مقاله)، در حالت
نقض proximity-محض (occ_ratio پایین ولی deadline_violation_rate>0) به‌جای
SCALE_UP بی‌مورد فقط جلوی SCALE_DOWN اشتباه گرفته می‌شود (NO_CHANGE)؛
اصلاح واقعی مکان از طریق select_placement_server/migration_decision که هر
دو از قبل proximity-aware بودند به‌تدریج اتفاق می‌افتد.

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
    OCC_UP_THRESHOLD = 0.75
    OCC_DOWN_THRESHOLD = 0.20
    SCALE_DOWN_PATIENCE_TICKS = 3
    # *** جدید: چند تیک بعد از یک نقض proximity، از SCALE_DOWN همان سرویس
    # صرف‌نظر شود - تقریب محافظ‌کارانه‌ی مفهوم «رپلیکای vital» در Procedure 7
    # مقاله (چون AlgorithmBase انتخاب قربانی SCALE_DOWN را به engine واگذار
    # می‌کند - مشترک بین هر ۴ الگوریتم - Voila نمی‌تواند یک رپلیکای مشخص را
    # مستقیماً "محافظت‌شده" اعلام کند؛ این نزدیک‌ترین معادل قابل‌اجرا است).
    PROXIMITY_MEMORY_TICKS = 5

    def __init__(self):
        self._good_streak: Dict[int, int] = {}
        self._proximity_recent: Dict[int, int] = {}
        self._last_snapshot: Optional[dict] = None

    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        self._last_snapshot = metrics_snapshot
        sv = metrics_snapshot["services"][service_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

        # Vco (Procedure 4): نقض ظرفیت واقعی
        capacity_violation = occ_ratio > self.OCC_UP_THRESHOLD or sv["rejection_rate"] > 0.0
        # *** Vlo واقعی حالا از proximity_violation_rate (RTT>l0 خالص) می‌آید،
        # نه از deadline_violation_rate که پروکسی نویزی بود.
        proximity_violation = (not capacity_violation) and sv["proximity_violation_rate"] > 0.0

        if capacity_violation:
            self._good_streak[service_id] = 0
            self._proximity_recent[service_id] = 0
            return ScaleAction.SCALE_UP

        if proximity_violation:
            # *** طبق Procedure 4/5: نقض proximity هم باید فعالانه رفع شود.
            # چون این اینترفیس اکشن "replace" مستقیم ندارد، معادل دوتیکی‌اش
            # اجرا می‌شود: یک SCALE_UP در مکان مناسب (select_placement_server
            # از قبل demand_centroid-aware است و نزدیک تقاضای واقعی جا
            # می‌گذارد)، و بعداً که SCALE_DOWN_PATIENCE سپری شد، رپلیکای
            # کم‌فایده‌ی قدیمی توسط چرخه‌ی عادی زیر حذف می‌شود - قبلاً این‌جا
            # فقط NO_CHANGE بود که عملاً هیچ اقدامی برای رفع Vlo نمی‌کرد.
            self._good_streak[service_id] = 0
            self._proximity_recent[service_id] = self.PROXIMITY_MEMORY_TICKS
            return ScaleAction.SCALE_UP

        self._good_streak[service_id] = self._good_streak.get(service_id, 0) + 1
        self._proximity_recent[service_id] = max(0, self._proximity_recent.get(service_id, 0) - 1)

        if self._proximity_recent.get(service_id, 0) > 0:
            return ScaleAction.NO_CHANGE

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
        # *** بخش ۶.۱ / یافته‌ی جدید: قبلاً این‌جا فقط utilization لحظه‌ای چک
        # می‌شد که یک سرور کاملاً پر (free_capacity=0) را می‌توانست به‌اشتباه
        # "غیراضافه‌بار" نشان دهد (چون busy-fraction آن لزوماً >0.95 نیست).
        # Greedy همین سیگنال را گرفته؛ اینجا هم اضافه شد تا مقایسه‌ی چهارگانه
        # منصفانه بماند - نگاه کنید algorithms/base.py:_capacity_starved_services.
        starved_services = self._capacity_starved_services(metrics_snapshot, servers)
        if overloaded or starved_services:
            off_servers = [s for s in servers.values() if s.state == ServerState.OFF]
            if off_servers:
                ref_lat, ref_lon = None, None
                if overloaded:
                    ref = overloaded[0]
                    ref_lat, ref_lon = ref.lat, ref.long
                elif starved_services:
                    # *** فلسفه‌ی Voila: مرکز ثقل تقاضای واقعی سرویس‌های
                    # starved را مرجع مکانی می‌گیریم (نه صرفاً سرور پرمشغول)،
                    # چون این همان چیزی است که Voila را از Greedy متمایز
                    # می‌کند - محل واقعی تقاضا، نه محل خودِ سرورها.
                    centroids = [metrics_snapshot["services"][sid].get("demand_centroid")
                                 for sid in starved_services]
                    centroids = [c for c in centroids if c is not None]
                    if centroids:
                        ref_lat = sum(c[0] for c in centroids) / len(centroids)
                        ref_lon = sum(c[1] for c in centroids) / len(centroids)
                    elif active:
                        ref = max(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
                        ref_lat, ref_lon = ref.lat, ref.long
                if ref_lat is not None:
                    off_servers.sort(key=lambda s: haversine_km(ref_lat, ref_lon, s.lat, s.long))
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