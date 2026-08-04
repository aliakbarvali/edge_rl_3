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

*** انحراف عمدی از بخش ۵ سند (مستند، با تأیید صریح): سند صراحتاً می‌گوید
routing/instance-selection بین هر ۴ الگوریتم مشترک و صرفاً بر پایه‌ی فاصله‌ی
جغرافیایی *واقعی* (oracle) است، نه latency واقعاً اندازه‌گیری‌شده مثل مقاله‌ی
اصلی VOILA (که از Vivaldi/Serf استفاده می‌کند). طبق درخواست صریح برای
مقایسه‌ی وفادارتر به مقاله، *فقط* VoilaAlgorithm.select_replica اینجا override
شده تا به‌جای فاصله‌ی جغرافیایی واقعی، از یک سیستم مختصات Vivaldi واقعی
(common/network_coordinates.py) استفاده کند - یعنی هیچ دانش پیشینی از موقعیت
BTS ندارد و فقط از طریق RTT مشاهده‌شده‌ی درخواست‌های قبلی یاد می‌گیرد.
Greedy/HPA/PPO دست‌نخورده می‌مانند و همچنان از AlgorithmBase.select_replica
(oracle) استفاده می‌کنند - این عمداً یک مقایسه‌ی نامتقارن‌تر ولی وفادارتر به
مقاله‌ی هرکدام ایجاد می‌کند: Voila دیگر oracle نیست.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km, network_delay_ms
from common.models import Server, ServerState, ReplicaState, Replica, Request
from common.network_coordinates import VivaldiNetwork
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep


class VoilaAlgorithm(AlgorithmBase):
    name = "voila"
    OCC_UP_THRESHOLD = 0.68
    OCC_DOWN_THRESHOLD = 0.20
    SCALE_DOWN_PATIENCE_TICKS = 3
    PROXIMITY_SUSTAIN_TICKS = 1 
    PROXIMITY_PROTECTION_TICKS = 3
   

    def __init__(self):
        self._good_streak: Dict[int, int] = {}
        self._last_snapshot: Optional[dict] = None

        self._proximity_violation_streak: Dict[int, int] = {} 
        self._proximity_recent: Dict[int, int] = {}

        # *** lazy: چون ساخت VivaldiNetwork نیاز به دیکشنری servers دارد که
        # در __init__ الگوریتم در دسترس نیست (VoilaAlgorithm() بدون آرگومان
        # ساخته می‌شود - نگاه کنید run.py/compare_runs.py)، اولین بار که
        # select_replica صدا زده شود ساخته می‌شود.
        self._vivaldi: Optional[VivaldiNetwork] = None

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
            self._proximity_violation_streak[service_id] = 0
            return ScaleAction.SCALE_UP

        if proximity_violation:
            streak = self._proximity_violation_streak.get(service_id, 0) + 1
            self._proximity_violation_streak[service_id] = streak
            self._good_streak[service_id] = 0
            if streak < self.PROXIMITY_SUSTAIN_TICKS:
                return ScaleAction.NO_CHANGE   
            self._proximity_violation_streak[service_id] = 0 
            self._proximity_recent[service_id] = self.PROXIMITY_PROTECTION_TICKS
            return ScaleAction.SCALE_UP

        self._proximity_violation_streak[service_id] = 0
        
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
    def select_replica(self, request: Request, candidate_replicas: List[Replica],
                        servers: Dict[int, Server], now: float) -> Optional[Replica]:
        """
        بازنویسی instance-selection (بخش ۳ و ۵ سند) طبق مقاله‌ی اصلی VOILA:
        رتبه‌بندی رپلیکاها بر پایه‌ی RTT *تخمینی* (Vivaldi، ممکن است هنوز
        ناهمگرا/ناقص باشد)، نه فاصله‌ی جغرافیایی واقعی. بعد از انتخاب نهایی،
        RTT *واقعی* رپلیکای انتخاب‌شده به‌عنوان یک نمونه‌ی مشاهده به سیستم
        Vivaldi بازخورد داده می‌شود تا تخمین آینده دقیق‌تر شود - دقیقاً مثل
        یک client واقعی که فقط با peerهایی که واقعاً باهاشان تعامل کرده RTT
        اندازه می‌گیرد.

        قید صف (queue_occupancy < queue_len) دقیقاً مثل نسخه‌ی مشترک حفظ
        شده - فقط معیار *ترتیب* رپلیکاها عوض شده، نه قید پذیرش.
        """
        if not candidate_replicas:
            return None
        if self._vivaldi is None:
            self._vivaldi = VivaldiNetwork(servers, CFG.base_latency_ms, CFG.k_ms_per_km,
                                            seed=CFG.seed)

        ranked = sorted(
            candidate_replicas,
            key=lambda r: self._vivaldi.estimate_rtt_ms(request.bts_lat, request.bts_long, r.server_id),
        )
        chosen = None
        for r in ranked:
            if r.queue_occupancy(now) < r.queue_len:
                chosen = r
                break

        if chosen is not None:
            # *** بازخورد پس از انتخاب: تنها اینجا RTT واقعی (نه تخمینی) در
            # دسترس است - چون فقط برای رپلیکای واقعاً انتخاب‌شده معنادار است.
            true_dist_km = haversine_km(request.bts_lat, request.bts_long,
                                         servers[chosen.server_id].lat, servers[chosen.server_id].long)
            true_rtt_ms = 2 * network_delay_ms(true_dist_km, CFG.base_latency_ms, CFG.k_ms_per_km)
            self._vivaldi.observe(request.bts_lat, request.bts_long, chosen.server_id, true_rtt_ms)

        return chosen

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
            active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
            clat = sum(s.lat for s in active) / len(active)
            clon = sum(s.long for s in active) / len(active)
            centroid = (clat, clon)
 
        distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long) for s in candidates}
        min_dist = min(distances.values()) 
        near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
        return max(near_pool, key=lambda s: s.free_capacity()).id
    
 

    # ------------------------------------------------------------------
    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        overloaded = [s for s in active
                      if metrics_snapshot["servers"][s.id]["utilization"] > CFG.util_scale_up_threshold]
       
        # *** هماهنگ با threshold داخلی خودِ Voila (OCC_UP_THRESHOLD=0.68)،
        # نه ۰.۷ هاردکد Greedy - تا سیگنال starvation دقیقاً همان لحظه‌ای
        # trigger شود که scale_decision خودِ Voila هم نیاز را تشخیص می‌دهد.
        starved_services = self._capacity_starved_services(metrics_snapshot, servers,
                                                             occ_threshold=self.OCC_UP_THRESHOLD)
        if overloaded or starved_services:
            off_servers = [s for s in servers.values() if s.state == ServerState.OFF]
            if off_servers:
                ref_lat, ref_lon = None, None
                if overloaded:
                    ref = overloaded[0]
                    ref_lat, ref_lon = ref.lat, ref.long
                elif starved_services:
                    worst = max(starved_services,
                                key=lambda sid: metrics_snapshot["services"][sid]["rejection_rate"])
                    centroid = metrics_snapshot["services"][worst].get("demand_centroid")
                    if centroid:
                        ref_lat, ref_lon = centroid
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
    
    
         
   
            
    def select_scale_down_victim(self, service_id, ready_replicas, servers, now):
        centroid = None
        if self._last_snapshot is not None:
            centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
        if centroid is None or len(ready_replicas) <= 1:
            return super().select_scale_down_victim(service_id, ready_replicas, servers, now) 
        by_load = sorted(ready_replicas, key=lambda r: r.queue_occupancy(now))
        low_load_pool = by_load[:max(1, len(by_load) // 2)]
        return max(low_load_pool, key=lambda r: haversine_km(
            centroid[0], centroid[1], servers[r.server_id].lat, servers[r.server_id].long))