"""
algorithms/voila/voila_algorithm.py
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
    OCC_UP_THRESHOLD = 0.65
    OCC_DOWN_THRESHOLD = 0.20

    SCALE_DOWN_PATIENCE_TICKS = 3
    PROXIMITY_SUSTAIN_TICKS = 2
    PROXIMITY_PROTECTION_TICKS = 5

    def __init__(self):
        self._good_streak: Dict[int, int] = {}
        self._last_snapshot: Optional[dict] = None

        self._proximity_violation_streak: Dict[int, int] = {}
        self._proximity_recent: Dict[int, int] = {}

        self._vivaldi: Optional[VivaldiNetwork] = None

    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        self._last_snapshot = metrics_snapshot
        sv = metrics_snapshot["services"][service_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

        capacity_violation = occ_ratio > self.OCC_UP_THRESHOLD or sv.get("rejection_rate_rolling", sv["rejection_rate"]) > 0.0
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
                occ_ratio < self.OCC_DOWN_THRESHOLD and sv["n_ready_replicas"] > 1):
            self._good_streak[service_id] = 0
            return ScaleAction.SCALE_DOWN

        return ScaleAction.NO_CHANGE
    
    
    """def select_replica(self, request: Request, candidate_replicas: List[Replica],
                        servers: Dict[int, Server], now: float) -> Optional[Replica]:
     
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
            true_dist_km = haversine_km(request.bts_lat, request.bts_long,
                                         servers[chosen.server_id].lat, servers[chosen.server_id].long)
            true_rtt_ms = 2 * network_delay_ms(true_dist_km, CFG.base_latency_ms, CFG.k_ms_per_km)
            self._vivaldi.observe(request.bts_lat, request.bts_long, chosen.server_id, true_rtt_ms)

        return chosen"""
    # ------------------------------------------------------------------
    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["resource_mips"]

        centroid = None
        if self._last_snapshot is not None:
            centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
        if centroid is None:
            active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
            clat = sum(s.lat for s in active) / len(active)
            clon = sum(s.long for s in active) / len(active)
            centroid = (clat, clon)

        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE
                      and s.can_host(service_id, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
        if not candidates:
            return None

        distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long) for s in candidates}
        min_dist = min(distances.values())
        near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
        return max(near_pool, key=lambda s: s.free_capacity()).id

    # ------------------------------------------------------------------
    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        self._last_snapshot = metrics_snapshot
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        overloaded = [s for s in active
                      if metrics_snapshot["servers"][s.id]["utilization"] > CFG.util_scale_up_threshold]

        starved_services = self._capacity_starved_services(metrics_snapshot, servers,
                                                             occ_threshold=self.OCC_UP_THRESHOLD)
        if overloaded or starved_services:
            off_servers = [s for s in servers.values() if s.state == ServerState.OFF]
            if off_servers:
                ref_lat, ref_lon = None, None
                ref = None
                if overloaded:
                    ref = overloaded[0]
                    ref_lat, ref_lon = ref.lat, ref.long
                elif starved_services:
                    worst = max(starved_services,
                                key=lambda sid: metrics_snapshot["services"][sid]["rejection_rate"])
                    centroid = metrics_snapshot["services"][worst].get("demand_centroid")
                    if centroid:
                        ref_lat, ref_lon = centroid
                    if active:
                        ref = max(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
                        if ref_lat is None:
                            ref_lat, ref_lon = ref.lat, ref.long

                # *** رفع باگ (heterogeneity-aware TURN_ON): قبلاً VOILA برخلاف
                # Greedy/HPA (که هر دو از _pick_profile_for_overload +
                # _filter_by_profile_with_fallback استفاده می‌کنند - نگاه کنید
                # README بخش ۸.۲) سرور OFF را فقط بر اساس نزدیکی جغرافیایی
                # انتخاب می‌کرد، بدون فیلتر پروفایل - یعنی ممکن بود یک
                # edge_small روشن شود وقتی نیاز واقعی به ظرفیت large بود و
                # سیستم مجبور شود چند TURN_ON پشت‌سرهم بزند.
                fallback_capacity = ref.capacity if ref is not None else off_servers[0].capacity
                # *** رفع بازگشت باگ: "overloaded or active" وقتی overloaded
                # خالی است (فقط starved_services) کل لیست active را پاس
                # می‌دهد - دقیقاً همان باگی که در greedy_algorithm.py رفع شده
                # بود (مجموع ظرفیت کل فلیت تقریباً همیشه از آستانه‌ی large
                # می‌گذرد، پس desired_profile صرف‌نظر از شدت واقعی starvation
                # تقریباً همیشه "large" می‌شود و fallback_capacity هیچ‌وقت
                # استفاده نمی‌شود). باید فقط overloaded پاس داده شود تا وقتی
                # خالی است، خودِ _pick_profile_for_overload طبق طراحی اصلی‌اش
                # از fallback_capacity (همان ظرفیت شلوغ‌ترین سرور فعال) استفاده کند.
                desired_profile = self._pick_profile_for_overload(overloaded, fallback_capacity)
                pool = self._filter_by_profile_with_fallback(off_servers, desired_profile)

                if ref_lat is not None:
                    pool = sorted(pool, key=lambda s: haversine_km(ref_lat, ref_lon, s.lat, s.long))
                else:
                    pool = sorted(pool, key=lambda s: s.id)
                return ProvisionAction(ProvisionActionType.TURN_ON, pool[0].id)
            # *** رفع باگ ۲ (fallthrough): هم‌راستا با Greedy/HPA - وقتی
            # overload/starvation تشخیص داده شده ولی سرور خاموشی نمانده، دیگر به
            # بررسی TURN_OFF زیر سقوط نمی‌کند.
            return ProvisionAction(ProvisionActionType.NO_CHANGE)

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
            cpu = CFG.services_info[service_id]["resource_mips"]
            centroid = None
            if self._last_snapshot is not None:
                centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
            ref_lat, ref_lon = centroid if centroid else (draining_server.lat, draining_server.long)
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu, bts_lat=ref_lat, bts_long=ref_lon)]
            if not candidates:
                continue
            candidates.sort(key=lambda s: haversine_km(ref_lat, ref_lon, s.lat, s.long))
            steps.append(MigrationStep(service_id=service_id, target_server_id=candidates[0].id))
        return steps

    def select_scale_down_victim(self, service_id, ready_replicas, servers, now, occupancy_fn=None):
        occ = occupancy_fn or (lambda r: r.queue_occupancy(now))
        centroid = None
        if self._last_snapshot is not None:
            centroid = self._last_snapshot["services"][service_id].get("demand_centroid")
        if centroid is None or len(ready_replicas) <= 1:
            return super().select_scale_down_victim(service_id, ready_replicas, servers, now,
                                                      occupancy_fn=occupancy_fn)
        by_load = sorted(ready_replicas, key=occ)
        low_load_pool = by_load[:max(1, len(by_load) // 2)]
        return max(low_load_pool, key=lambda r: haversine_km(
            centroid[0], centroid[1], servers[r.server_id].lat, servers[r.server_id].long))