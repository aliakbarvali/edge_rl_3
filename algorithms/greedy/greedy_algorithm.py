from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep
 

class GreedyAlgorithm(AlgorithmBase):
    name = "greedy"

    def __init__(self):
        # *** رفع باگ (fairness/SLA feasibility): قبلاً Greedy هیچ snapshot ای
        # cache نمی‌کرد، پس select_placement_server/migration_decision هیچ
        # راهی برای گرفتن demand_centroid واقعی سرویس نداشتند و can_host
        # همیشه بدون bts_lat/bts_long صدا زده می‌شد -> همیشه از مسیر
        # محافظه‌کارانه‌ی «بدترین فاصله‌ی ممکن از ۴ گوشه‌ی نقشه» رد می‌شد
        # (نگاه کنید common/config.py:is_sla_feasible). این یعنی مقایسه‌ی
        # Greedy در برابر VOILA (که همین snapshot را cache می‌کند) منصفانه
        # نبود: تفاوت observed performance تا حدی از این می‌آمد که فقط VOILA
        # اطلاعات موقعیت دقیق‌تری در همین چک SLA داشت، نه صرفاً سیاست بهتر.
        # الان Greedy هم دقیقاً مثل VOILA/PPO یک self._last_snapshot cache
        # می‌کند تا از همان demand_centroid برای can_host استفاده کند.
        self._last_snapshot: dict | None = None

    def scale_decision(self, service_id, metrics_snapshot):
        self._last_snapshot = metrics_snapshot
        svc = metrics_snapshot["services"][service_id]
        queue_len = svc["queue_len"]
        occ_ratio = svc["avg_queue_occupancy"] / queue_len if queue_len else 0.0
        if occ_ratio > 0.7 or svc["rejection_rate"] > 0:
            return ScaleAction.SCALE_UP
        if occ_ratio < 0.1 and svc["n_ready_replicas"] > 1:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE

    def provision_decision(self, servers, metrics_snapshot, now):
        self._last_snapshot = metrics_snapshot
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        overloaded = [s for s in active
                      if metrics_snapshot["servers"][s.id]["utilization"] > CFG.util_scale_up_threshold]
        starved_services = self._capacity_starved_services(metrics_snapshot, servers, occ_threshold=0.7)
        if overloaded or starved_services:
            off_servers = [s for s in servers.values() if s.state == ServerState.OFF]
            if off_servers:
                if overloaded:
                    ref = overloaded[0]
                elif active: 
                    ref = max(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
                else: 
                    ref = None
                if ref is not None:
                    desired_profile = self._pick_profile_for_overload(overloaded or active, ref.capacity)
                    pool = self._filter_by_profile_with_fallback(off_servers, desired_profile)
                    pool.sort(key=lambda s: haversine_km(ref.lat, ref.long, s.lat, s.long))
                else:
                    pool = off_servers
                return ProvisionAction(ProvisionActionType.TURN_ON, pool[0].id)

        if active:
            idle = min(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
            if metrics_snapshot["servers"][idle.id]["utilization"] < CFG.util_scale_down_threshold:
                return ProvisionAction(ProvisionActionType.TURN_OFF, idle.id)

        return ProvisionAction(ProvisionActionType.NO_CHANGE)

    def _demand_centroid_or_none(self, service_id):
        if self._last_snapshot is None:
            return None
        return self._last_snapshot["services"].get(service_id, {}).get("demand_centroid")

    def select_placement_server(self, service_id, servers):
        cpu = CFG.services_info[service_id]["resource_mips"]
        active_all = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        clat = sum(s.lat for s in active_all) / len(active_all)
        clon = sum(s.long for s in active_all) / len(active_all)
        # *** رفع باگ: از demand_centroid واقعی سرویس (اگر موجود باشد) به‌جای
        # فقط مرکز سرورهای فعال، هم برای can_host (چک SLA) و هم برای مرتب‌سازی
        # فاصله استفاده می‌شود - همان چیزی که VOILA/PPO از قبل انجام می‌دادند.
        centroid = self._demand_centroid_or_none(service_id) or (clat, clon)
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE
                      and s.can_host(service_id, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
        if not candidates:
            return None
        candidates.sort(key=lambda s: haversine_km(clat, clon, s.lat, s.long))
        return candidates[0].id

    def migration_decision(self, draining_server, servers):
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
            centroid = self._demand_centroid_or_none(service_id)
            ref_lat, ref_lon = centroid if centroid else (draining_server.lat, draining_server.long)
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu, bts_lat=ref_lat, bts_long=ref_lon)]
            if candidates:
                candidates.sort(key=lambda s: haversine_km(draining_server.lat, draining_server.long,
                                                             s.lat, s.long))
                steps.append(MigrationStep(service_id=service_id, target_server_id=candidates[0].id))
        return steps