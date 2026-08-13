from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep


class GreedyAlgorithm(AlgorithmBase):
    name = "greedy"

    def __init__(self):
        # cache برای demand_centroid واقعی هر سرویس (fairness/SLA feasibility
        # با VOILA/PPO) - نگاه کنید select_placement_server/migration_decision
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
                    # *** رفع باگ ۴: قبلاً "overloaded or active" پاس داده می‌شد؛ وقتی
                    # overloaded خالی بود (فقط starved_services)، کل لیست active (تا
                    # ۱۰ سرور) به‌عنوان مجموعه‌ی "overload‌شده" به _pick_profile_for_overload
                    # می‌رفت. چون آن تابع مجموع capacity ورودی را با آستانه‌ی large/medium
                    # مقایسه می‌کند، مجموع کل فلیت تقریباً همیشه از آستانه‌ی large می‌گذرد -
                    # یعنی تابع صرف‌نظر از شدت واقعی کمبود، تقریباً همیشه "large" برمی‌گرداند
                    # (اثر مستقیم روی cumulative_energy_joule). حالا فقط overloadهای واقعی
                    # پاس داده می‌شود؛ وقتی خالی است، خودِ تابع (طبق طراحی اصلی‌اش) به
                    # fallback_capacity=ref.capacity برمی‌گردد که نماینده‌ی معنادارتری از
                    # شدت starvation (شلوغ‌ترین سرور فعال) است.
                    desired_profile = self._pick_profile_for_overload(overloaded, ref.capacity)
                    pool = self._filter_by_profile_with_fallback(off_servers, desired_profile)
                    pool.sort(key=lambda s: haversine_km(ref.lat, ref.long, s.lat, s.long))
                else:
                    pool = off_servers
                return ProvisionAction(ProvisionActionType.TURN_ON, pool[0].id)
            # *** رفع باگ ۲ (fallthrough): وقتی overload/starvation تشخیص داده شده
            # ولی هیچ سرور خاموشی برای روشن‌کردن نمانده (off_servers خالی)، قبلاً کد
            # بدون return به بلوک بررسی TURN_OFF زیر سقوط می‌کرد و ممکن بود دقیقاً در
            # همان تیکی که سیستم گرسنه/اضافه‌بار است، یک سرور idle دیگر را هم خاموش
            # کند - تناقض مستقیم با جهت درست تصمیم. حالا صریحاً NO_CHANGE برمی‌گردد.
            return ProvisionAction(ProvisionActionType.NO_CHANGE)

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