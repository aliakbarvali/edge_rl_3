from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep


class GreedyAlgorithm(AlgorithmBase):
    name = "greedy"

    def scale_decision(self, service_id, metrics_snapshot):
        svc = metrics_snapshot["services"][service_id]
        queue_len = svc["queue_len"]
        occ_ratio = svc["avg_queue_occupancy"] / queue_len if queue_len else 0.0
        if occ_ratio > 0.7 or svc["rejection_rate"] > 0:
            return ScaleAction.SCALE_UP
        if occ_ratio < 0.1 and svc["n_replicas"] > 1:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE

    def provision_decision(self, servers, metrics_snapshot, now):
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
                    # *** اضافه‌بار utilization-محور نبود ولی capacity-starved
                    # بودیم؛ پرمشغول‌ترین سرور ACTIVE فعلی را مرجع جغرافیایی
                    # می‌گیریم.
                    ref = max(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
                else:
                    # *** لبه‌ی مرزی: هیچ سروری هنوز ACTIVE نیست (مثلاً همان
                    # اولین DECISION_TICK در t=0 که همه‌ی سرورهای اولیه هنوز
                    # BOOTING هستند ولی درخواست‌های همین لحظه چون replica
                    # READY ای نیست REJECTED_NO_REPLICA می‌خورند و سرویس را
                    # «starved» نشان می‌دهند). در این حالت معیار مکانی معناداری
                    # نداریم؛ فقط اولین سرور OFF را بدون اولویت جغرافیایی
                    # انتخاب می‌کنیم (به‌هرحال initial_placement به‌زودی چند
                    # سرور دیگر هم boot می‌کند).
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

    def select_placement_server(self, service_id, servers):
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None
        active_all = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        clat = sum(s.lat for s in active_all) / len(active_all)
        clon = sum(s.long for s in active_all) / len(active_all)
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
            cpu = CFG.services_info[service_id]["cpu_demand"]
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu)]
            if candidates:
                candidates.sort(key=lambda s: haversine_km(draining_server.lat, draining_server.long,
                                                             s.lat, s.long))
                steps.append(MigrationStep(service_id=service_id, target_server_id=candidates[0].id))
        return steps