from __future__ import annotations
import math
from typing import Dict, List, Optional

from common.config import CFG
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep

TARGET_UTILIZATION = 0.70


class HPAAlgorithm(AlgorithmBase):
    name = "hpa"

    # تصمیم عمدی (نه باگ): can_host بدون bts_lat/bts_long صدا زده می‌شود چون
    # HPA واقعی location-unaware است.

    def scale_decision(self, service_id, metrics_snapshot):
        sv = metrics_snapshot["services"][service_id]
        current_replicas = max(sv["n_ready_replicas"], 1)
        current_util = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

        if current_util <= 0 and sv["rejection_rate"] <= 0:
            desired = 1
        else:
            desired = math.ceil(current_replicas * (current_util / TARGET_UTILIZATION))
        desired = max(1, desired)

        if sv["rejection_rate"] > 0:
            desired = max(desired, current_replicas + 1)

        if desired > current_replicas:
            return ScaleAction.SCALE_UP
        if desired < current_replicas and current_replicas > 1:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE

    def select_placement_server(self, service_id, servers):
        cpu = CFG.services_info[service_id]["resource_mips"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.free_capacity()).id

    def provision_decision(self, servers, metrics_snapshot, now):
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        if not active:
            return ProvisionAction(ProvisionActionType.NO_CHANGE)

        avg_util = sum(metrics_snapshot["servers"][s.id]["utilization"] for s in active) / len(active)
        overloaded = [s for s in active
                      if metrics_snapshot["servers"][s.id]["utilization"] > CFG.util_scale_up_threshold]
        starved_services = self._capacity_starved_services(metrics_snapshot, servers, occ_threshold=0.7)

        if avg_util > CFG.util_scale_up_threshold or starved_services:
            off_servers = sorted([s for s in servers.values() if s.state == ServerState.OFF],
                                  key=lambda s: s.id)
            if off_servers:
                # *** رفع باگ ۴: هم‌راستا با greedy_algorithm.py - فقط overloadهای
                # واقعی پاس داده می‌شود، نه "overloaded or active".
                desired_profile = self._pick_profile_for_overload(overloaded, active[0].capacity)
                pool = self._filter_by_profile_with_fallback(off_servers, desired_profile)
                pool = sorted(pool, key=lambda s: s.id)
                return ProvisionAction(ProvisionActionType.TURN_ON, pool[0].id)
            # *** رفع باگ ۲ (fallthrough): هم‌راستا با Greedy/VOILA - دیگر به
            # بررسی TURN_OFF سقوط نمی‌کند وقتی سرور خاموشی برای روشن‌کردن نمانده.
            return ProvisionAction(ProvisionActionType.NO_CHANGE)

        if avg_util < CFG.util_scale_down_threshold:
            idle = min(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
            return ProvisionAction(ProvisionActionType.TURN_OFF, idle.id)

        return ProvisionAction(ProvisionActionType.NO_CHANGE)

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
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu)]
            if not candidates:
                continue
            best = max(candidates, key=lambda s: s.free_capacity())
            steps.append(MigrationStep(service_id=service_id, target_server_id=best.id))
        return steps