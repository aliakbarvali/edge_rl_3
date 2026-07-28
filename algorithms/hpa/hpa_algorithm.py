"""
algorithms/hpa/hpa_algorithm.py

پیاده‌سازی فاز ۲: الگوریتم مبتنی بر Kubernetes HPA استاندارد. طبق مقدمه‌ی
مقاله‌ی Voila، HPA/scheduler پیش‌فرض کوبرنتیز «location-unaware» است - یعنی
هیچ تصمیمی (نه scale، نه provision، نه placement) بر اساس فاصله‌ی جغرافیایی
گرفته نمی‌شود؛ فقط بر پایه‌ی utilization/ظرفیت آزاد است. این دقیقاً همان
تمایزی است که این الگوریتم را از Voila (location-aware) متمایز می‌کند.

فرمول استاندارد HPA (Kubernetes docs):
    desiredReplicas = ceil(currentReplicas * (currentUtilization / targetUtilization))
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

from common.config import CFG
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep

TARGET_UTILIZATION = 0.70  # مقدار پیش‌فرض معمول HPA واقعی کوبرنتیز


class HPAAlgorithm(AlgorithmBase):
    name = "hpa"

    # ------------------------------------------------------------------
    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        sv = metrics_snapshot["services"][service_id]
        current_replicas = max(sv["n_replicas"], 1)
        current_util = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

        if current_util <= 0 and sv["rejection_rate"] <= 0:
            desired = 1
        else:
            desired = math.ceil(current_replicas * (current_util / TARGET_UTILIZATION))
        desired = max(1, desired)

        if sv["rejection_rate"] > 0:  # درخواست رد شده -> قطعاً کمبود ظرفیت
            desired = max(desired, current_replicas + 1)

        if desired > current_replicas:
            return ScaleAction.SCALE_UP
        if desired < current_replicas and current_replicas > 1:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE

    # ------------------------------------------------------------------
    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        # *** latency-unaware: فقط بیشترین ظرفیت آزاد (bin-packing best-fit
        # پیش‌فرض scheduler کوبرنتیز)، بدون هیچ ترجیح جغرافیایی.
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.free_capacity()).id

    # ------------------------------------------------------------------
    def provision_decision(self, servers: Dict[int, Server], metrics_snapshot: dict,
                            now: float) -> ProvisionAction:
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        if not active:
            return ProvisionAction(ProvisionActionType.NO_CHANGE)

        avg_util = sum(metrics_snapshot["servers"][s.id]["utilization"] for s in active) / len(active)

        if avg_util > CFG.util_scale_up_threshold:
            off_servers = sorted([s for s in servers.values() if s.state == ServerState.OFF],
                                  key=lambda s: s.id)  # *** ترتیب ثابت/دلخواه، نه بر اساس مکان
            if off_servers:
                return ProvisionAction(ProvisionActionType.TURN_ON, off_servers[0].id)

        if avg_util < CFG.util_scale_down_threshold:
            idle = min(active, key=lambda s: metrics_snapshot["servers"][s.id]["utilization"])
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
            # *** latency-unaware: بیشترین ظرفیت آزاد، نه نزدیک‌ترین
            best = max(candidates, key=lambda s: s.free_capacity())
            steps.append(MigrationStep(service_id=service_id, target_server_id=best.id))
        return steps
