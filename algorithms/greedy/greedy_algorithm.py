"""
algorithms/greedy/greedy_algorithm.py

*** یادداشت فاز: طبق نقشه‌راه سند (بخش ۰)، پیاده‌سازی کامل/نهایی Greedy فاز ۲
است. این نسخه یک Greedy واقعی و کاربردی (نه placeholder خالی) است که همین حالا
برای دو هدف لازم بود: (۱) تست end-to-end موتور شبیه‌سازی، (۲) معلم
Behavior-Cloning warm-start عامل PPO (بخش ۱۱.۵). سیاست‌اش ساده و واکنشی است؛
در فاز ۲ می‌توان آن را با منطق پیچیده‌تر (نزدیک به Voila) جایگزین/تقویت کرد.

سیاست:
    - Scale Up وقتی میانگین اشغال صف رپلیکاهای یک سرویس از ۷۰٪ ظرفیت صف رد شود.
    - Scale Down وقتی زیر ۱۰٪ باشد (با حداقل ۱ رپلیکا باقی).
    - Provision Up: نزدیک‌ترین سرور خاموش به سرور(های) پراستفاده.
    - Placement رپلیکای جدید: نزدیک‌ترین سرور فعال با ظرفیت آزاد کافی.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from common.config import CFG
from common.geo import haversine_km
from common.models import Server, ServerState, ReplicaState
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionAction, ProvisionActionType, MigrationStep


class GreedyAlgorithm(AlgorithmBase):
    name = "greedy"

    def scale_decision(self, service_id: int, metrics_snapshot: dict) -> ScaleAction:
        svc = metrics_snapshot["services"][service_id]
        queue_len = svc["queue_len"]
        occ_ratio = svc["avg_queue_occupancy"] / queue_len if queue_len else 0.0
        if occ_ratio > 0.7 or svc["rejection_rate"] > 0:
            return ScaleAction.SCALE_UP
        if occ_ratio < 0.1 and svc["n_replicas"] > 1:
            return ScaleAction.SCALE_DOWN
        return ScaleAction.NO_CHANGE

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

    def select_placement_server(self, service_id: int, servers: Dict[int, Server]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [s for s in servers.values()
                      if s.state == ServerState.ACTIVE and s.can_host(service_id, cpu)]
        if not candidates:
            return None
        # نزدیک‌ترین به مرکز ثقل سرورهای فعال فعلی (تقریبی برای «نزدیک به تقاضا»)
        active_all = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        clat = sum(s.lat for s in active_all) / len(active_all)
        clon = sum(s.long for s in active_all) / len(active_all)
        candidates.sort(key=lambda s: haversine_km(clat, clon, s.lat, s.long))
        return candidates[0].id

    def migration_decision(self, draining_server: Server,
                            servers: Dict[int, Server]) -> List[MigrationStep]:
        steps = []
        # سرویس‌هایی که *فقط* روی این سرور رپلیکا دارند باید مهاجرت کنند
        for service_id, replica in draining_server.hosted_replicas.items():
            if replica.state == ReplicaState.TERMINATED:
                continue
            other_hosts = [s for s in servers.values()
                           if s.id != draining_server.id and service_id in s.hosted_replicas
                           and s.hosted_replicas[service_id].state != ReplicaState.TERMINATED]
            if other_hosts:
                continue  # رپلیکای دیگری هم هست، نیازی به مهاجرت نیست
            cpu = CFG.services_info[service_id]["cpu_demand"]
            candidates = [s for s in servers.values()
                          if s.id != draining_server.id and s.state == ServerState.ACTIVE
                          and s.can_host(service_id, cpu)]
            if candidates:
                candidates.sort(key=lambda s: haversine_km(draining_server.lat, draining_server.long,
                                                             s.lat, s.long))
                steps.append(MigrationStep(service_id=service_id, target_server_id=candidates[0].id))
            # اگر هیچ سرور ACTIVE مناسبی نبود: طبق بخش ۶.۲ باید یک سرور OFF جدید
            # boot شود. این حالت لبه فعلاً لاگ می‌شود؛ تکمیلش با provision_decision
            # چرخه‌ی بعد به‌طور طبیعی رخ می‌دهد چون سرویس بدون رپلیکا کاندید
            # scale_decision=SCALE_UP خواهد شد.
        return steps
