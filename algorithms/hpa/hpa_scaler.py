"""
Baseline دوم: Horizontal Pod Autoscaler کلاسیک کوبرنتیز.

برخلاف Voila، HPA پیش‌فرض کوبرنتیز:
  - فقط از utilization (نسبت بار کل به ظرفیت کل) استفاده می‌کند، نه latency.
  - فرمول استاندارد HPA:
        desiredReplicas = ceil(currentReplicas * (currentUtilization / targetUtilization))
    (مستندات رسمی Kubernetes، همانی که در مقدمه به عنوان baseline معرفی مقاله ازش
    یاد می‌شود: "Kubernetes' default filtering/scoring algorithms" که location-unaware اند.)
  - جایگذاری replicaها توسط scheduler پیش‌فرض کوبرنتیز صورت می‌گیرد که به تاخیر
    شبکه توجهی ندارد؛ اینجا این رفتار را با یک ترتیب ثابت و از‌پیش‌تعیین‌شده
    (latency-unaware) از گره‌ها شبیه‌سازی می‌کنیم.
"""

from __future__ import annotations
import math
import numpy as np

from common.config import CFG


class HPAScaler:
    name = "HPA"

    def __init__(self, n_nodes, co_per_cycle, target_util=CFG.HPA_TARGET_UTIL,
                 min_replicas=CFG.HPA_MIN_REPLICAS, cooldown=CFG.HPA_COOLDOWN_CYCLES,
                 seed=CFG.SEED):
        self.n_nodes = n_nodes
        self.co = co_per_cycle
        self.target_util = target_util
        self.min_replicas = min_replicas
        self.cooldown = cooldown
        # ترتیب ثابت و latency-unaware برای انتخاب گره‌ها (معادل رفتار scheduler
        # پیش‌فرض کوبرنتیز که مکان را در نظر نمی‌گیرد)
        rng = np.random.default_rng(seed)
        self.node_order = rng.permutation(n_nodes)
        self.placement = np.zeros(n_nodes, dtype=bool)
        self.cycles_since_last_change = self.cooldown

    def _set_replica_count(self, desired: int, allowed_servers: np.ndarray | None = None):
        desired = max(self.min_replicas, desired)
        current_idx = list(np.where(self.placement)[0])
        if allowed_servers is not None:
            # *** replicaهایی که روی سرور خاموش‌شده افتاده‌اند دیگر معتبر نیستند
            current_idx = [n for n in current_idx if allowed_servers[n]]
        if desired > len(current_idx):
            for node in self.node_order:
                if len(current_idx) >= desired:
                    break
                if allowed_servers is not None and not allowed_servers[node]:
                    continue  # *** سرور خاموش کاندید نیست
                if node not in current_idx:
                    current_idx.append(int(node))
        elif desired < len(current_idx):
            # آخرین‌های اضافه‌شده (بر اساس ترتیب latency-unaware) اول حذف می‌شوند
            order_rank = {n: r for r, n in enumerate(self.node_order)}
            current_idx.sort(key=lambda n: order_rank[n])
            current_idx = current_idx[:desired]
        new_placement = np.zeros(self.n_nodes, dtype=bool)
        new_placement[current_idx] = True
        self.placement = new_placement

    def decide(self, gateway_load: np.ndarray, allowed_servers: np.ndarray | None = None) -> np.ndarray:
        total_load = gateway_load.sum()
        current_replicas = max(1, int(self.placement.sum()))

        if not self.placement.any():
            # اولین چرخه: به همان تعداد replica اولیه‌ی Voila شروع می‌کنیم تا مقایسه منصفانه باشد
            self._set_replica_count(self.min_replicas, allowed_servers)
            current_replicas = max(1, int(self.placement.sum()))

        if self.cycles_since_last_change < self.cooldown:
            self.cycles_since_last_change += 1
            return self.placement

        current_util = total_load / (current_replicas * self.co) if self.co > 0 else 0
        if current_util <= 0:
            desired = self.min_replicas
        else:
            desired = math.ceil(current_replicas * (current_util / self.target_util))
        desired = max(self.min_replicas, desired)

        if desired != current_replicas:
            self._set_replica_count(desired, allowed_servers)
            self.cycles_since_last_change = 0
        else:
            self.cycles_since_last_change += 1
        return self.placement
