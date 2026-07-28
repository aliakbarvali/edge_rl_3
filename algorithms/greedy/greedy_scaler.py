"""
Baseline اول: Greedy ساده.

برخلاف Voila که چیدمان را بین چرخه‌ها نگه می‌دارد و به‌تدریج replacement /
scale-up / scale-down انجام می‌دهد (Procedures 4 تا 7)، این baseline در *هر*
چرخه از صفر یک چیدمان تازه با همان پوشش حریصانه‌ی Procedure 3 مقاله می‌سازد:
نزدیک‌ترین مجموعه‌ی گره‌ها که تمام گیت‌وی‌های فعال آن چرخه را از نظر latency
(lo) پوشش می‌دهند. هیچ توجهی به ظرفیت (co) یا تعداد replicaهای چرخه‌ی قبل
نمی‌شود. این baseline نشان می‌دهد Voila دقیقاً به خاطر منطق تطبیقی‌اش
(نه فقط به خاطر پوشش latency) عملکرد بهتری دارد.
"""

from __future__ import annotations
import numpy as np

from common.placement_utils import initial_placement


class GreedyScaler:
    """رابط یکسان با سایر اسکیلرها: متد decide(gateway_load) -> placement (bool array)."""
    name = "Greedy"

    def __init__(self, T: np.ndarray, n_nodes: int):
        self.T = T
        self.n_nodes = n_nodes

    def decide(self, gateway_load: np.ndarray, allowed_servers: np.ndarray | None = None) -> np.ndarray:
        active_idx = np.where(gateway_load > 0)[0]
        if len(active_idx) == 0:
            return np.zeros(self.n_nodes, dtype=bool)
        return initial_placement(active_idx, self.T, self.n_nodes, allowed_servers)
