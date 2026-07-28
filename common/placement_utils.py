"""
توابع مشترک بین الگوریتم‌های Greedy و Voila؛ هر دو از همین «پوشش حریصانه‌ی
اولیه» (Procedure 3 مقاله) استفاده می‌کنند. Voila علاوه بر این از توابع
TBR/TBT هم برای replacement/scale-up/scale-down استفاده می‌کند (بخش IV-C).
نگه‌داشتن این توابع در یک‌جا از تکرار کد جلوگیری می‌کند و تضمین می‌کند که
مقایسه‌ی Greedy در برابر Voila واقعاً فقط تفاوت در "منطق تطبیقی" را نشان دهد،
نه تفاوت در پیاده‌سازی پوشش اولیه.
"""

from __future__ import annotations
import numpy as np


def initial_placement(active_idx: np.ndarray, T: np.ndarray, n_nodes: int,
                       allowed_servers: np.ndarray | None = None) -> np.ndarray:
    """
    Procedure 3: پوشش حریصانه‌ی گیت‌وی‌های فعال با کمترین تعداد replica ممکن.
    *** allowed_servers: ماسک بولی به طول n_nodes - اگر داده شود، فقط سرورهایی
    که server scaler روشن کرده (allowed_servers[i]=True) کاندید میزبانی می‌شوند.
    """
    if allowed_servers is None:
        allowed_servers = np.ones(n_nodes, dtype=bool)
    placement = np.zeros(n_nodes, dtype=bool)
    remaining = set(active_idx.tolist())
    while remaining:
        remaining_arr = np.array(sorted(remaining))
        coverage_counts = T[remaining_arr].sum(axis=0).astype(float)
        coverage_counts[placement] = -1       # گره‌های از قبل انتخاب‌شده را دوباره انتخاب نکن
        coverage_counts[~allowed_servers] = -1  # سرورهای خاموش اصلاً کاندید نیستند
        best_node = int(np.argmax(coverage_counts))
        if coverage_counts[best_node] <= 0:
            # هیچ سرور روشنی گیت‌وی‌های باقی‌مانده را پوشش نمی‌دهد؛ اگر سرور مجاز
            # دیگری باقی مانده از آن استفاده کن، وگرنه این گیت‌وی‌ها بدون پوشش می‌مانند
            candidates_left = np.where(allowed_servers & ~placement)[0]
            if len(candidates_left) == 0:
                break  # هیچ سرور روشن آزادی برای اضافه کردن باقی نمانده
            best_node = int(candidates_left[0])
        placement[best_node] = True
        newly_covered = remaining_arr[T[remaining_arr, best_node] == 1]
        remaining -= set(newly_covered.tolist())
    return placement


def uncovered_gateways(active_idx: np.ndarray, placement: np.ndarray, T: np.ndarray) -> np.ndarray:
    placed_idx = np.where(placement)[0]
    if len(placed_idx) == 0:
        return active_idx.copy()
    covered = np.any(T[np.ix_(active_idx, placed_idx)] == 1, axis=1)
    return active_idx[~covered]


def vital_replicas(active_idx: np.ndarray, placement: np.ndarray, T: np.ndarray) -> set:
    """replica «حیاتی» = تنها replica در محدوده lo برای حداقل یک گیت‌وی فعال (بخش IV-C)."""
    placed_idx = np.where(placement)[0]
    vital = set()
    if len(placed_idx) == 0 or len(active_idx) == 0:
        return vital
    coverage = T[np.ix_(active_idx, placed_idx)]
    for row in coverage:
        covering = placed_idx[row == 1]
        if len(covering) == 1:
            vital.add(int(covering[0]))
    return vital


def tbr_tbt_proximity(active_idx, placement, T, max_tbt, allowed_servers: np.ndarray | None = None):
    if allowed_servers is None:
        allowed_servers = np.ones(len(placement), dtype=bool)
    vital = vital_replicas(active_idx, placement, T)
    placed_idx = np.where(placement)[0]
    TBR = np.array([j for j in placed_idx if j not in vital], dtype=int)

    uncovered = uncovered_gateways(active_idx, placement, T)
    candidates = set()
    for i in uncovered:
        for c in np.where(T[i] == 1)[0]:
            if not placement[c] and allowed_servers[c]:  # *** فقط سرورهای روشن کاندید شوند
                candidates.add(int(c))
    TBT = np.array(sorted(candidates)[:max_tbt], dtype=int)
    return TBR, TBT


def tbr_tbt_saturation(placement, replica_loads, co_per_cycle, T, max_tbt,
                        allowed_servers: np.ndarray | None = None):
    if allowed_servers is None:
        allowed_servers = np.ones(len(placement), dtype=bool)
    placed_idx = np.where(placement)[0]
    order = np.argsort(replica_loads[placed_idx])  # کم‌بارترین‌ها اول
    TBR = placed_idx[order]

    overloaded = placed_idx[replica_loads[placed_idx] > co_per_cycle]
    candidates = set()
    for o in overloaded:
        for c in np.where(T[o] == 1)[0]:
            if not placement[c] and allowed_servers[c]:  # *** فقط سرورهای روشن کاندید شوند
                candidates.add(int(c))
    TBT = np.array(sorted(candidates)[:max_tbt], dtype=int)
    return TBR, TBT
