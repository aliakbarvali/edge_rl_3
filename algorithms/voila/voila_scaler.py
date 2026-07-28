"""
پیاده‌سازی کامل الگوریتم Voila طبق بخش IV مقاله:
    Procedure 3: Initial replica placement  (common/placement_utils.py)
    Procedure 4: QoS check                  (تشخیص نوع تخلف: proximity/saturation)
    Procedure 5: Replica replacement        (جابه‌جایی replica بدون تغییر تعداد)
    Procedure 6: Scale up                   (افزودن replica جدید)
    Procedure 7: Scale down                 (حذف replica اضافی وقتی چند چرخه تخلفی نبوده)

تفاوت کلیدی این الگوریتم با Greedy (algorithms/greedy): Voila حالت (چیدمان فعلی)
را بین چرخه‌ها نگه می‌دارد و آن را به‌صورت تدریجی و هوشمند اصلاح می‌کند، در حالی
که Greedy در هر چرخه از صفر یک چیدمان تازه می‌سازد.
"""

from __future__ import annotations
import numpy as np

from common.config import CFG
from common.qos import evaluate_placement
from common.placement_utils import (
    initial_placement, tbr_tbt_proximity, tbr_tbt_saturation,
)


def qos_check(placement, gateway_load, P, L, co_per_cycle, eo_pct):
    """Procedure 4."""
    res = evaluate_placement(placement, gateway_load, P, L, co_per_cycle)
    if res["E_pct"] <= eo_pct:
        return "none", res
    return ("proximity" if res["Vco"] <= res["Vlo"] else "saturation"), res


def try_replacement(placement, TBR, TBT, gateway_load, P, L, co_per_cycle, eo_pct):
    """Procedure 5."""
    best_placement, best_E = None, None
    for i in TBR:
        for j in TBT:
            trial = placement.copy()
            trial[i] = False
            trial[j] = True
            res = evaluate_placement(trial, gateway_load, P, L, co_per_cycle)
            if best_E is None or res["E_pct"] < best_E:
                best_E, best_placement = res["E_pct"], trial
    if best_placement is not None and best_E < eo_pct:
        return best_placement
    return None


def scale_up(placement, TBT, gateway_load, P, L, co_per_cycle, eo_pct):
    """Procedure 6."""
    placement = placement.copy()
    remaining = list(TBT)
    while remaining:
        res = evaluate_placement(placement, gateway_load, P, L, co_per_cycle)
        if res["E_pct"] <= eo_pct:
            break
        best_j, best_E = None, None
        for j in remaining:
            trial = placement.copy()
            trial[j] = True
            r = evaluate_placement(trial, gateway_load, P, L, co_per_cycle)
            if best_E is None or r["E_pct"] < best_E:
                best_E, best_j = r["E_pct"], j
        if best_j is None:
            break
        placement[best_j] = True
        remaining.remove(best_j)
    return placement


def scale_down(placement, gateway_load, P, L, co_per_cycle, eo_pct):
    """Procedure 7."""
    placement = placement.copy()
    improved = True
    while improved:
        improved = False
        placed_idx = np.where(placement)[0]
        if len(placed_idx) <= 1:
            break
        best_j, best_E = None, None
        for j in placed_idx:
            trial = placement.copy()
            trial[j] = False
            r = evaluate_placement(trial, gateway_load, P, L, co_per_cycle)
            if r["E_pct"] < eo_pct and (best_E is None or r["E_pct"] < best_E):
                best_E, best_j = r["E_pct"], j
        if best_j is not None:
            placement[best_j] = False
            improved = True
    return placement


class VoilaScaler:
    """رابط یکسان با سایر اسکیلرها: متد decide(gateway_load) -> placement (bool array)."""
    name = "Voila"

    def __init__(self, P, L, T, co_per_cycle, eo_pct=CFG.EO_PCT, safety=CFG.SAFETY,
                 max_tbt=CFG.MAX_TBT_CANDIDATES, patience=CFG.SCALE_DOWN_PATIENCE):
        self.P, self.L, self.T = P, L, T
        self.n_servers = P.shape[1]  # *** تعداد سرور = ستون‌های P/T/L (نه تعداد گیت‌وی‌ها)
        self.co = co_per_cycle
        self.eo = eo_pct
        self.safety = safety
        self.max_tbt = max_tbt
        self.patience = patience
        self.placement = None
        self.no_violation_streak = 0

    def decide(self, gateway_load: np.ndarray, allowed_servers: np.ndarray | None = None) -> np.ndarray:
        if allowed_servers is None:
            allowed_servers = np.ones(self.n_servers, dtype=bool)
        active_idx = np.where(gateway_load > 0)[0]

        # *** کف تضمینی: همان پوشش حریصانه‌ی کاملی که Greedy هر چرخه از صفر
        # می‌سازد (Procedure 3، فقط با سرورهای در دسترس). این تضمین می‌کند
        # Voila هرگز از نظر proximity بدتر از Greedy نشود - قبلاً چون Voila
        # فقط به‌صورت تدریجی و بر اساس تخمین داخلی‌اش گسترش می‌یافت، وقتی
        # سرور کم بود (۲-۵ تا) محافظه‌کارتر از Greedy عمل می‌کرد و همین باعث
        # می‌شد از Greedy در QoS واقعی عقب بیفتد.
        coverage_baseline = initial_placement(active_idx, self.T, self.n_servers, allowed_servers)

        if self.placement is None:
            self.placement = coverage_baseline.copy()
        else:
            # سرورهای خاموش‌شده را از چیدمان قبلی حذف کن، بعد کف تضمینی را اضافه کن
            self.placement = (self.placement & allowed_servers) | coverage_baseline

        # حاشیه‌ی امنیت: ظرفیت موثر کوچک‌تر برای فعال کردن scale-up زودتر (بخش V-C)
        co_eff = self.co * (1 - self.safety)
        vtype, res = qos_check(self.placement, gateway_load, self.P, self.L, co_eff, self.eo)

        if vtype == "none":
            self.no_violation_streak += 1
            if self.no_violation_streak >= self.patience:
                reduced = scale_down(self.placement, gateway_load, self.P, self.L,
                                      self.co, self.eo)
                self.placement = reduced | coverage_baseline  # *** هرگز کمتر از کف تضمینی
                self.no_violation_streak = 0
            return self.placement

        self.no_violation_streak = 0
        if vtype == "proximity":
            TBR, TBT = tbr_tbt_proximity(active_idx, self.placement, self.T, self.max_tbt,
                                          allowed_servers)
        else:
            TBR, TBT = tbr_tbt_saturation(self.placement, res["replica_loads"], co_eff,
                                           self.T, self.max_tbt, allowed_servers)

        if len(TBT) > 0:
            new_placement = None
            if len(TBR) > 0:
                new_placement = try_replacement(self.placement, TBR, TBT, gateway_load,
                                                 self.P, self.L, co_eff, self.eo)
            if new_placement is not None:
                self.placement = new_placement
            else:
                self.placement = scale_up(self.placement, TBT, gateway_load, self.P, self.L,
                                           co_eff, self.eo)

        self.placement = self.placement | coverage_baseline  # *** تضمین نهایی قبل از بازگشت
        return self.placement