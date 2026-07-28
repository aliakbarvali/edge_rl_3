"""
پیاده‌سازی Procedure 1 (Probability matrix) و Procedure 2 (Test matrix) و
فرمول‌های محاسبه‌ی E% در بخش IV-B مقاله.

نکته‌ی مهم درباره‌ی P:
مقاله می‌گوید P توسط سیستم Proxy-mity (کار جداگانه‌ای [7]) و به صورت "یک تابع
نزولی یکنوا از تاخیر تخمینی" ساخته می‌شود، اما فرمول دقیق را در همین مقاله
نمی‌دهد. اینجا از یک softmax نزولی نمایی روی تاخیر استفاده می‌کنیم که دقیقاً
همان دو خاصیت ریاضی ذکرشده در مقاله را برآورده می‌کند:
    0 <= p_ij <= 1  و  sum_j p_ij = 1
با pij که هرچه lij کوچک‌تر باشد بزرگ‌تر است (نزدیک‌ترها احتمال بیشتری می‌گیرند).
"""

from __future__ import annotations
import numpy as np

from common.config import CFG


def build_probability_matrix(L: np.ndarray, temperature: float = CFG.P_TEMPERATURE) -> np.ndarray:
    """
    P[i,j] = احتمال اینکه یک درخواست از گیت‌وی i به گره j مسیریابی شود،
    به فرض این‌که همه‌ی گره‌ها میزبان یک replica باشند (تعریف مقاله، بخش IV-A).
    softmax(-L/temperature) روی هر سطر: هرچه temperature کوچک‌تر باشد، مسیریابی
    شدیدتر به سمت نزدیک‌ترین گره متمرکز می‌شود.
    """
    scores = -L / temperature
    scores = scores - scores.max(axis=1, keepdims=True)  # پایداری عددی
    exp_scores = np.exp(scores)
    P = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return P


def build_test_matrix(L: np.ndarray, lo_ms: float = CFG.LO_MS) -> np.ndarray:
    """Procedure 2: T[i,j] = 1 اگر l_ij <= lo وگرنه 0."""
    return (L <= lo_ms).astype(np.int8)


def evaluate_placement(placement: np.ndarray, gateway_load: np.ndarray,
                        P: np.ndarray, L: np.ndarray,
                        co_per_cycle: float, lo_ms: float = CFG.LO_MS):
    """
    Procedure 1 (ادامه): برای یک چیدمان مشخص (placement = آرایه‌ی بولی/ایندکس
    گره‌های میزبانِ replica)، درصد درخواست‌های کند E% را حساب می‌کند.

    ورودی‌ها:
        placement      : آرایه بولی به طول n (True یعنی آن گره یک replica دارد)
        gateway_load    : آرایه به طول n با بار هر گره در چرخه‌ی جاری (gi.load)
        P, L            : ماتریس‌های احتمال و تاخیر کامل n×n
        co_per_cycle    : آستانه ظرفیت هر replica در این چرخه (co * tau)

    خروجی: dict شامل E_pct, Vlo, Vco, active_gateways, replica_loads
    """
    active = gateway_load > 0
    total_load = gateway_load.sum()
    if total_load == 0 or not placement.any():
        return {"E_pct": 0.0 if total_load == 0 else 100.0,
                "Vlo": 0.0, "Vco": 0.0,
                "replica_loads": np.zeros_like(gateway_load)}

    # --- Procedure 1: انتخاب ستون‌های placement، حذف سطرهای idle، نرمال‌سازی ---
    P_masked = P.copy()
    P_masked[:, ~placement] = 0.0
    row_sums = P_masked.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # جلوگیری از تقسیم بر صفر برای گیت‌وی‌های بدون مسیر مجاز
    P_hat = P_masked / row_sums

    active_idx = np.where(active)[0]
    loads_active = gateway_load[active_idx]
    P_active = P_hat[active_idx]  # (|Ĝ|, n)
    L_active = L[active_idx]      # (|Ĝ|, n) - همان سطرها برای محاسبه f1

    # --- V_lo: مجموع بارِ مسیریابی‌شده به سمت گره‌های با تاخیر > lo ---
    far_mask = (L_active > lo_ms).astype(np.float64)
    Vlo = (P_active * far_mask * loads_active[:, None]).sum()

    # --- بار واقعیِ هر replica (برای V_co) ---
    replica_loads_active = P_active * loads_active[:, None]  # (|Ĝ|, n)
    replica_loads = replica_loads_active.sum(axis=0)          # (n,)

    # --- V_co: مازاد بار روی هر replica نسبت به ظرفیت ---
    overload = np.clip(replica_loads - co_per_cycle, 0, None)
    overload[~placement] = 0.0
    Vco = overload.sum()

    E_pct = 100.0 * (Vlo + Vco) / total_load
    return {"E_pct": E_pct, "Vlo": Vlo, "Vco": Vco, "replica_loads": replica_loads}


if __name__ == "__main__":
    from common.data_loader import prepare_dataset, cycle_loads_for_service
    from common.latency import build_latency_matrix
    from common.config import CFG

    gateways, servers, events, _ = prepare_dataset()
    L = build_latency_matrix(gateways.Lat.values, gateways.Long.values,
                              servers.Lat.values, servers.Long.values)
    P = build_probability_matrix(L)
    T = build_test_matrix(L)
    loads = cycle_loads_for_service(events, CFG.ACTIVE_SERVICES[0], len(gateways))

    placement = np.zeros(len(servers), dtype=bool)
    placement[:5] = True  # یک چیدمان آزمایشی با ۵ replica در ۵ سرور اول
    cycle = np.argmax(loads.sum(axis=1))  # پرترافیک‌ترین چرخه روز
    result = evaluate_placement(placement, loads[cycle], P, L, co_per_cycle=5)
    print(f"چرخه پرترافیک‌ترین: {cycle}, بار کل={loads[cycle].sum():.0f}")
    print(f"E%={result['E_pct']:.2f}  Vlo={result['Vlo']:.2f}  Vco={result['Vco']:.2f}")
