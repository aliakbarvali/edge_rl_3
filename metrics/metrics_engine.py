"""
Metrics Engine — تجمیع خروجی dispatcher (per-request) به متریک‌های سطح
سرویس/سیستم که در main.py گزارش می‌شوند:
    avg/p95/p99_response_time_sec, deadline_violations(_rate_pct),
    avg_distance_km, avg/p95/p99_network_delay_ms, avg_load_balance_cv,
    E_pct (شاخص قدیمی سازگار با نسخه‌ی قبلی), cumulative_energy_joule
    (این یکی چون به ready_mask در طول زمان نیاز دارد، در simulator.py محاسبه
    و به این ماژول اضافه می‌شود، نه اینجا).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _percentile_stats(series: pd.Series) -> dict:
    valid = series.dropna()
    if len(valid) == 0:
        return {"avg": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "avg": float(valid.mean()),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
    }


def load_balance_cv(replica_load: dict[int, np.ndarray], placements: dict[int, np.ndarray]) -> float:
    """
    *** جدید: میانگین ضریب تغییرات (CV = std/mean) بار میان replicaهای *فعال*
    هر سرویس در این چرخه؛ عدد کوچک‌تر یعنی توزیع بار عادلانه‌تر بین replicaها.
    فقط سرویس‌هایی که >=۲ replica با بار>۰ دارند حساب می‌شوند (وگرنه CV بی‌معنی است).
    """
    cvs = []
    for sid, placement in placements.items():
        loads = replica_load.get(sid)
        if loads is None:
            continue
        active_loads = loads[placement]
        active_loads = active_loads[active_loads > 0]
        if len(active_loads) >= 2 and active_loads.mean() > 0:
            cvs.append(float(active_loads.std() / active_loads.mean()))
    return float(np.mean(cvs)) if cvs else 0.0


def per_service_metrics(outcomes: pd.DataFrame, service_id: int, n_replicas: int) -> dict:
    df = outcomes[outcomes.ServiceID == service_id]
    total = len(df)
    if total == 0:
        return {"ServiceID": service_id, "total_requests": 0, "E_pct": 0.0,
                "n_replicas": n_replicas, "avg_response_time_sec": 0.0,
                "p95_response_time_sec": 0.0, "p99_response_time_sec": 0.0,
                "deadline_violations": 0, "deadline_violation_rate_pct": 0.0,
                "avg_distance_km": 0.0, "avg_network_delay_ms": 0.0,
                "p95_network_delay_ms": 0.0, "p99_network_delay_ms": 0.0}

    slow = int(df.slow.sum())
    rt = _percentile_stats(df.response_time_sec)
    delay = _percentile_stats(df.latency_ms)
    dv = int(df.deadline_violation.sum())

    return {
        "ServiceID": service_id,
        "total_requests": total,
        "slow_requests": slow,
        "E_pct": 100.0 * slow / total,
        "n_replicas": n_replicas,
        "avg_response_time_sec": rt["avg"], "p95_response_time_sec": rt["p95"],
        "p99_response_time_sec": rt["p99"],
        "deadline_violations": dv, "deadline_violation_rate_pct": 100.0 * dv / total,
        "avg_distance_km": float(df.distance_km.dropna().mean()) if df.distance_km.notna().any() else 0.0,
        "avg_network_delay_ms": delay["avg"], "p95_network_delay_ms": delay["p95"],
        "p99_network_delay_ms": delay["p99"],
    }


def system_metrics(outcomes: pd.DataFrame, ready_mask: np.ndarray,
                    replica_load: dict[int, np.ndarray], placements: dict[int, np.ndarray]) -> dict:
    total = len(outcomes)
    if total == 0:
        return {"total_requests": 0, "E_pct": 0.0, "servers_on": int(ready_mask.sum()),
                "avg_response_time_sec": 0.0, "p95_response_time_sec": 0.0,
                "p99_response_time_sec": 0.0, "deadline_violations": 0,
                "deadline_violation_rate_pct": 0.0, "avg_distance_km": 0.0,
                "avg_network_delay_ms": 0.0, "p95_network_delay_ms": 0.0,
                "p99_network_delay_ms": 0.0, "avg_load_balance_cv": 0.0}

    slow = int(outcomes.slow.sum())
    rt = _percentile_stats(outcomes.response_time_sec)
    delay = _percentile_stats(outcomes.latency_ms)
    dv = int(outcomes.deadline_violation.sum())

    return {
        "total_requests": total,
        "slow_requests": slow,
        "E_pct": 100.0 * slow / total,
        "servers_on": int(ready_mask.sum()),
        "avg_response_time_sec": rt["avg"], "p95_response_time_sec": rt["p95"],
        "p99_response_time_sec": rt["p99"],
        "deadline_violations": dv, "deadline_violation_rate_pct": 100.0 * dv / total,
        "avg_distance_km": float(outcomes.distance_km.dropna().mean()) if outcomes.distance_km.notna().any() else 0.0,
        "avg_network_delay_ms": delay["avg"], "p95_network_delay_ms": delay["p95"],
        "p99_network_delay_ms": delay["p99"],
        "avg_load_balance_cv": load_balance_cv(replica_load, placements),
    }


def cycle_summary(cycle: int, outcomes: pd.DataFrame, placements: dict[int, np.ndarray],
                   ready_mask: np.ndarray) -> list[dict]:
    """یک سطر به ازای هر سرویس برای این چرخه (برای ساخت DataFrame لاگ کامل)."""
    rows = []
    for sid, placement in placements.items():
        m = per_service_metrics(outcomes, sid, int(placement.sum()))
        m["cycle"] = cycle
        m["servers_on"] = int(ready_mask.sum())
        rows.append(m)
    return rows
