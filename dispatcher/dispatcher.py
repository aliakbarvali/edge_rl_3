"""
Dispatcher — مسیریابی event-level هر درخواست + محاسبه‌ی متریک‌های کامل سطح
درخواست (response time، deadline، فاصله، تاخیر شبکه).

برخلاف `common/qos.py` (تخمین‌گر سریع و احتمالاتی-تحلیلی برای استفاده‌ی
*داخلی* هر service scaler هنگام جست‌وجوی بهترین چیدمان)، این ماژول واقعاً هر
درخواست را تک‌به‌تک، به ترتیب زمان وقوعش، به نزدیک‌ترین replica در دسترس
مسیریابی می‌کند و متریک نهایی/واقعی گزارش از همین‌جا به دست می‌آید.

*** مدل‌سازی response time (چون داده‌ی خام مستقیماً این را نمی‌دهد، این
فرمول‌ها یک انتخاب طراحی صریح‌اند که باید در گزارش نهایی ذکر شوند):

    network_delay_sec   = L[gateway, replica] / 1000                (از ماتریس تاخیر)
    execution_time_sec  = ServiceExecutionTime همان سرویس (ثابت داده‌شده در دیتاست)
    queueing_delay_sec  = (تعداد درخواست‌های همان سرویس که *قبل* از این یکی در
                           همین چرخه به همین replica رسیده‌اند) × execution_time_sec
                           (تقریب صف تک‌سروره‌ی غیر‌preemptive - M/D/1-like)
    response_time_sec   = network_delay_sec + queueing_delay_sec + execution_time_sec

    deadline_sec         = (lo_ms/1000) + execution_time_sec × DEADLINE_SLACK
    deadline_violation   = response_time_sec > deadline_sec  (یا اگر اصلاً replica نبود)

قانون مسیریابی هر درخواست:
    ۱) کاندیدها = گره‌هایی که این سرویس رویشان replica دارد و سرور «آماده» است.
    ۲) اگر کاندیدی نبود -> درخواست drop می‌شود (deadline_violation=True, بدون response_time).
    ۳) وگرنه نزدیک‌ترین کاندید از نظر تاخیر شبکه انتخاب می‌شود.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from common.config import CFG


class Dispatcher:
    def __init__(self, L: np.ndarray, D_km: np.ndarray, lo_ms: float,
                 service_profile: dict[int, dict]):
        """
        L               : ماتریس تاخیر شبکه (ms)
        D_km            : ماتریس فاصله‌ی جغرافیایی (km) - هم‌شکل با L
        lo_ms           : آستانه‌ی تاخیر مقاله (برای محاسبه‌ی latency_ok و deadline پایه)
        service_profile : dict {service_id: {"execution_time_sec": float, "resource": float}}
        """
        self.L = L
        self.D_km = D_km
        self.lo_ms = lo_ms
        self.profile = service_profile

    def process_cycle(self, events: pd.DataFrame, placements: dict[int, np.ndarray],
                       ready_mask: np.ndarray, co_per_service: dict[int, float]):
        n_nodes = len(ready_mask)
        replica_load = {sid: np.zeros(n_nodes) for sid in placements}
        records = []

        for row in events.itertuples(index=False):
            sid = int(row.ServiceID)
            gw = int(row.gateway_idx)
            placement = placements.get(sid)
            exec_time = self.profile.get(sid, {}).get("execution_time_sec", 0.0)
            deadline_sec = (self.lo_ms / 1000.0) + exec_time * CFG.DEADLINE_SLACK

            if placement is None:
                continue  # سرویسی که در این اجرا فعال نشده

            candidates = np.where(placement & ready_mask)[0]
            if len(candidates) == 0:
                # *** هیچ replica در دسترسی نبود: drop، همیشه نقض deadline
                records.append((sid, gw, -1, np.nan, np.nan, False, False,
                                 np.nan, deadline_sec, True, True))
                continue

            dists = self.L[gw, candidates]
            best = candidates[int(np.argmin(dists))]
            best_L = float(self.L[gw, best])
            best_D = float(self.D_km[gw, best])

            latency_ok = best_L <= self.lo_ms
            co = co_per_service.get(sid, np.inf)
            queue_position = replica_load[sid][best]        # تعداد صف جلوتر از این درخواست
            capacity_ok = queue_position < co
            replica_load[sid][best] += 1

            network_delay_sec = best_L / 1000.0
            queueing_delay_sec = queue_position * exec_time
            response_time_sec = network_delay_sec + queueing_delay_sec + exec_time
            deadline_violation = response_time_sec > deadline_sec

            slow = not (latency_ok and capacity_ok)
            records.append((sid, gw, best, best_L, best_D, latency_ok, capacity_ok,
                             response_time_sec, deadline_sec, deadline_violation, slow))

        outcomes = pd.DataFrame(records, columns=[
            "ServiceID", "gateway", "served_by", "latency_ms", "distance_km",
            "latency_ok", "capacity_ok", "response_time_sec", "deadline_sec",
            "deadline_violation", "slow",
        ])
        return outcomes, replica_load
