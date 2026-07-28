"""
common/state_builder.py
طبق بخش ۱۱.۲ سند: «تابع build_state_vector() باید در یک ماژول مستقل نوشته
شود و هم توسط simulator/engine.py (از طریق algorithms/ppo/env.py) هم توسط
k8s_adapter/ (فاز ۳) فراخوانی شود؛ هیچ منطق ساخت state نباید در جای دیگر
تکرار/بازنویسی شود.»

بردار state (بخش ۱۱.۲):
    برای هر سرور (۱۰ تا): state one-hot (۴) + utilization (۱) + n_replicas (۱) = ۶ × ۱۰ = ۶۰
    برای هر سرویس (۱۵ تا): n_active_replicas (۱) + میانگین نسبت اشغال صف (۱) +
                             نرخ اخیر نقض deadline (۱) + نرخ ورودی اخیر (۱) = ۴ × ۱۵ = ۶۰
    سراسری: میانگین response_time اخیر (۱) + انرژی مصرفی اخیر (۱) = ۲
    مجموع = ۱۲۲ بعد.
"""

from __future__ import annotations
import numpy as np

from common.config import CFG
from common.models import ServerState

STATE_DIM = CFG.n_servers * 6 + CFG.n_services * 4 + 2

_SERVER_STATE_ORDER = [ServerState.OFF, ServerState.BOOTING, ServerState.ACTIVE, ServerState.DRAINING]

# ثابت‌های نرمال‌سازی (مقادیر معقول پیش‌فرض؛ در صورت نیاز کالیبره کنید - بخش ۱۳ سند)
_NORM_RESPONSE_TIME_SEC = 300.0
_NORM_ENERGY_JOULE = 12_000.0  # *** کالیبره‌شده با میانگین/حداکثر واقعی انرژی هر تیک
# (اندازه‌گیری‌شده روی Data4.csv با Greedy: میانگین~۸۲۰۰J، صدک۹۰~۱۰۲۰۰J،
# حداکثر~۱۲۷۰۰J؛ مقدار قبلی ۵۰۰۰۰J خیلی بزرگ بود و باعث می‌شد جریمه‌ی
# انرژی در reward/observation عملاً بی‌اثر شود - نگاه کنید به بحث کالیبراسیون).
_NORM_ARRIVAL_RATE = 20.0


def build_state_vector(snapshot: dict, servers: dict) -> np.ndarray:
    parts = []

    for sid in sorted(CFG.server_info.keys()):
        s_snap = snapshot["servers"][sid]
        one_hot = [1.0 if s_snap["state"] == st else 0.0 for st in _SERVER_STATE_ORDER]
        n_replicas = len(servers[sid].hosted_replicas)
        parts.extend(one_hot)
        parts.append(float(s_snap["utilization"]))
        parts.append(n_replicas / 15.0)  # نرمال‌شده با حداکثر نظری (۱۵ سرویس)

    for svc_id in CFG.active_services:
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
        parts.append(sv["n_replicas"] / CFG.n_servers)
        parts.append(min(occ_ratio, 2.0) / 2.0)
        parts.append(sv["deadline_violation_rate"])
        parts.append(min(sv["recent_arrivals"] / _NORM_ARRIVAL_RATE, 2.0) / 2.0)

    g = snapshot["global"]
    parts.append(min(g["avg_response_time_recent"] / _NORM_RESPONSE_TIME_SEC, 2.0) / 2.0)
    parts.append(min(g["energy_recent_joule"] / _NORM_ENERGY_JOULE, 2.0) / 2.0)

    vec = np.array(parts, dtype=np.float32)
    assert vec.shape[0] == STATE_DIM, f"state dim mismatch: {vec.shape[0]} != {STATE_DIM}"
    return vec