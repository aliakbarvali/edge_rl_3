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

# ثابت‌های نرمال‌سازی - بازکالیبره‌شده با calibrate_constants.py روی
# Data4.csv با Greedy، *بعد از* اعمال کل فیکس‌های موتور مشترک این دوره
# (DRAINING در utilization/power، capacity-starved با BOOTING، drain
# دینامیک، demand_centroid در لحظه‌ی ورود، ...). عدد انتخابی p95 است -
# مقداری که ۹۵٪ تیک‌ها زیر آن هستند.
_NORM_RESPONSE_TIME_SEC = 85.2   # *** بازکالیبره‌شده: p95 واقعی
                                   # avg_response_time_recent (n=2875 تیک
                                   # غیرصفر). مقدار قبلی (300.0) بدون
                                   # کالیبراسیون مستند و ~3.5 برابر بزرگ‌تر
                                   # از نیاز واقعی بود - این بُعد از
                                   # state/reward عملاً همیشه دور از سقف
                                   # کلمپ می‌ماند، یعنی کم‌تمایز بود.
_NORM_ENERGY_JOULE = 20_843.65   # *** بازکالیبره‌شده: p95 واقعی
                                   # energy_recent_joule (n=2882 تیک).
                                   # تقریباً ۲ برابر مقدار قبلی (12000) -
                                   # طبیعی و منتظره چون فیکس شمول DRAINING
                                   # در instantaneous_utilization/power
                                   # مصرف واقعی هر تیک را بالاتر نشان می‌دهد.
_NORM_ARRIVAL_RATE = 3.0         # *** بازکالیبره‌شده: p95 واقعی recent_arrivals
                                   # (هر سرویس، هر تیک؛ n=43230). مقدار
                                   # قبلی (20.0) بیش از ۶ برابر بزرگ‌تر از
                                   # نیاز واقعی بود (mean واقعی فقط ۰.۷۹) -
                                   # این بُعد از state تقریباً همیشه نزدیک
                                   # صفر و عملاً بی‌فایده برای عامل بود.

# *** رفع باگ مسدودکننده (کشف‌شده بعد از افت کیفیت PPO با seed=42): این سه
# ثابت قبلاً *همزمان* در algorithms/ppo/env.py (برای محاسبه‌ی reward) با
# مقادیر قدیمی و کالیبره‌نشده (300.0 و 12_000.0) کپی/هاردکد شده بودند. وقتی
# این‌جا بازکالیبره شدند (calibrate_constants.py)، آن کپی هرگز به‌روزرسانی
# نشد - یعنی state vector (این فایل) واقعیت را با مقیاس درست می‌دید، ولی
# reward (env.py) هنوز با مخرج‌های ۱.۷۴ تا ۳.۵ برابر بزرگ‌تر از مقیاس واقعی
# محاسبه می‌شد و عملاً هرگز به سقف کلمپ نمی‌رسید - یعنی سهم response_time و
# energy در reward تقریباً بی‌اثر شده بود، دقیقاً همان چیزی که به عامل اجازه
# داد avg_active_servers را به ~1.16 برساند و rejection را بدون جریمه‌ی
# مؤثر بالا ببرد. برای اینکه این دو دیگر هرگز از هم جدا نیفتند، این ثابت‌ها
# public export می‌شوند و algorithms/ppo/env.py مستقیماً از همین‌جا import
# می‌کند - نه یک کپی مجزا.
NORM_RESPONSE_TIME_SEC = _NORM_RESPONSE_TIME_SEC
NORM_ENERGY_JOULE = _NORM_ENERGY_JOULE
NORM_ARRIVAL_RATE = _NORM_ARRIVAL_RATE


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