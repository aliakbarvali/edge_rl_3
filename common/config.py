from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# منابع علمی این پیکربندی:
# - Calheiros et al., "CloudSim: a toolkit for modeling and simulation of cloud
#   computing environments", 2011 — واحد MIPS برای ظرفیت محاسباتی
# - Gupta et al., "iFogSim: A toolkit for modeling and simulation of resource
#   management techniques in Internet of Things, Edge and Fog computing
#   environments", 2017 — سطح‌بندی گره‌های fog، مدل AppModule/AppEdge
# - Sonmez et al., "EdgeCloudSim: An environment for performance evaluation of
#   edge computing systems", 2018 — تعمیم چندلایه
# - 3GPP TS 23.501 §5.7.4 (جدول 5QI) — استاندارد Packet Delay Budget
# - Little's Law (M/M/1/K queueing) — برای اشتقاق قابل‌توجیه queue_len
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# بخش ۱.۱: پروفایل‌ها و سرورها (heterogeneous، ثابت - مکان‌یابی قبلاً با solver انجام شده)
# واحد ظرفیت: MIPS (Million Instructions Per Second) طبق CloudSim/iFogSim/EdgeCloudSim
# capacity_mips = n_cores × mips_per_core  (تأییدیه: 4×750=3000, 4×2500=10000, 8×3750=30000)
# ---------------------------------------------------------------------------

SERVER_PROFILES: Dict[str, dict] = {
    # کلاس RPi4 / IoT gateway — 4 هسته‌ی ARM Cortex-A72
    "edge_small": {"n_cores": 4, "mips_per_core": 750,  "capacity_mips": 3000,  "p_idle": 40,  "p_max": 130},
    # کلاس Intel NUC / Xeon-D کم‌مصرف — 4 هسته
    "medium":     {"n_cores": 4, "mips_per_core": 2500, "capacity_mips": 10000, "p_idle": 70,  "p_max": 220},
    # کلاس MEC رک‌مانت چندهسته — 8 هسته
    "large":      {"n_cores": 8, "mips_per_core": 3750, "capacity_mips": 30000, "p_idle": 110, "p_max": 380},
}

_CAPACITY_TO_PROFILE = {p["capacity_mips"]: name for name, p in SERVER_PROFILES.items()}

# *** bts_id هرکدام = نزدیک‌ترین BTSID واقعی دیتاست به مختصات داده‌شده (محاسبه‌شده
# یک‌بار روی هر ۴ روز؛ همه زیر ۱ کیلومتر فاصله دارند - نگاه کنید به یادداشت توسعه).
# capacity_mips: ظرفیت کل سرور = n_cores × mips_per_core (طبق SERVER_PROFILES بالا)
SERVER_INFO: Dict[int, dict] = {
    1: {"bts_id": 1498, "lat": 31.37, "long": 121.25, "capacity_mips": 3000},
    2: {"bts_id": 777,  "lat": 31.31, "long": 121.51, "capacity_mips": 3000},
    3: {"bts_id": 530,  "lat": 31.10, "long": 121.18, "capacity_mips": 3000},
    4: {"bts_id": 121,  "lat": 31.25, "long": 121.37, "capacity_mips": 3000},
    5: {"bts_id": 292,  "lat": 31.10, "long": 121.36, "capacity_mips": 10000},
    6: {"bts_id": 1344, "lat": 31.04, "long": 121.74, "capacity_mips": 10000},
    7: {"bts_id": 182,  "lat": 31.17, "long": 121.57, "capacity_mips": 10000},
    8: {"bts_id": 505,  "lat": 31.15, "long": 121.41, "capacity_mips": 10000},
    9: {"bts_id": 419,  "lat": 31.20, "long": 121.43, "capacity_mips": 30000},
    10: {"bts_id": 609, "lat": 31.16, "long": 121.49, "capacity_mips": 30000},
}
for _sid, _info in SERVER_INFO.items():
    _info["profile"] = _CAPACITY_TO_PROFILE[_info["capacity_mips"]]

N_SERVERS = len(SERVER_INFO)

# ---------------------------------------------------------------------------
# بخش ۱.۲: سرویس‌ها (ثابت) — استاندارد MIPS/MI + 3GPP 5QI
#
# task_length_mi: اندازه‌ی محاسباتی درخواست (Million Instructions) — همان
#                 مقدار cpu_demand قدیمی، بدون تغییر عددی، فقط با معنای
#                 فیزیکی صریح (MI واقعی، نه واحد انتزاعی).
# resource_mips: سهمیه‌ی رزرو‌شده‌ی این سرویس از ظرفیت هر میزبان، **صرفاً
#                برای پذیرش/هم‌مکانی چند سرویس روی یک سرور** (قید:
#                sum(resource_mips of replicas on server) <= server.capacity_mips).
#                این یک کوانتای منطقیِ زمان‌بندی است (دقیقاً مثل CPU
#                request در Kubernetes)، **کاملاً مستقل از سرعت واقعی
#                اجرا** — چون سرعت واقعی هر هسته بین کلاس‌های سرور متفاوت
#                است (750/2500/3750 MIPS-per-core)، در حالی که رزرو
#                منطقی (میلی‌کور) مستقل از نوع سخت‌افزار میزبان تعریف
#                می‌شود.
# exec_time دیگر اینجا ذخیره نمی‌شود؛ همیشه در لحظه‌ی جای‌گذاری رپلیکا از
#           compute_exec_time_sec(service_id, server.capacity_mips) محاسبه
#           می‌شود = task_length_mi / سرعت واقعی MIPS همان سرور میزبان
#           (تک‌منبع حقیقت، وابسته به میزبان طبق رابطه‌ی فیزیکی
#           CloudSim/iFogSim: exec_time = MI / MIPS تخصیص‌یافته).
# deadline: بر اساس جدول 5QI (Packet Delay Budget) از 3GPP TS 23.501 §5.7.4
# queue_len: از M/M/1/K + قانون Little (N_target ≈ λ_avg × W_max)
#
# اعتبارسنجی: مجموع resource_mips هر ۱۵ سرویس ≈ 21200 MIPS < 30000 (ظرفیت large)
# ---------------------------------------------------------------------------
SERVICES_INFO: Dict[int, dict] = {
    1:  {"resource_mips": 200,  "task_length_mi": 30,     "queue_len": 1,  "deadline": 0.010, "memory": "32Mi"},
    2:  {"resource_mips": 200,  "task_length_mi": 50,     "queue_len": 2,  "deadline": 0.015, "memory": "40Mi"},
    3:  {"resource_mips": 400,  "task_length_mi": 100,    "queue_len": 2,  "deadline": 0.030, "memory": "48Mi"},
    4:  {"resource_mips": 400,  "task_length_mi": 140,    "queue_len": 3,  "deadline": 0.040, "memory": "56Mi"},
    5:  {"resource_mips": 800,  "task_length_mi": 250,    "queue_len": 3,  "deadline": 0.080, "memory": "64Mi"},
    6:  {"resource_mips": 800,  "task_length_mi": 350,    "queue_len": 4,  "deadline": 0.100, "memory": "80Mi"},
    7:  {"resource_mips": 800,  "task_length_mi": 450,    "queue_len": 4,  "deadline": 0.120, "memory": "96Mi"},
    8:  {"resource_mips": 1200, "task_length_mi": 700,    "queue_len": 5,  "deadline": 0.200, "memory": "112Mi"},
    9:  {"resource_mips": 1200, "task_length_mi": 900,    "queue_len": 5,  "deadline": 0.250, "memory": "128Mi"},
    10: {"resource_mips": 1200, "task_length_mi": 1100,   "queue_len": 6,  "deadline": 0.300, "memory": "144Mi"},
    11: {"resource_mips": 2000, "task_length_mi": 3500,   "queue_len": 8,  "deadline": 1.0,   "memory": "192Mi"},
    12: {"resource_mips": 2000, "task_length_mi": 7000,   "queue_len": 10, "deadline": 2.0,   "memory": "256Mi"},
    13: {"resource_mips": 2000, "task_length_mi": 10000,  "queue_len": 12, "deadline": 3.0,   "memory": "320Mi"},
    14: {"resource_mips": 3000, "task_length_mi": 50000,  "queue_len": 15, "deadline": 15.0,  "memory": "512Mi"},
    15: {"resource_mips": 5000, "task_length_mi": 150000, "queue_len": 20, "deadline": 40.0,  "memory": "768Mi"},
}
N_SERVICES = len(SERVICES_INFO)
ACTIVE_SERVICES = tuple(sorted(SERVICES_INFO.keys()))


def compute_exec_time_sec(service_id: int, server_capacity_mips: float) -> float:
    """
    تنها منبع محاسبه‌ی زمان اجرا: task_length_mi / سرعت واقعی سرور میزبان.

    *** به‌روزرسانی مهم: قبلاً این تابع فقط service_id می‌گرفت و بر مبنای
    resource_mips (یک رزرو ثابت، مستقل از میزبان) محاسبه می‌کرد؛ یعنی
    exec_time هیچ‌وقت به این‌که رپلیکا روی edge_small یا large نشسته
    بستگی نداشت - این دقیقاً برخلاف رابطه‌ی فیزیکی مدنظر پرامپت اصلی بود
    (exec_time_sec = task_length_MI / allocated_MIPS واقعیِ میزبان).
    این نسخه اصلاح‌شده صریحاً server_capacity_mips (ظرفیت MIPS واقعی همان
    سروری که رپلیکا رویش قرار می‌گیرد) می‌گیرد، بنابراین همان سرویس روی
    edge_small کندتر و روی large سریع‌تر اجرا می‌شود - دقیقاً طبق مدل
    CloudSim/iFogSim.
    """
    svc = SERVICES_INFO[service_id]
    return svc["task_length_mi"] / server_capacity_mips


COLD_START_PENALTY_FRACTION = 0.20   # ۲۰٪ از deadline هر سرویس
COLD_START_PENALTY_CAP_SEC  = 0.500  # سقف: ۵۰۰ms (معقول برای container init واقعی)
COLD_START_WINDOW_SEC       = 10.0   # فعلاً دست‌نخورده (توضیح پایین‌تر)


def compute_cold_start_penalty_sec(service_id: int) -> float:
    """
    جریمه‌ی cold-start: درصدی از deadline خود سرویس (5QI-aware)، با سقف مطلق.

    *** یادداشت (بخش ۴ پرامپت اصلی): این تابع پیشنهاد بود و اکنون با تأیید
    شما به‌صورت رسمی در موتور فعال شده (simulator/engine.py). طراحی از
    نسخه‌ی قبلی ساده‌تر شده: مؤلفه‌ی «۵۰٪ exec_time» حذف شد چون exec_time
    از این تغییرات به بعد به سرور میزبان وابسته است (نگاه کنید
    compute_exec_time_sec) و این تابع اطلاعی از میزبان ندارد؛ ترکیب یک
    نسبت host-dependent با یک ثابت host-independent هم منطقاً ناسازگار و
    هم به‌طور بالقوه گمراه‌کننده بود. معیار واحدِ deadline-relative (که در
    5QI هم مبنای طبقه‌بندی است) ساده‌تر، قابل‌استناد‌تر، و عاری از این
    ناسازگاری است.

    منطق: min(20% از deadline, سقف مطلق ۵۰۰ms) — برای سرویس‌های URLLC
    (مثلاً deadline=10ms) جریمه عملاً به ۲ms محدود می‌شود، نه یک ثابت
    سراسری یک‌ثانیه‌ای که بی‌قید و شرط کل درخواست را رد deadline می‌کرد.
    """
    svc = CFG.services_info[service_id]
    dl = svc["deadline"]
    penalty = dl * COLD_START_PENALTY_FRACTION
    return min(penalty, COLD_START_PENALTY_CAP_SEC)

# ---------------------------------------------------------------------------
# بخش ۱.۳: داده و تایم‌لاین
# ---------------------------------------------------------------------------

import os as _os

DATA_DIR = _os.environ.get(
    "EOTCH_DATA_DIR",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "raw"),
)
TRAIN_FILES = ["Data1.csv", "Data2.csv", "Data3.csv"]
TEST_FILE = "Data4.csv"
SECONDS_PER_DAY = 86400
LAT_MIN, LAT_MAX = 30.5, 31.7
LON_MIN, LON_MAX = 120.7, 122.0

BASE_LATENCY_MS = 2.0
K_MS_PER_KM = 0.02
L0_MS = 20.0

# DISPATCH_OVERHEAD_MS: تأخیر یک‌طرفه‌ی ثابتِ hop مسیریابی (RTT = 2x این عدد).
# کالیبره‌شده اولیه برای معماری co-located control-plane در 5G-MEC محلی
# (تأخیر control-plane زیر میلی‌ثانیه). باید با داده‌ی زیرساخت واقعی
# خودتان تنظیم شود.
DISPATCH_OVERHEAD_MS = 0.3

PROXIMITY_L0_MS = 7.0

BOOT_DELAY_SEC = 30.0
POD_STARTUP_DELAY_SEC = 5.0
GRACEFUL_TERMINATION_DELAY_SEC = 10.0
SERVER_DRAIN_GRACE_SEC = 15.0
MIN_ACTIVE_DURATION_SEC = 300.0
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0

COLD_START_WINDOW_SEC = 10.0
#COLD_START_PENALTY_SEC = 1.0
# *** یادداشت (بخش ۴ پرامپت): برای سرویس‌های URLLC (deadline=10ms) این جریمه‌ی
# ثابت کشنده است. پیشنهاد: جریمه‌ی متناسب با کلاس سرویس (درصدی از exec_time).
# تغییر نیاز به تأیید صریح دارد.
E_BOOT_SERVER_J = 500.0
E_POD_CREATE_J = 20.0

UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_LOW_SEC = 60.0
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0

DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.7
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2

DECISION_INTERVAL_SEC = 30.0

PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.08,
    "w2_deadline": 0.35,
    "w3_energy": 0.20,
    "w4_load_balance": 0.12,
    "w5_rejected": 0.25,
}
PPO_PENALTY_PER_ACTION = 0.012

SEED = int(_os.environ.get("EOTCH_SEED", "42"))


@dataclass(frozen=True)
class Config:
    server_profiles: dict = field(default_factory=lambda: SERVER_PROFILES)
    server_info: dict = field(default_factory=lambda: SERVER_INFO)
    services_info: dict = field(default_factory=lambda: SERVICES_INFO)
    n_servers: int = N_SERVERS
    n_services: int = N_SERVICES
    active_services: tuple = ACTIVE_SERVICES
    data_dir: str = DATA_DIR
    train_files: tuple = tuple(TRAIN_FILES)
    test_file: str = TEST_FILE
    seconds_per_day: int = SECONDS_PER_DAY
    lat_min: float = LAT_MIN
    lat_max: float = LAT_MAX
    lon_min: float = LON_MIN
    lon_max: float = LON_MAX
    base_latency_ms: float = BASE_LATENCY_MS
    dispatch_overhead_ms: float = DISPATCH_OVERHEAD_MS
    k_ms_per_km: float = K_MS_PER_KM
    l0_ms: float = L0_MS
    proximity_l0_ms: float = PROXIMITY_L0_MS
    boot_delay_sec: float = BOOT_DELAY_SEC
    pod_startup_delay_sec: float = POD_STARTUP_DELAY_SEC
    graceful_termination_delay_sec: float = GRACEFUL_TERMINATION_DELAY_SEC
    server_drain_grace_sec: float = SERVER_DRAIN_GRACE_SEC
    cold_start_window_sec: float = COLD_START_WINDOW_SEC
    e_boot_server_j: float = E_BOOT_SERVER_J
    e_pod_create_j: float = E_POD_CREATE_J
    util_scale_up_threshold: float = UTIL_SCALE_UP_THRESHOLD
    util_scale_down_threshold: float = UTIL_SCALE_DOWN_THRESHOLD
    monitor_window_sec: float = MONITOR_WINDOW_SEC
    sustain_low_sec: float = SUSTAIN_LOW_SEC
    sustain_high_sec: float = SUSTAIN_HIGH_SEC
    cooldown_sec: float = COOLDOWN_SEC
    decision_audit_scale_up_occ_threshold: float = DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD
    decision_audit_scale_down_occ_threshold: float = DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD
    decision_interval_sec: float = DECISION_INTERVAL_SEC
    ppo_reward_weights: dict = field(default_factory=lambda: PPO_REWARD_WEIGHTS)
    ppo_penalty_per_action: float = PPO_PENALTY_PER_ACTION
    seed: int = SEED
    min_active_duration_sec: float = MIN_ACTIVE_DURATION_SEC
    min_replica_age_before_scale_down_sec: float = MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC

CFG = Config()