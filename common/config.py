"""
common/config.py
تنظیمات مرکزی پروژه - دقیقاً طبق «سند معماری کامل پروژه» بخش‌های ۱، ۹، ۱۱.
هیچ عدد پارامتری نباید در جای دیگر کد hardcode شود؛ همه از اینجا خوانده می‌شوند.

*** CHANGELOG (بازبینی ۲): اضافه شدن SUSTAIN_HIGH_SEC - بخش ۶.۱ سند می‌گوید
utilization باید «به‌طور مداوم» بالای آستانه باشد تا TURN_ON اعمال شود، ولی
این تداوم قبلاً فقط سمت پایین (SUSTAIN_LOW_SEC) پیاده‌سازی شده بود. حالا
simulator/engine.py از این مقدار برای سمت بالا هم استفاده می‌کند (نگاه کنید
_any_active_server_sustained_overloaded در engine.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# بخش ۱.۱: پروفایل‌ها و سرورها (heterogeneous، ثابت - مکان‌یابی قبلاً با solver انجام شده)
# ---------------------------------------------------------------------------

SERVER_PROFILES: Dict[str, dict] = {
    "edge_small": {"capacity": 60, "p_idle": 40, "p_max": 130},
    "medium": {"capacity": 100, "p_idle": 70, "p_max": 220},
    "large": {"capacity": 200, "p_idle": 110, "p_max": 380},
}

_CAPACITY_TO_PROFILE = {p["capacity"]: name for name, p in SERVER_PROFILES.items()}

# *** bts_id هرکدام = نزدیک‌ترین BTSID واقعی دیتاست به مختصات داده‌شده (محاسبه‌شده
# یک‌بار روی هر ۴ روز؛ همه زیر ۱ کیلومتر فاصله دارند - نگاه کنید به یادداشت توسعه).
SERVER_INFO: Dict[int, dict] = {
    1: {"bts_id": 1498, "lat": 31.37, "long": 121.25, "capacity": 60},
    2: {"bts_id": 777, "lat": 31.31, "long": 121.51, "capacity": 60},
    3: {"bts_id": 530, "lat": 31.10, "long": 121.18, "capacity": 60},
    4: {"bts_id": 121, "lat": 31.25, "long": 121.37, "capacity": 60},
    5: {"bts_id": 292, "lat": 31.10, "long": 121.36, "capacity": 100},
    6: {"bts_id": 1344, "lat": 31.04, "long": 121.74, "capacity": 100},
    7: {"bts_id": 182, "lat": 31.17, "long": 121.57, "capacity": 100},
    8: {"bts_id": 505, "lat": 31.15, "long": 121.41, "capacity": 100},
    9: {"bts_id": 419, "lat": 31.20, "long": 121.43, "capacity": 200},
    10: {"bts_id": 609, "lat": 31.16, "long": 121.49, "capacity": 200},
}
for _sid, _info in SERVER_INFO.items():
    _info["profile"] = _CAPACITY_TO_PROFILE[_info["capacity"]]

N_SERVERS = len(SERVER_INFO)

# ---------------------------------------------------------------------------
# بخش ۱.۲: سرویس‌ها (ثابت)
# ---------------------------------------------------------------------------

SERVICES_INFO: Dict[int, dict] = {
    1: {"cpu_demand": 2, "exec_time": 4, "queue_len": 3, "deadline": 16, "memory": "64Mi"},
    2: {"cpu_demand": 4, "exec_time": 6, "queue_len": 3, "deadline": 20, "memory": "80Mi"},
    3: {"cpu_demand": 6, "exec_time": 8, "queue_len": 4, "deadline": 28, "memory": "96Mi"},
    4: {"cpu_demand": 6, "exec_time": 10, "queue_len": 4, "deadline": 35, "memory": "112Mi"},
    5: {"cpu_demand": 8, "exec_time": 12, "queue_len": 5, "deadline": 40, "memory": "128Mi"},
    6: {"cpu_demand": 10, "exec_time": 20, "queue_len": 5, "deadline": 180, "memory": "144Mi"},
    7: {"cpu_demand": 12, "exec_time": 25, "queue_len": 6, "deadline": 200, "memory": "176Mi"},
    8: {"cpu_demand": 14, "exec_time": 30, "queue_len": 6, "deadline": 220, "memory": "208Mi"},
    9: {"cpu_demand": 16, "exec_time": 35, "queue_len": 7, "deadline": 240, "memory": "240Mi"},
    10: {"cpu_demand": 18, "exec_time": 40, "queue_len": 7, "deadline": 260, "memory": "272Mi"},
    11: {"cpu_demand": 22, "exec_time": 55, "queue_len": 8, "deadline": 420, "memory": "304Mi"},
    12: {"cpu_demand": 26, "exec_time": 70, "queue_len": 8, "deadline": 460, "memory": "336Mi"},
    13: {"cpu_demand": 30, "exec_time": 85, "queue_len": 9, "deadline": 500, "memory": "400Mi"},
    14: {"cpu_demand": 34, "exec_time": 100, "queue_len": 9, "deadline": 540, "memory": "528Mi"},
    15: {"cpu_demand": 40, "exec_time": 120, "queue_len": 10, "deadline": 580, "memory": "656Mi"},
}
N_SERVICES = len(SERVICES_INFO)
ACTIVE_SERVICES = tuple(sorted(SERVICES_INFO.keys()))

# ---------------------------------------------------------------------------
# بخش ۱.۳: داده و تایم‌لاین
# ---------------------------------------------------------------------------

DATA_DIR = r"D:\PT\edge_rl_3\data"
TRAIN_FILES = ["Data1.csv", "Data2.csv", "Data3.csv"]  # شنبه‌های هفته ۱ تا ۳
TEST_FILE = "Data4.csv"  # شنبه‌ی هفته ۴
SECONDS_PER_DAY = 86400
# محدوده‌ی جغرافیایی برای حذف نویز/دورافتاده‌ها (همان محدوده‌ای که سرورها در آن قرار دارند)
LAT_MIN, LAT_MAX = 30.5, 31.7
LON_MIN, LON_MAX = 120.7, 122.0

# ---------------------------------------------------------------------------
# بخش ۳: مدل تاخیر شبکه
# ---------------------------------------------------------------------------

BASE_LATENCY_MS = 2.0
K_MS_PER_KM = 0.02

# *** l0: آستانه‌ی تاخیر رفت‌وبرگشت برای «پوشش قابل‌قبول» در جایگذاری اولیه
# (بخش ۴ و ۵ سند به l0 اشاره می‌کنند ولی مقدار عددی‌اش در بخش ۹ فراموش شده
# بود). طبق مقدار پیش‌فرض خود مقاله‌ی Voila (بخش V-D) = 20ms قرار داده شد؛
# در صورت نیاز کالیبره کنید.
L0_MS = 20.0

# ---------------------------------------------------------------------------
# بخش ۹: تاخیرها، جریمه‌ها، پارامترهای پیکربندی‌پذیر
# ---------------------------------------------------------------------------

BOOT_DELAY_SEC = 30.0                  # سرور OFF -> ACTIVE (از طریق BOOTING)
POD_STARTUP_DELAY_SEC = 5.0            # رپلیکا STARTING -> READY
GRACEFUL_TERMINATION_DELAY_SEC = 10.0  # رپلیکا DRAINING -> TERMINATED
SERVER_DRAIN_GRACE_SEC = 15.0          # سرور DRAINING -> OFF پس از خالی‌شدن

COLD_START_WINDOW_SEC = 10.0
COLD_START_PENALTY_SEC = 1.0

E_BOOT_SERVER_J = 500.0
E_POD_CREATE_J = 20.0

UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_LOW_SEC = 60.0
# *** بخش ۶.۱ سند: «اگر utilization(t) (میانگین متحرک روی MONITOR_WINDOW_SEC)
# > 95% به‌طور مداوم -> نیاز به روشن‌کردن سرور». قبلاً این سمت (بر خلاف سمت
# پایین که SUSTAIN_LOW_SEC داشت) هیچ الزام تداومی نداشت - یک نمونه‌ی
# لحظه‌ای بالای آستانه فوراً TURN_ON را trigger می‌کرد. اضافه شد تا با سمت
# پایین متقارن باشد. مقدار پیش‌فرض = یک MONITOR_WINDOW کامل (یعنی حداقل ۲
# تیک تصمیم متوالی باید overload را نشان دهند)؛ طبق بخش ۱۳ قابل کالیبراسیون.
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0

# ---------------------------------------------------------------------------
# بخش ۱۱: PPO-DRL
# ---------------------------------------------------------------------------

DECISION_INTERVAL_SEC = 30.0  # = MONITOR_WINDOW_SEC (عمداً یکسان - یک تیک تصمیم مشترک)

PPO_REWARD_WEIGHTS = {"w1_response_time": 0.24, "w2_deadline": 0.24,
                       "w3_energy": 0.29, "w4_load_balance": 0.23}
PPO_PENALTY_PER_REJECTED = 0.5
PPO_PENALTY_PER_ACTION = 0.01  # جریمه‌ی ثابت کوچک هر SCALE_UP/DOWN/TURN_ON/OFF

SEED = 42


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
    k_ms_per_km: float = K_MS_PER_KM
    l0_ms: float = L0_MS
    boot_delay_sec: float = BOOT_DELAY_SEC
    pod_startup_delay_sec: float = POD_STARTUP_DELAY_SEC
    graceful_termination_delay_sec: float = GRACEFUL_TERMINATION_DELAY_SEC
    server_drain_grace_sec: float = SERVER_DRAIN_GRACE_SEC
    cold_start_window_sec: float = COLD_START_WINDOW_SEC
    cold_start_penalty_sec: float = COLD_START_PENALTY_SEC
    e_boot_server_j: float = E_BOOT_SERVER_J
    e_pod_create_j: float = E_POD_CREATE_J
    util_scale_up_threshold: float = UTIL_SCALE_UP_THRESHOLD
    util_scale_down_threshold: float = UTIL_SCALE_DOWN_THRESHOLD
    monitor_window_sec: float = MONITOR_WINDOW_SEC
    sustain_low_sec: float = SUSTAIN_LOW_SEC
    sustain_high_sec: float = SUSTAIN_HIGH_SEC
    cooldown_sec: float = COOLDOWN_SEC
    decision_interval_sec: float = DECISION_INTERVAL_SEC
    ppo_reward_weights: dict = field(default_factory=lambda: PPO_REWARD_WEIGHTS)
    ppo_penalty_per_rejected: float = PPO_PENALTY_PER_REJECTED
    ppo_penalty_per_action: float = PPO_PENALTY_PER_ACTION
    seed: int = SEED


CFG = Config()