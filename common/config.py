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

import os as _os

# *** پورتابیلیتی: قبلاً یک مسیر مطلق ویندوزی هاردکد بود
# (r"D:\PT\edge_rl_3\data") که فقط روی یک سیستم خاص کار می‌کرد. حالا از
# env var با fallback به <ریشه‌ی پروژه>/data/raw استفاده می‌شود.
#   لینوکس/مک:   export EOTCH_DATA_DIR=/path/to/data
#   ویندوز(cmd): set EOTCH_DATA_DIR=D:\PT\edge_rl_3\data
DATA_DIR = _os.environ.get(
    "EOTCH_DATA_DIR",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "raw"),
)
TRAIN_FILES = ["Data1.csv", "Data2.csv", "Data3.csv"]  # شنبه‌های هفته ۱ تا ۳
TEST_FILE = "Data4.csv"  # شنبه‌ی هفته ۴
SECONDS_PER_DAY = 86400
LAT_MIN, LAT_MAX = 30.5, 31.7
LON_MIN, LON_MAX = 120.7, 122.0

BASE_LATENCY_MS = 2.0
K_MS_PER_KM = 0.02
L0_MS = 20.0

BOOT_DELAY_SEC = 30.0
POD_STARTUP_DELAY_SEC = 5.0
GRACEFUL_TERMINATION_DELAY_SEC = 10.0
SERVER_DRAIN_GRACE_SEC = 15.0
MIN_ACTIVE_DURATION_SEC = 300.0  # حداقل مدت ACTIVE بودن قبل از واجد شرایط TURN_OFF
                                   # (ضد flapping - طبق تحلیل analyze_decision_quality.py:
                                   #  ۹۱-۹۶٪ چرخه‌های on/off زیر ۵ دقیقه dwell داشتند)
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0

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
# بخش ۸: معیار «درستی تصمیم» - آستانه‌ی ممیزی *مستقل* از سیاست داخلی هر
# الگوریتم (Greedy=0.7, Voila=0.75, HPA=فرمول K8s, PPO=یادگرفته‌شده). سند:
# «هر الگوریتم گزارش بده ... از این تصمیم‌ها چند تا (با معیار: آیا واقعاً
# لازم بود) درست بودن». اگر از threshold خودِ همان الگوریتم استفاده می‌شد،
# هر تصمیم به‌تعریف «درست» می‌بود - این معیار باید یک خط‌کش واحد و مستقل
# برای هر ۴ الگوریتم باشد، نه بازتاب منطق داخلی خودشان.
# ---------------------------------------------------------------------------
DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.7    # اشغال صف بالاتر از این یا rejection>0 -> واقعاً نیاز به SCALE_UP بود
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2  # اشغال صف پایین‌تر از این -> واقعاً ظرفیت اضافی بود

DECISION_INTERVAL_SEC = 30.0

# ---------------------------------------------------------------------------
# بخش ۱۱.۴: وزن‌های reward PPO.
# *** CHANGELOG (بازبینی ۳): قبلاً فقط ۴ وزن بود (w1..w4) و جریمه‌ی
# «درخواست ردشده» جدا و نرمال‌نشده با PPO_PENALTY_PER_REJECTED اعمال
# می‌شد (نگاه کنید algorithms/ppo/env.py برای شرح کامل باگ: این جمله‌ی
# نرمال‌نشده می‌توانست ۵-۱۵ برابر بقیه‌ی اجزای reward بزرگ‌تر شود و کل
# سیگنال را تحت‌الشعاع قرار دهد -> عامل یاد گرفت هیچ اکشنی نزند و فقط
# سرور اضافه نگه دارد). حالا num_rejected_recent هم مثل بقیه نرمال و با
# وزن صریح w5_rejected در همان مجموع وزن‌دار ترکیب می‌شود.
# مقادیر کالیبره‌شده (جمع=۱.۰): پاسخ‌گویی به رد شدن (w5) و انرژی (w3)
# سنگین‌تر از پاسخ‌گویی خام (w1) وزن گرفته‌اند - بخش ۱۳ سند: قابل تنظیم.
# ---------------------------------------------------------------------------
PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.12,
    "w2_deadline": 0.20,
    "w3_energy": 0.30,
    "w4_load_balance": 0.23,
    "w5_rejected": 0.15,
}
# *** PPO_PENALTY_PER_REJECTED حذف شد: env.py دیگر جریمه‌ی رد را جداگانه و
# نرمال‌نشده اعمال نمی‌کند (نگاه کنید بالا)؛ اگر کد دیگری (مثلاً نسخه‌ی
# قدیمی‌تر train.py/infer.py) هنوز به CFG.ppo_penalty_per_rejected ارجاع
# می‌دهد، آن ارجاع باید حذف/به w5_rejected منتقل شود.
PPO_PENALTY_PER_ACTION = 0.01
SEED = 42
#python -m evaluation.compare_runs --output-dir outputs/seed44



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
    decision_audit_scale_up_occ_threshold: float = DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD
    decision_audit_scale_down_occ_threshold: float = DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD
    decision_interval_sec: float = DECISION_INTERVAL_SEC
    ppo_reward_weights: dict = field(default_factory=lambda: PPO_REWARD_WEIGHTS)
    ppo_penalty_per_action: float = PPO_PENALTY_PER_ACTION
    seed: int = SEED
    min_active_duration_sec: float = MIN_ACTIVE_DURATION_SEC
    min_replica_age_before_scale_down_sec: float = MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC

CFG = Config()