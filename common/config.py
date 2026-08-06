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
    1: {"cpu_demand": 2, "exec_time": 4, "queue_len": 3, "deadline": 8, "memory": "64Mi"},# 2k -1.5-2.5*exec
    2: {"cpu_demand": 4, "exec_time": 6, "queue_len": 3, "deadline": 20, "memory": "80Mi"}, #200ms 10s
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

# ---------------------------------------------------------------------------
# *** اصلاح معماری (بازبینی): مسیریابی دیگر داخل خودِ دیسپچر/موتور بدون
# مدل‌سازی هزینه انجام نمی‌شود. جریان واقعی (که فاز ۳ - k8s_adapter/dispatcher_api.py
# و realtime_dispatcher.py:route_request - از قبل پیاده‌سازی می‌کرد ولی
# شبیه‌سازی فاز ۱/۲ هنوز مدل نمی‌کرد): BTS اول به دیسپچر مرکزی (control-plane،
# روی یک ماشین/دیتاسنتر ثابت، نه یک BTS دیگر) وصل می‌شود تا سرور مقصد را
# بگیرد (هزینه‌ی این hop سبک است - فقط چند بایت جابه‌جا می‌شود - و مستقل از
# فاصله‌ی جغرافیایی هر BTS با یک ثابت ساده مدل می‌شود، نه هاورسین). سپس BTS
# *خودش* مستقیماً و بدون واسطه به سرور واقعی (data-plane) وصل می‌شود - این
# hop دوم همان مدل هاورسین قبلی (BASE_LATENCY_MS + K_MS_PER_KM*distance) را
# می‌گیرد. response_time نهایی حالا هر دو hop را جمع می‌زند: مسیریابی
# (BTS<->دیسپچر) + داده‌ی واقعی (BTS<->سرور).
#
# DISPATCH_OVERHEAD_MS: تأخیر یک‌طرفه‌ی ثابتِ hop مسیریابی (RTT = 2x این عدد،
# دقیقاً مثل BASE_LATENCY_MS برای hop داده). چون دیسپچر processing سبکی دارد
# (فقط یک lookup/تصمیم گزینش replica، نه I/O سنگین)، این عدد کوچک و عمدتاً
# نماینده‌ی تأخیر شبکه‌ی BTS<->دیتاسنتر core (نه فاصله‌ی جغرافیایی BTS<->سرور
# edge) است - بخش ۱۳ سند: قابل کالیبراسیون بیشتر با داده‌ی واقعی شبکه‌ی شما.
DISPATCH_OVERHEAD_MS = 5.0
# ---------------------------------------------------------------------------

# *** رفع باگ (بازبینی: proximity_violation_rate همیشه صفر بود): L0_MS بالا
# برای «پوشش اولیه» (بخش ۴ سند، initial_placement) کالیبره شده و مقداری
# سخاوتمندانه است. اما وقتی همان L0_MS برای متریک لحظه‌ای proximity_violation
# (Vlo مقاله‌ی VOILA) هم استفاده می‌شد، هرگز trigger نمی‌شد: حداکثر تأخیر
# رفت‌وبرگشت ممکن داخل کل محدوده‌ی جغرافیایی پیکربندی‌شده فقط ~۱۱.۳ میلی‌ثانیه
# است (فاصله‌ی حداکثر ۱۸۲ کیلومتر × k_ms_per_km × ۲) - همیشه زیر ۲۰. یک ثابت
# مستقل و در مقیاس واقعی تأخیرهای این پروژه (چند میلی‌ثانیه) لازم است. مقدار
# ۴.۰ حدوداً میانه‌ی بازه‌ی واقعی تأخیر رفت‌وبرگشت (۴ تا ۱۱.۳ میلی‌ثانیه) است -
# بخش ۱۳ سند: قابل کالیبراسیون بیشتر با داده‌ی واقعی.
PROXIMITY_L0_MS = 4.0

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
# *** یادداشت طراحی (بخش ۲.۵/۱۰ سند): این جریمه به‌صورت افزایش واقعی
# service_time اعمال می‌شود (common/models.py:Replica.try_admit)، نه فقط
# افزودن عددی به response_time گزارش‌شده - یعنی روی wait_time درخواست‌های
# بعدیِ همان replica، انرژی، و reward PPO هم اثر غیرمستقیم دارد. تصمیم
# آگاهانه برای واقع‌گرایی بیشتر (replica در حال cold-start واقعاً کندتر
# است)، نه یک باگ.
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
        # *** بازگردانده شد به مجموعه‌ی وزن‌های اصلی (که با reward *سالم*،
        # یعنی بعد از رفع باگ نرمال‌سازی norm_rt/norm_energy در env.py، قبلاً
        # نتیجه‌ی قابل‌قبول داده بود). مجموعه‌ی وزن جدید (w3_energy=0.18/
        # w5_rejected=0.25 پایین، کامنت شده) هرگز روی یک reward سالم آزمایش
        # نشده بود - هم‌زمان با باگ نرمال‌سازی معرفی شده بود، پس نمی‌شود
        # گفت خودش باعث افت بوده یا نه. بعد از این‌که مدل با این وزن‌های
        # اصلی و reward درست‌شده دوباره train شد و ارزیابی گرفت، اگر باز هم
        # جای بهبود بود می‌شود دوباره این وزن‌های جایگزین را روی reward
        # سالم امتحان کرد (نه هم‌زمان با فیکس باگ، تا اثر هرکدام جدا سنجیده شود).
"""PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.12,
    "w2_deadline": 0.20,
    "w3_energy": 0.30,
    "w4_load_balance": 0.23,
    "w5_rejected": 0.15,
}"""

# *** آزمایش بعدی (بعد از دوباره train با وزن‌های بالا روی reward سالم):
# PPO_REWARD_WEIGHTS = {
#     "w1_response_time": 0.12,
#     "w2_deadline": 0.25,
#     "w3_energy": 0.18,
#     "w4_load_balance": 0.20,
#     "w5_rejected": 0.25,
# }
# common/config.py
PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.10,  # ↓ کاهش (Greedy already خوب است)
    "w2_deadline": 0.40,       # ↑↑ افزایش (مشکل PPO!)
    "w3_energy": 0.25,         # ↑ افزایش (خیلی SCALE_UP می‌زند)
    "w4_load_balance": 0.15,   # ↓ کاهش
    "w5_rejected": 0.10,       # ↓ کاهش (از 0.25)
}
PPO_PENALTY_PER_ACTION = 0.02  # ↑ بیشتر شدن (flapping کم کند)

# *** قابل override با env var (هم‌راستا با الگوی EOTCH_DATA_DIR بالا)، تا
# اجرای چند-seed (algorithms/ppo/train.py + evaluation/aggregate_seeds.py)
# نیازی به ویرایش دستی این فایل قبل از هر اجرا نداشته باشد:
#   EOTCH_SEED=44 python3 -m algorithms.ppo.train
#   EOTCH_SEED=44 python3 -m evaluation.compare_runs --output-dir outputs/seed44
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