from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


SERVER_PROFILES = {
    # HPE ProLiant DL360 Gen10, 1x Xeon Silver 4110 (8c, 2.10GHz, 85W TDP)
    # idle/max تخمینی با درون‌یابی TDP نسبت به Platinum 8280 اندازه‌گیری‌شده
    # (SPECpower_ssj2008 res2019q2-00916: idle=39.4W/max=217W @ TDP=205W)
    "edge_small": {"n_cores": 8,  "mips_per_core": 2520, "capacity_mips": 20160,
                   "p_idle": 28, "p_max": 113},
    # HPE ProLiant DL360 Gen10, 1x Xeon Gold 5118 (12c, 2.30GHz, 105W TDP)
    "medium":     {"n_cores": 12, "mips_per_core": 2760, "capacity_mips": 33120,
                   "p_idle": 30, "p_max": 130},
    # HPE ProLiant DL360 Gen10, 1x Xeon Gold 6130 (16c, 2.10GHz, 125W TDP)
    "large":      {"n_cores": 16, "mips_per_core": 2520, "capacity_mips": 40320,
                   "p_idle": 33, "p_max": 148},
}
_CAPACITY_TO_PROFILE = {p["capacity_mips"]: name for name, p in SERVER_PROFILES.items()}
REFERENCE_MIPS_PER_CORE = SERVER_PROFILES["medium"]["mips_per_core"]  # 2760

SERVER_INFO = {
    1: {"bts_id": 1498, "lat": 31.37, "long": 121.25, "capacity_mips": 20160},
    2: {"bts_id": 777,  "lat": 31.31, "long": 121.51, "capacity_mips": 20160},
    3: {"bts_id": 530,  "lat": 31.10, "long": 121.18, "capacity_mips": 33120},
    4: {"bts_id": 121,  "lat": 31.25, "long": 121.37, "capacity_mips": 33120},
    5: {"bts_id": 292,  "lat": 31.10, "long": 121.36, "capacity_mips": 33120},
    6: {"bts_id": 1344, "lat": 31.04, "long": 121.74, "capacity_mips": 33120},
    7: {"bts_id": 182,  "lat": 31.17, "long": 121.57, "capacity_mips": 40320},
    8: {"bts_id": 505,  "lat": 31.15, "long": 121.41, "capacity_mips": 40320},
    9: {"bts_id": 419,  "lat": 31.20, "long": 121.43, "capacity_mips": 40320},
    10: {"bts_id": 609, "lat": 31.16, "long": 121.49, "capacity_mips": 40320},
}
for _sid, _info in SERVER_INFO.items():
    _info["profile"] = _CAPACITY_TO_PROFILE[_info["capacity_mips"]]
N_SERVERS = len(SERVER_INFO)
SERVICES_INFO: Dict[int, dict] = {
    1:  {"resource_mips": 4800, "task_length_mi": 30,     "queue_len": 1,  "deadline": 0.010, "memory": "32Mi"},
    2:  {"resource_mips": 5200, "task_length_mi": 50,     "queue_len": 2,  "deadline": 0.015, "memory": "40Mi"},
    3:  {"resource_mips": 5200, "task_length_mi": 100,    "queue_len": 2,  "deadline": 0.030, "memory": "48Mi"},
    4:  {"resource_mips": 5400, "task_length_mi": 140,    "queue_len": 3,  "deadline": 0.040, "memory": "56Mi"},
    5:  {"resource_mips": 4900, "task_length_mi": 250,    "queue_len": 3,  "deadline": 0.080, "memory": "64Mi"},
    6:  {"resource_mips": 5400, "task_length_mi": 350,    "queue_len": 4,  "deadline": 0.100, "memory": "80Mi"},
    7:  {"resource_mips": 5800, "task_length_mi": 450,    "queue_len": 4,  "deadline": 0.120, "memory": "96Mi"},
    8:  {"resource_mips": 5400, "task_length_mi": 700,    "queue_len": 5,  "deadline": 0.200, "memory": "112Mi"},
    9:  {"resource_mips": 5600, "task_length_mi": 900,    "queue_len": 5,  "deadline": 0.250, "memory": "128Mi"},
    10: {"resource_mips": 5700, "task_length_mi": 1100,   "queue_len": 6,  "deadline": 0.300, "memory": "144Mi"},
    11: {"resource_mips": 5400, "task_length_mi": 3500,   "queue_len": 8,  "deadline": 1.0,   "memory": "192Mi"},
    12: {"resource_mips": 5400, "task_length_mi": 7000,   "queue_len": 10, "deadline": 2.0,   "memory": "256Mi"},
    13: {"resource_mips": 5200, "task_length_mi": 10000,  "queue_len": 12, "deadline": 3.0,   "memory": "320Mi"},
    14: {"resource_mips": 5200, "task_length_mi": 50000,  "queue_len": 15, "deadline": 15.0,  "memory": "512Mi"},
    15: {"resource_mips": 5800, "task_length_mi": 150000, "queue_len": 20, "deadline": 40.0,  "memory": "768Mi"},
}
N_SERVICES = len(SERVICES_INFO)

_max_single_service = max(s["resource_mips"] for s in SERVICES_INFO.values())
_min_server_capacity = min(p["capacity_mips"] for p in SERVER_PROFILES.values())
assert _max_single_service <= _min_server_capacity, (
    f"سرویس با resource_mips={_max_single_service} روی کوچک‌ترین سرور "
    f"({_min_server_capacity} MIPS) هم جا نمی‌شود")

N_SERVICES = len(SERVICES_INFO)
ACTIVE_SERVICES = tuple(sorted(SERVICES_INFO.keys()))


 

REFERENCE_MIPS_PER_CORE = SERVER_PROFILES["medium"]["mips_per_core"]   
 

def compute_exec_time_sec(service_id: int, host_mips_per_core: float) -> float:
 
    svc = SERVICES_INFO[service_id]
    speed_factor = host_mips_per_core / REFERENCE_MIPS_PER_CORE
    effective_mips = svc["resource_mips"] * speed_factor
    return svc["task_length_mi"] / effective_mips

 

COLD_START_PENALTY_FRACTION = 0.20  
COLD_START_PENALTY_CAP_SEC  = 0.500  
COLD_START_WINDOW_SEC       = 10.0   
 

def compute_cold_start_penalty_sec(service_id: int, host_mips_per_core: float) -> float:
 
    svc = CFG.services_info[service_id]
    dl = svc["deadline"]
    et = compute_exec_time_sec(service_id, host_mips_per_core)

    penalty_deadline = dl * COLD_START_PENALTY_FRACTION
    penalty_exec = et * 0.50
    penalty = min(penalty_deadline, penalty_exec)
 
    return min(penalty, COLD_START_PENALTY_CAP_SEC)



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

# تأخیر شبکه (Latency)
BASE_LATENCY_MS = 2.0       # پایه‌ای
K_MS_PER_KM = 0.02          # ضریب جغرافیایی
L0_MS = 20.0                # آستانه L0 coverage
DISPATCH_OVERHEAD_MS = 0.3  # تأخیر dispatcher
PROXIMITY_L0_MS = 7.0       # آستانه proximity

# Server Lifecycle
BOOT_DELAY_SEC = 30.0
POD_STARTUP_DELAY_SEC = 5.0
GRACEFUL_TERMINATION_DELAY_SEC = 10.0
SERVER_DRAIN_GRACE_SEC = 15.0
MIN_ACTIVE_DURATION_SEC = 300.0
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0

# انرژی (Energy)
E_BOOT_SERVER_J = 500.0
E_POD_CREATE_J = 20.0

# Scaling Thresholds
UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_LOW_SEC = 60.0
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0

DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.7
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2
DECISION_INTERVAL_SEC = 30.0

# PPO Reward
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