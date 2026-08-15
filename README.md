# مدیریت پویای منابع محاسباتی لبه (Edge Resource Management)

## Greedy / Kubernetes-HPA / VOILA / PPO-DRL — شبیه‌سازی + اجرای واقعی روی Kubernetes

---

## ۱. ایده‌ی کلی سیستم

۱۰ سرور لبه (edge server) در نقاط مختلف یک شهر قرار دارند و باید به درخواست‌هایی که از ایستگاه‌های پایه‌ی موبایل (BTS) می‌رسند، برای ۱۵ نوع سرویس مختلف پاسخ دهند. سؤال اصلی این است: چه زمانی یک سرور روشن یا خاموش شود، چند نمونه (replica) از هر سرویس اجرا شود، و هر درخواست به کدام سرور هدایت شود — طوری که هم‌زمان تأخیر پاسخ‌دهی پایین بماند، سررسیدها (deadline) نقض نشوند، و مصرف انرژی هم کنترل‌شده باشد.

چهار استراتژی تصمیم‌گیری برای پاسخ به این سؤال پیاده‌سازی و روی داده‌ی واقعی ترافیک BTS شانگهای، با معیارهای یکسان، با هم مقایسه می‌شوند:

| الگوریتم | فلسفه |
|---|---|
| **Greedy** | آستانه‌ساده و مکان‌آگاه: بر اساس اشغال صف و نرخ رد شدن تصمیم می‌گیرد؛ برای جای‌گذاری و مهاجرت سرویس از مرکز ثقل واقعی تقاضا استفاده می‌کند. baseline پروژه است. |
| **HPA** | معادل Kubernetes Horizontal Pod Autoscaler: عمداً مکان‌ناآگاه است، فقط بر اساس نسبت اشغال صف نسبت به یک هدف ثابت (۷۰٪) تعداد replica مطلوب را حساب می‌کند و جای‌گذاری را فقط از روی ظرفیت آزاد انتخاب می‌کند — این دقیقاً همان رفتار واقعی HPA است. |
| **VOILA** | placement، migration، و انتخاب قربانی هنگام کاهش مقیاس را بر اساس مرکز ثقل تقاضای واقعی هر سرویس (medoid موقعیت درخواست‌های اخیر) انجام می‌دهد. علاوه بر نقض ظرفیت، نقض نزدیکی جغرافیایی را هم به‌عنوان سیگنال دوم برای افزایش مقیاس در نظر می‌گیرد. برای مسیریابی لحظه‌ای درخواست‌ها همان منطق مشترک پایه را استفاده می‌کند. |
| **PPO-DRL** | یک عامل یادگیری تقویتی (Proximal Policy Optimization با MaskablePPO) که هر سه نوع تصمیم (اسکیل، provisioning، جای‌گذاری) را هم‌زمان از یک بردار حالت یاد می‌گیرد. آموزش با warm-start از دموی Greedy (Behavior Cloning) شروع می‌شود و سپس با RL روی یک پاداش وزن‌دار fine-tune می‌شود. برخلاف سه الگوریتم دیگر، تصمیمات provisioning آن نیازی به تداوم زمانی overload/underload ندارد. |

هر چهار الگوریتم برای مسیریابی لحظه‌ای درخواست از همان منطق مشترک استفاده می‌کنند تا مقایسه منصفانه بماند؛ فقط تصمیم‌های scale، provision، placement و migration با هم فرق دارند.

---

## ۲. منابع سیستم

### ۲.۱ سرورها

سه پروفایل سرور روی مدل سخت‌افزاری واقعی HPE ProLiant DL360 Gen10 با سه CPU متفاوت تعریف شده‌اند. عددهای توان مصرفی (`p_idle`/`p_max`، بر حسب وات) با درون‌یابی نسبت به داده‌ی اندازه‌گیری‌شده‌ی SPECpower_ssj2008 به‌دست آمده‌اند؛ یعنی تخمین کالیبره‌شده‌اند، نه اندازه‌گیری مستقیم.

```python
SERVER_PROFILES = {
    # Xeon Silver 4110 — 8 هسته، 2.10GHz، TDP=85W
    "edge_small": {"n_cores": 8,  "mips_per_core": 2520, "capacity_mips": 20160,
                   "p_idle": 28, "p_max": 113},
    # Xeon Gold 5118 — 12 هسته، 2.30GHz، TDP=105W
    "medium":     {"n_cores": 12, "mips_per_core": 2760, "capacity_mips": 33120,
                   "p_idle": 30, "p_max": 130},
    # Xeon Gold 6130 — 16 هسته، 2.10GHz، TDP=125W
    "large":      {"n_cores": 16, "mips_per_core": 2520, "capacity_mips": 40320,
                   "p_idle": 33, "p_max": 148},
}
REFERENCE_MIPS_PER_CORE = SERVER_PROFILES["medium"]["mips_per_core"]   # 2760، سرعت مرجع
```

`capacity_mips = n_cores × mips_per_core` — واحد ظرفیت هر سرور، مجموع MIPS (میلیون دستور در ثانیه) همه‌ی هسته‌هایش است.

ده سرور با موقعیت جغرافیایی ثابت، روی BTSهای واقعی دیتاست شانگهای:

```python
SERVER_INFO = {
    1:  {"bts_id": 1498, "lat": 31.37, "long": 121.25, "capacity_mips": 20160},  # edge_small
    2:  {"bts_id": 777,  "lat": 31.31, "long": 121.51, "capacity_mips": 20160},  # edge_small
    3:  {"bts_id": 530,  "lat": 31.10, "long": 121.18, "capacity_mips": 33120},  # medium
    4:  {"bts_id": 121,  "lat": 31.25, "long": 121.37, "capacity_mips": 33120},  # medium
    5:  {"bts_id": 292,  "lat": 31.10, "long": 121.36, "capacity_mips": 33120},  # medium
    6:  {"bts_id": 1344, "lat": 31.04, "long": 121.74, "capacity_mips": 33120},  # medium
    7:  {"bts_id": 182,  "lat": 31.17, "long": 121.57, "capacity_mips": 40320},  # large
    8:  {"bts_id": 505,  "lat": 31.15, "long": 121.41, "capacity_mips": 40320},  # large
    9:  {"bts_id": 419,  "lat": 31.20, "long": 121.43, "capacity_mips": 40320},  # large
    10: {"bts_id": 609,  "lat": 31.16, "long": 121.49, "capacity_mips": 40320},  # large
}
N_SERVERS = 10
```

پروفایل هر سرور از روی `capacity_mips` استخراج می‌شود (نگاشت معکوس، یک بار در زمان بارگذاری پیکربندی).

قید سخت جای‌گذاری، که در همه‌ی تصمیمات placement/scaling هر چهار الگوریتم رعایت می‌شود:

```
مجموع مصرف MIPS همه‌ی سرویس‌های میزبانی‌شده روی یک سرور  ≤  ظرفیت آن سرور
```

هر سرور حداکثر یک رپلیکا از هر سرویس می‌تواند میزبانی کند.

### ۲.۲ سرویس‌ها

۱۵ نوع سرویس با بار CPU متفاوت و سررسید (deadline) متفاوت وجود دارد:

```python
SERVICES_INFO = {
    1:  {"resource_mips": 4800, "task_length_mi": 55,     "queue_len": 1,  "deadline": 0.030, "memory": "32Mi"},
    2:  {"resource_mips": 4900, "task_length_mi": 110,    "queue_len": 2,  "deadline": 0.050, "memory": "48Mi"},
    3:  {"resource_mips": 5000, "task_length_mi": 140,    "queue_len": 2,  "deadline": 0.060, "memory": "48Mi"},
    4:  {"resource_mips": 5100, "task_length_mi": 190,    "queue_len": 3,  "deadline": 0.075, "memory": "56Mi"},
    5:  {"resource_mips": 5200, "task_length_mi": 260,    "queue_len": 3,  "deadline": 0.100, "memory": "64Mi"},
    6:  {"resource_mips": 5300, "task_length_mi": 310,    "queue_len": 3,  "deadline": 0.100, "memory": "72Mi"},
    7:  {"resource_mips": 5400, "task_length_mi": 420,    "queue_len": 4,  "deadline": 0.150, "memory": "96Mi"},
    8:  {"resource_mips": 5400, "task_length_mi": 570,    "queue_len": 5,  "deadline": 0.200, "memory": "128Mi"},
    9:  {"resource_mips": 5500, "task_length_mi": 880,    "queue_len": 6,  "deadline": 0.300, "memory": "160Mi"},
    10: {"resource_mips": 5600, "task_length_mi": 1050,   "queue_len": 6,  "deadline": 0.300, "memory": "176Mi"},
    11: {"resource_mips": 5400, "task_length_mi": 3500,   "queue_len": 8,  "deadline": 1.0,   "memory": "192Mi"},
    12: {"resource_mips": 5400, "task_length_mi": 7000,   "queue_len": 10, "deadline": 2.0,   "memory": "256Mi"},
    13: {"resource_mips": 5200, "task_length_mi": 10000,  "queue_len": 12, "deadline": 3.0,   "memory": "320Mi"},
    14: {"resource_mips": 5200, "task_length_mi": 50500,  "queue_len": 15, "deadline": 15.0,  "memory": "512Mi"},
    15: {"resource_mips": 5800, "task_length_mi": 151000, "queue_len": 20, "deadline": 40.0,  "memory": "768Mi"},
}
N_SERVICES = 15
ACTIVE_SERVICES = tuple(sorted(SERVICES_INFO.keys()))
```

سررسیدها بین ۰.۰۳ ثانیه (سبک‌ترین سرویس) تا ۴۰ ثانیه (سنگین‌ترین) هستند، طوری که نقض deadline در هر بخشی از طیف بار معنادار باشد.

قید صحت‌سنجی هنگام بارگذاری: بزرگ‌ترین `resource_mips` بین سرویس‌ها باید از کوچک‌ترین `capacity_mips` بین پروفایل‌ها کمتر باشد؛ در غیر این صورت هیچ سروری نمی‌تواند سنگین‌ترین سرویس را میزبانی کند.

قواعد ثابت درباره‌ی رپلیکاها:
- حداکثر یک رپلیکا از هر سرویس روی هر سرور.
- هر رپلیکا یک صف FIFO واقعی با ظرفیت `queue_len` دارد (پیاده‌سازی به‌صورت صف M/D/1/K با ظرفیت K).
- هر رپلیکا هم‌زمان فقط یک درخواست پردازش می‌کند؛ یک سرور می‌تواند هم‌زمان چند سرویس/رپلیکای مختلف داشته باشد که هرکدام صف و پردازش مستقل خودشان را دارند.

### ۲.۳ زمان اجرا (heterogeneity-aware)

`exec_time` یک عدد ثابت نیست؛ از تقسیم طول کار سرویس (`task_length_mi`، بر حسب میلیون دستور) بر توان مؤثر رپلیکای میزبان محاسبه می‌شود:

```python
def compute_exec_time_sec(service_id, host_mips_per_core):
    svc = SERVICES_INFO[service_id]
    speed_factor = host_mips_per_core / REFERENCE_MIPS_PER_CORE
    effective_mips = svc["resource_mips"] * speed_factor
    return svc["task_length_mi"] / effective_mips
```

چون `large` و `edge_small` هر دو `mips_per_core=2520` دارند، سرعت اجرای این دو پروفایل برابر است؛ تفاوت اصلی بین آن‌ها در ظرفیت کل (تعداد هسته) است، نه سرعت هر هسته. فقط `medium` سرعت متفاوت (۲۷۶۰) دارد. هر بار که رپلیکای یک سرویس روی یک سرور جای‌گذاری می‌شود، `exec_time` مخصوص همان جفت (سرویس، پروفایل سرور) یک‌بار محاسبه و در شیء رپلیکا ذخیره می‌شود.

### ۲.۴ تبدیل واحد MIPS خام به مقدار مؤثر

`resource_mips` هر سرویس نسبت به سرور مرجع (`medium`) تعریف شده است. برای این‌که ظرفیت واقعی یک سرور مشخص درست محاسبه شود، هر جا مصرف CPU یک سرویس با ظرفیت آزاد یک سرور مقایسه می‌شود، ابتدا باید با ضریب سرعت همان سرور تبدیل شود:

```python
def _speed_factor(server):
    return SERVER_PROFILES[server.profile]["mips_per_core"] / REFERENCE_MIPS_PER_CORE

def _cpu_of(server, replica):
    resource_mips = SERVICES_INFO[replica.service_id]["resource_mips"]
    return round(resource_mips * _speed_factor(server))
```

قاعده‌ی طلایی: هر تابعی که ظرفیت آزاد یک سرور را با نیاز CPU یک سرویس مقایسه می‌کند، باید همیشه از همین توابع تبدیل (یا از `can_host`/`free_capacity`) استفاده کند، نه این‌که `resource_mips` خام را مستقیماً با ظرفیت مقایسه کند. فراخوانی‌کننده همیشه مقدار خام را پاس می‌دهد؛ خودِ تابع مقصد آن را به effective تبدیل می‌کند.

---

## ۳. محاسبات جغرافیایی و شبکه

فاصله‌ی بین دو نقطه با فرمول هاورساین محاسبه می‌شود:

```python
EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    dlat = lat2_r - lat1_r; dlon = lon2_r - lon1_r
    a = sin(dlat/2)**2 + cos(lat1_r)*cos(lat2_r)*sin(dlon/2)**2
    c = 2 * asin(min(1.0, sqrt(a)))
    return EARTH_RADIUS_KM * c
```

تأخیر شبکه بر اساس فاصله محاسبه می‌شود:

```python
def network_delay_ms(distance_km, base_latency_ms, k_ms_per_km):
    return base_latency_ms + k_ms_per_km * distance_km
```

ثابت‌های مربوطه: `BASE_LATENCY_MS=2.0`، `K_MS_PER_KM=0.02`، `L0_MS=20.0` (آستانه‌ی پوشش‌دهی برای جای‌گذاری اولیه)، `DISPATCH_OVERHEAD_MS=0.3`، `PROXIMITY_L0_MS=7.0` (آستانه‌ی نزدیکی برای VOILA)، و محدوده‌ی جغرافیایی داده: `LAT_MIN=30.5, LAT_MAX=31.7, LON_MIN=120.7, LON_MAX=122.0`.

**بررسی شدنی‌بودن SLA** (`is_sla_feasible`): قبل از این‌که یک سرور برای میزبانی یک سرویس پذیرفته شود، بررسی می‌شود که آیا با در نظر گرفتن بدترین حالت زمان اجرا و تأخیر شبکه، امکان برآورده‌شدن deadline آن سرویس وجود دارد یا نه. اگر مختصات BTS داده نشود، محاسبه به مسیر محافظه‌کارانه‌تر (بدترین فاصله‌ی ممکن در محدوده‌ی جغرافیایی) می‌رود.

---

## ۴. راه‌اندازی سرد (Cold Start)

وقتی یک رپلیکای تازه‌آماده‌شده اولین درخواست‌هایش را دریافت می‌کند، یک جریمه‌ی زمانی کوچک به زمان پردازش اضافه می‌شود تا اثر گرم‌نشدن (cache/JIT/…) شبیه‌سازی شود:

```python
COLD_START_PENALTY_FRACTION = 0.20
COLD_START_PENALTY_CAP_SEC  = 0.500
COLD_START_WINDOW_RATIO     = 3.0
COLD_START_WINDOW_CAP_SEC   = 10.0

def compute_cold_start_window_sec(service_id, host_mips_per_core):
    et = compute_exec_time_sec(service_id, host_mips_per_core)
    return min(et * COLD_START_WINDOW_RATIO, COLD_START_WINDOW_CAP_SEC)

def compute_cold_start_penalty_sec(service_id, host_mips_per_core):
    svc = SERVICES_INFO[service_id]
    dl = svc["deadline"]
    et = compute_exec_time_sec(service_id, host_mips_per_core)
    penalty_deadline = dl * COLD_START_PENALTY_FRACTION
    penalty_exec = et * 0.50
    penalty = min(penalty_deadline, penalty_exec)
    return min(penalty, COLD_START_PENALTY_CAP_SEC)
```

اگر رپلیکای انتخاب‌شده داخل «پنجره‌ی cold-start» خودش باشد (یعنی از زمان آماده‌شدنش کمتر از `window_sec` گذشته)، مقدار جریمه به زمان پردازش همان درخواست اضافه می‌شود.

---

## ۵. پیکربندی کامل ثابت‌های سیستم

```python
# چرخه‌ی حیات سرور/رپلیکا
BOOT_DELAY_SEC = 30.0
POD_STARTUP_DELAY_SEC = 5.0
GRACEFUL_TERMINATION_DELAY_SEC = 10.0
SERVER_DRAIN_GRACE_SEC = 15.0
MIN_ACTIVE_DURATION_SEC = 300.0
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0

# انرژی
E_BOOT_SERVER_J = 500.0
E_POD_CREATE_J = 20.0

# پایش و تصمیم‌گیری
MONITOR_WINDOW_SEC = 30.0
UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
SUSTAIN_LOW_SEC = 60.0
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0
DECISION_INTERVAL_SEC = 30.0

# ممیزی مستقل تصمیم (عمداً با آستانه‌ی داخلی هیچ الگوریتمی برابر نیست، تا سنجش بی‌طرفانه بماند)
DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.85
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2

# پاداش PPO
PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.08, "w2_deadline": 0.35, "w3_energy": 0.20,
    "w4_load_balance": 0.12, "w5_rejected": 0.25,
}
PPO_PENALTY_PER_ACTION = 0.02
PPO_DEADLINE_FAIRNESS_ALPHA = 0.7

SEED = int(os.environ.get("EOTCH_SEED", "42"))
DATA_DIR = os.environ.get("EOTCH_DATA_DIR", "<project_root>/data/raw")
TRAIN_FILES = ["Data1.csv", "Data2.csv", "Data3.csv"]
TEST_FILE = "Data4.csv"
SECONDS_PER_DAY = 86400
```

ثابت‌های نرمال‌سازی state (کالیبره‌شده روی داده‌ی واقعی با یک اسکریپت کالیبراسیون):

```python
NORM_RESPONSE_TIME_SEC = 1.232
NORM_ENERGY_JOULE      = 4431.91
NORM_ARRIVAL_RATE      = 3.0
NORM_REJECTED_PER_TICK = float(os.environ.get("EOTCH_NORM_REJECTED_PER_TICK", "2.0"))
```

همه‌ی این ثابت‌ها در یک آبجکت پیکربندی مرکزی (frozen) جمع می‌شوند و یک نمونه‌ی سراسری از آن در سراسر پروژه import می‌شود.

---

## ۶. مدل داده و ماشین حالت

```python
class ServerState(Enum): OFF = auto(); BOOTING = auto(); ACTIVE = auto(); DRAINING = auto()
class ReplicaState(Enum): STARTING = auto(); READY = auto(); DRAINING = auto(); TERMINATED = auto()
class RequestStatus(Enum): PENDING = auto(); COMPLETED = auto(); REJECTED_QUEUE_FULL = auto(); REJECTED_NO_REPLICA = auto()
```

ماشین حالت:

```
سرور:   OFF --بوت--> BOOTING --طی BOOT_DELAY_SEC--> ACTIVE
        ACTIVE --تخلیه--> DRAINING --تخلیه کامل--> OFF
رپلیکا: (ساخته‌شدن) --> STARTING --طی POD_STARTUP_DELAY_SEC--> READY
        READY --حذف/مهاجرت/تخلیه--> DRAINING --طی GRACEFUL_TERMINATION_DELAY_SEC--> TERMINATED
```

### رپلیکا (Replica)

فیلدها: `service_id, server_id, queue_len, exec_time, state, created_at, ready_since, drain_started_at, available_at, departures`.

صف واقعی با یک صف (deque) از زمان‌های خروج پیاده می‌شود — معادل دقیق یک صف FIFO تک‌سرور M/D/1/K:

```python
def is_selectable(self): return self.state == ReplicaState.READY

def queue_occupancy(self, now):
    while self.departures and self.departures[0] <= now:
        self.departures.popleft()
    return len(self.departures)

def try_admit(self, arrival_time, cold_start_extra=0.0):
    occ = self.queue_occupancy(arrival_time)
    if occ >= self.queue_len:
        return None
    start = max(arrival_time, self.available_at)
    service_time = self.exec_time + cold_start_extra
    finish = start + service_time
    self.available_at = finish
    self.departures.append(finish)
    return {"queue_enter_time": arrival_time, "service_start_time": start,
            "service_end_time": finish, "wait_time_sec": start - arrival_time}

def is_idle(self, now): return self.queue_occupancy(now) == 0
```

### سرور (Server)

```python
def _speed_factor(self):
    return SERVER_PROFILES[self.profile]["mips_per_core"] / REFERENCE_MIPS_PER_CORE

def _cpu_of(self, replica):
    resource_mips = SERVICES_INFO[replica.service_id]["resource_mips"]
    return round(resource_mips * self._speed_factor())

def used_cpu(self): return sum(self._cpu_of(r) for r in self.hosted_replicas.values())
def free_capacity(self): return self.capacity - self.used_cpu()

def can_host(self, service_id, cpu_demand, bts_lat=None, bts_long=None):
    if service_id in self.hosted_replicas:
        return False
    effective_demand = round(cpu_demand * self._speed_factor())
    if self.free_capacity() < effective_demand:
        return False
    return is_sla_feasible(service_id, self.lat, self.long,
                            SERVER_PROFILES[self.profile]["mips_per_core"],
                            bts_lat=bts_lat, bts_long=bts_long)

def in_cooldown(self, now, cooldown_sec):
    return (now - self.last_transition_time) < cooldown_sec

def instantaneous_utilization(self, now):
    busy_cpu = sum(self._cpu_of(r) for r in self.hosted_replicas.values()
                   if r.state in (READY, DRAINING) and not r.is_idle(now))
    return busy_cpu / self.capacity if self.capacity > 0 else 0.0

def instantaneous_power_w(self, now):
    if self.state == OFF: return 0.0
    if self.state == BOOTING: return self.p_idle
    util = self.instantaneous_utilization(now)
    return self.p_idle + (self.p_max - self.p_idle) * util
```

### درخواست (Request)

فیلدها: `id, bts_lat, bts_long, service_id, arrival_time, assigned_server_id, queue_enter_time, service_start_time, service_end_time, network_delay_ms, routing_delay_sec, wait_time_sec, response_time_sec, deadline_violated, status`.

---

## ۷. چرخه‌ی کامل یک درخواست

1. یک `Request` با موقعیت BTS، سرویس و زمان ورود ساخته می‌شود.
2. تأخیر مسیریابی (رفت‌وبرگشت واقعی بین BTS و دیسپچر مرکزی) محاسبه می‌شود:

```python
dispatcher_lat = mean(s.lat for s in servers.values())
dispatcher_lon = mean(s.long for s in servers.values())
distance_to_dispatcher_km = haversine_km(req.bts_lat, req.bts_long, dispatcher_lat, dispatcher_lon)
one_way_dispatch_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM*distance_to_dispatcher_km + DISPATCH_OVERHEAD_MS
routing_delay_sec = 2 * one_way_dispatch_delay_ms / 1000.0
```

3. با این تأخیر، رویداد «مسیریابی‌شده» در آینده زمان‌بندی می‌شود (نه بلافاصله).
4. وقتی این رویداد اجرا شد، انتخاب‌گر نمونه (بخش ۸) روی رپلیکاهای آماده‌ی همان سرویس اجرا می‌شود.
5. اگر رد شد (نه رپلیکایی موجود بود، نه صفی خالی)، وضعیت رد ثبت می‌شود و به‌عنوان یک نقض deadline هم به‌حساب می‌آید.
6. اگر پذیرفته شد: فاصله و تأخیر شبکه‌ی BTS تا سرور محاسبه می‌شود، نقض نزدیکی جغرافیایی بررسی می‌شود، و اگر رپلیکا در پنجره‌ی cold-start باشد، جریمه‌ی متناظر اعمال می‌شود.
7. فرمول نهایی زمان پاسخ:

```python
response_time_sec = (
    routing_delay_sec
    + 2 * network_delay_ms / 1000.0
    + wait_time_sec
    + (service_end_time - service_start_time)
)
deadline_violated = response_time_sec > SERVICES_INFO[service_id]["deadline"]
```

برای متریک گزارشی، `network_delay_ms` فقط مقدار یک‌طرفه ثبت می‌شود (بدون ضرب در ۲ و بدون تأخیر مسیریابی).

---

## ۸. مسیریابی/انتخاب نمونه (Instance Selection)

این منطق بین همه‌ی الگوریتم‌ها مشترک است:

```python
def select_replica(self, request, candidate_replicas, servers, now, admit_fn=None, occupancy_fn=None):
    if not candidate_replicas:
        return None
    occupancy_fn = occupancy_fn or (lambda r: r.queue_occupancy(now))
    admit_fn = admit_fn or (lambda r: occupancy_fn(r) < r.queue_len)

    dist_pairs = [(haversine_km(request.bts_lat, request.bts_long,
                                 servers[r.server_id].lat, servers[r.server_id].long), r)
                  for r in candidate_replicas]
    min_dist = min(d for d, _ in dist_pairs)

    near_pool = [(d, r) for d, r in dist_pairs
                 if d <= min_dist + 5.0 and occupancy_fn(r) < r.queue_len]
    ordered = sorted(near_pool, key=lambda pair: occupancy_fn(pair[1]) / max(pair[1].queue_len, 1))
    ordered += sorted([p for p in dist_pairs if p not in near_pool], key=lambda pair: pair[0])

    for _, r in ordered:
        if admit_fn(r):
            return r
    return None
```

نکات کلیدی:
- استخر «تقریباً هم‌فاصله»: همه‌ی رپلیکاهایی که حداکثر ۵ کیلومتر بیشتر از نزدیک‌ترین فاصله دارند. در این استخر، معیار انتخاب کمترین نسبت اشغال صف است.
- اگر در استخر نزدیک هیچ گزینه‌ای پذیرفته نشد، جست‌وجو با ترتیب فاصله روی بقیه‌ی کاندیدها ادامه می‌یابد.
- بررسی امکان پذیرش دقیقاً یک‌بار روی هر کاندیدا، به ترتیب رتبه، تا اولین موفقیت انجام می‌شود.
- هر چهار الگوریتم از همین پیاده‌سازی پایه برای مسیریابی لحظه‌ای استفاده می‌کنند.

قابلیت اختیاری PPO با نام «مسیریابی حساس به تأخیر»: به‌جای فاصله‌ی خام، رپلیکایی با کمترین تخمین کل تأخیر انتخاب می‌شود:

```python
occ = occupancy_fn(r)
if occ >= r.queue_len: continue
distance_km = haversine_km(request.bts_lat, request.bts_long, server.lat, server.long)
delay_ms = network_delay_ms(distance_km, BASE_LATENCY_MS, K_MS_PER_KM)
rtt_sec = 2 * delay_ms / 1000.0
est_wait_sec = occ * r.exec_time
est_total_latency = rtt_sec + est_wait_sec + r.exec_time
```

همه‌ی کاندیدها بر اساس این تخمین مرتب می‌شوند و به ترتیب بررسی می‌شوند تا یکی پذیرفته شود.

---

## ۹. مقداردهی اولیه‌ی سیستم (t=0)

### ۹.۱ استراتژی پایه (مشترک بین Greedy، VOILA و HPA)

یک الگوریتم پوشش حریصانه (greedy set cover) برای انتخاب مجموعه‌ی سرورهایی که در ابتدا روشن می‌شوند:

```python
def initial_placement(self, servers, active_bts):
    remaining = set(range(len(active_bts)))
    covers = {}
    for sid, srv in servers.items():
        covered = set()
        for i, (lat, lon) in enumerate(active_bts):
            d_km = haversine_km(lat, lon, srv.lat, srv.long)
            delay = BASE_LATENCY_MS + K_MS_PER_KM * d_km
            if delay <= L0_MS:
                covered.add(i)
        covers[sid] = covered

    selected = []
    while remaining:
        best_sid, best_cover = None, set()
        for sid, covered in covers.items():
            if sid in selected: continue
            inter = covered & remaining
            if len(inter) > len(best_cover):
                best_sid, best_cover = sid, inter
        if best_sid is None or len(best_cover) == 0:
            break
        selected.append(best_sid)
        remaining -= best_cover
        if len(selected) == len(servers):
            break
    if not selected:
        selected = [next(iter(servers))]

    total_cpu_needed = sum(s["resource_mips"] for s in SERVICES_INFO.values())
    remaining_servers = [sid for sid in servers if sid not in selected]
    remaining_servers.sort(key=lambda sid: min(
        haversine_km(servers[sid].lat, servers[sid].long, servers[s2].lat, servers[s2].long)
        for s2 in selected) if selected else 0)
    while sum(servers[sid].capacity for sid in selected) < total_cpu_needed and remaining_servers:
        selected.append(remaining_servers.pop(0))
    return selected
```

«BTS فعال» یعنی هر ایستگاهی که در `MONITOR_WINDOW_SEC` ثانیه‌ی ابتدایی تایم‌لاین حداقل یک درخواست فرستاده باشد. بعد از انتخاب، هر سرور انتخاب‌شده وارد فرآیند بوت می‌شود؛ سپس برای هر یک از ۱۵ سرویس، نزدیک‌ترین سرور از میان همان انتخاب‌شده‌ها که ظرفیت و شرایط SLA را داشته باشد، رپلیکای اولیه را میزبانی می‌کند.

### ۹.۲ استراتژی PPO: بهینه‌سازی با ILP

به‌صورت پیش‌فرض، PPO از یک solver برنامه‌ریزی خطی عدد صحیح (با کتابخانه‌ی pulp، حل‌کننده‌ی CBC) برای انتخاب مجموعه‌ی اولیه‌ی سرورها استفاده می‌کند. هدف چندجزئی:

```
کمینه‌کردن:
  w_count   × (نسبت تعداد سرور روشن به کل سرورها)
+ w_energy  × (نسبت مجموع توان idle سرورهای انتخاب‌شده به مجموع کل)
+ w_distance× (نسبت مجموع وزنی فاصله‌ی هر نقطه‌ی تقاضا تا نزدیک‌ترین سرور پوشش‌دهنده)
```

قیود: هر نقطه‌ی تقاضای پوشش‌پذیر دقیقاً به یک سرور فعال منصوب می‌شود؛ انتصاب فقط به سرور فعال‌شده مجاز است؛ مجموع ظرفیت سرورهای انتخاب‌شده باید حداقل به‌اندازه‌ی مجموع نیاز ۱۵ سرویس باشد. نقاط تقاضا از کل تایم‌لاین سه‌روزه‌ی آموزش استخراج می‌شوند (نه فقط پنجره‌ی ابتدایی). اگر solver جواب پیدا نکرد یا کتابخانه نصب نبود، به همان استراتژی پوشش حریصانه‌ی بخش ۹.۱ برمی‌گردد.

---

## ۱۰. Provisioning پویا سرور و مهاجرت سرویس

### ۱۰.۱ آستانه‌ها و پایش

هر سرور به‌طور مداوم پایش می‌شود: اگر بهره‌وری‌اش از `UTIL_SCALE_UP_THRESHOLD=0.95` بالاتر برود و این وضعیت به‌اندازه‌ی کافی (`SUSTAIN_HIGH_SEC=30`) پایدار بماند، سرور «بیش‌بارشده‌ی پایدار» شناخته می‌شود. اگر بهره‌وری از `UTIL_SCALE_DOWN_THRESHOLD=0.45` پایین‌تر بماند و این وضعیت به‌اندازه‌ی کافی (`SUSTAIN_LOW_SEC=60`) پایدار بماند، «کم‌باره‌ی پایدار» شناخته می‌شود. این سیگنال‌ها برای گزارش و ممیزی محاسبه می‌شوند؛ گیت‌های واقعی که تصمیم‌ها را عملاً مسدود می‌کنند، cooldown (۶۰ ثانیه بعد از هر تغییر حالت روی همان سرور) و حداقل مدت فعال بودن (`MIN_ACTIVE_DURATION_SEC=300`) هستند.

### ۱۰.۲ اعمال تصمیم provisioning

```python
def apply_provisioning(action, snapshot, now):
    applied = False; skip_reason = None
    turn_on_necessary = any_active_server_sustained_overloaded(now) or any_service_capacity_starved(snapshot)
    turn_off_opportunity = any_active_server_sustained_underloaded(now)

    if action.action == TURN_ON and action.server_id is not None:
        s = servers[action.server_id]
        if s.state != OFF: skip_reason = "not_off"
        elif s.in_cooldown(now, COOLDOWN_SEC): skip_reason = "cooldown"
        else:
            start_server_boot(action.server_id)
            metrics.record_scale_action("TURN_ON"); applied = True

    elif action.action == TURN_OFF and action.server_id is not None:
        s = servers[action.server_id]
        n_active = count(state==ACTIVE)
        if s.state != ACTIVE: skip_reason = "not_active"
        elif n_active <= 1: skip_reason = "last_active_server"
        elif s.in_cooldown(now, COOLDOWN_SEC): skip_reason = "cooldown"
        elif (now - s.last_transition_time) < MIN_ACTIVE_DURATION_SEC: skip_reason = "min_active_duration"
        else:
            if start_server_drain(action.server_id):
                metrics.record_scale_action("TURN_OFF"); applied = True
            else:
                skip_reason = "migration_incomplete"

    log("provision_decision", action=action.action.name, server_id=action.server_id,
        applied=applied, skip_reason=skip_reason,
        necessary_turn_on=turn_on_necessary, turn_off_opportunity=turn_off_opportunity)
```

هر بار که یک سرور روشن یا خاموش می‌شود، فارغ از هر عامل دیگر، `COOLDOWN_SEC` ثانیه cooldown روی همان سرور اعمال می‌شود.

### ۱۰.۳ انتخاب پروفایل سرور برای اضطرار

وقتی چند سرور هم‌زمان بیش‌بار هستند و باید تصمیم گرفت که یک سرور خاموش با چه پروفایلی روشن شود، مجموع ظرفیت مورد نیاز سرورهای واقعاً بیش‌بارشده محاسبه و با آستانه‌های پروفایل‌ها مقایسه می‌شود:

```python
def pick_profile_for_overload(overloaded_servers, fallback_capacity):
    large_threshold = SERVER_PROFILES["large"]["capacity_mips"]      # 40320
    medium_threshold = SERVER_PROFILES["medium"]["capacity_mips"]    # 33120
    total = sum(s.capacity for s in overloaded_servers) if overloaded_servers else fallback_capacity
    if total >= large_threshold: return "large"
    if total >= medium_threshold: return "medium"
    return "edge_small"
```

باید فقط لیست سرورهای واقعاً بیش‌بارشده به این تابع پاس داده شود. اگر این لیست خالی است، یک مقدار جایگزین معنادار (مثلاً ظرفیت شلوغ‌ترین سرور فعال) به‌جای مجموع کل فلیت استفاده می‌شود.

### ۱۰.۴ مهاجرت سرویس هنگام تخلیه‌ی سرور

وقتی یک سرور قرار است خاموش شود، ابتدا سرویس‌های میزبانی‌شده‌اش باید جابه‌جا شوند:

```python
def start_server_drain(server_id):
    s = servers[server_id]
    if s.state != ACTIVE: return False
    steps = algorithm.migration_decision(s, servers)   # لیست (service_id, target_server_id)

    # پیش-اعتبارسنجی ظرفیت مقصد: چند مهاجرت هم‌زمان نباید مجموعاً از ظرفیت یک مقصد فراتر روند
    reserved_cpu = defaultdict(int); valid_steps = []
    for step in steps:
        target = servers[step.target_server_id]
        cpu = SERVICES_INFO[step.service_id]["resource_mips"]
        if target.free_capacity() - reserved_cpu[target.id] >= cpu:
            reserved_cpu[target.id] += cpu
            valid_steps.append(step)
    steps = valid_steps

    migrated_services = {step.service_id for step in steps}
    # سرویس‌هایی که تنها رپلیکایشان در کل سیستم روی همین سرور است
    sole_hosted = {svc_id for svc_id, r in s.hosted_replicas.items()
                   if r.state != TERMINATED and not any(
                       other.id != server_id and svc_id in other.hosted_replicas and
                       other.hosted_replicas[svc_id].state != TERMINATED for other in servers.values())}
    unmigrated = sole_hosted - migrated_services
    if unmigrated:
        trigger_emergency_boot(unmigrated, s)
        return False   # تخلیه لغو می‌شود، بعداً دوباره تلاش می‌شود

    s.state = DRAINING; s.drain_started_at = now; s.last_transition_time = now

    for step in steps:
        placed = place_replica(step.target_server_id, step.service_id)   # رپلیکای جدید، حالت STARTING
        if placed is not None:
            pending_migrations[(step.target_server_id, step.service_id)] = server_id

    for r in list(s.hosted_replicas.values()):
        if r.service_id in migrated_services: continue   # منتظر آماده‌شدن مقصد می‌مانند
        start_replica_drain(r)   # رپلیکاهای چندنسخه‌ای فوراً تخلیه می‌شوند

    schedule(now + SERVER_DRAIN_GRACE_SEC, SERVER_DRAIN_DONE, server_id)
    return True
```

مهاجرت به‌صورت بدون قطعی انجام می‌شود: رپلیکای قدیمِ سرویس‌های تک‌نسخه‌ای فقط بعد از آماده‌شدن رپلیکای جدید تخلیه می‌شود. اگر مهاجرت یک سرویس تک‌نسخه‌ای ممکن نبود، نزدیک‌ترین سرور خاموش با ظرفیت کافی به‌صورت اضطراری روشن می‌شود و تخلیه‌ی سرور مبدأ برای این دور متوقف می‌شود؛ چون سرور همچنان فعال می‌ماند، در دور تصمیم‌گیری بعدی دوباره امکان تخلیه بررسی خواهد شد.

پس از سپری‌شدن `SERVER_DRAIN_GRACE_SEC`، اگر همه‌ی رپلیکاهای روی سرور واقعاً حذف شده باشند، سرور به حالت خاموش می‌رود؛ در غیر این صورت این بررسی به تعویق می‌افتد.

---

## ۱۱. کاهش/افزایش مقیاس رپلیکا

رابط مشترک هر الگوریتم برای تصمیم اسکیل هر سرویس در هر تیک، یکی از سه مقدار «بدون تغییر»، «افزایش» یا «کاهش» را برمی‌گرداند. محافظ کاهش مقیاس: فقط رپلیکاهایی که حداقل `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC=120` ثانیه از ساخته‌شدنشان گذشته باشد، کاندید حذف هستند.

**Greedy:**

```python
occ_ratio = avg_queue_occupancy / queue_len if queue_len else 0.0
if occ_ratio > 0.7 or rejection_rate > 0: return SCALE_UP
if occ_ratio < 0.1 and n_ready_replicas > 1: return SCALE_DOWN
return NO_CHANGE
```

**HPA** (هدف بهره‌وری ۷۰٪):

```python
current_replicas = max(n_ready_replicas, 1)
current_util = avg_queue_occupancy / queue_len if queue_len else 0.0
desired = 1 if (current_util <= 0 and rejection_rate <= 0) else ceil(current_replicas * (current_util / 0.70))
desired = max(1, desired)
if rejection_rate > 0:
    desired = max(desired, current_replicas + 1)
if desired > current_replicas: return SCALE_UP
if desired < current_replicas and current_replicas > 1: return SCALE_DOWN
return NO_CHANGE
```

**VOILA** (آستانه‌های `OCC_UP=0.65`، `OCC_DOWN=0.20`، صبر ۳ تیک برای کاهش، حفاظت ۲ تیک برای پایداری نزدیکی، محافظت ۵ تیک بعد از افزایش به‌دلیل نزدیکی):

```python
occ_ratio = avg_queue_occupancy / queue_len if queue_len else 0.0
capacity_violation = occ_ratio > 0.65 or rejection_rate > 0.0
proximity_violation = (not capacity_violation) and proximity_violation_rate > 0.0

if capacity_violation:
    good_streak[svc] = 0; proximity_violation_streak[svc] = 0
    return SCALE_UP

if proximity_violation:
    proximity_violation_streak[svc] += 1
    good_streak[svc] = 0
    if proximity_violation_streak[svc] < 2:
        return NO_CHANGE
    proximity_violation_streak[svc] = 0
    proximity_recent[svc] = 5
    return SCALE_UP

proximity_violation_streak[svc] = 0
good_streak[svc] += 1
proximity_recent[svc] = max(0, proximity_recent.get(svc, 0) - 1)
if proximity_recent.get(svc, 0) > 0:
    return NO_CHANGE
if good_streak[svc] >= 3 and occ_ratio < 0.20 and n_ready_replicas > 1:
    good_streak[svc] = 0
    return SCALE_DOWN
return NO_CHANGE
```

**PPO:** تصمیم از خروجی مدل آموزش‌دیده می‌آید (بخش ۱۳).

### جای‌گذاری رپلیکای جدید هنگام افزایش مقیاس

**Greedy** نزدیک‌ترین سرور به مرکز ثقل تقاضا را انتخاب می‌کند:

```python
centroid = demand_centroid(svc) or (mean_lat_of_active_servers, mean_long_of_active_servers)
candidates = [s for s in active_servers if s.can_host(svc, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
candidates.sort(key=lambda s: haversine_km(centroid[0], centroid[1], s.lat, s.long))
return candidates[0].id if candidates else None
```

**VOILA و PPO** از میان نزدیک‌ترین‌ها به مرکز ثقل، سروری با بیشترین ظرفیت آزاد را انتخاب می‌کنند:

```python
centroid = demand_centroid(svc) or (mean_lat_active, mean_long_active)
candidates = [s for s in active_servers if s.can_host(svc, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long) for s in candidates}
min_dist = min(distances.values())
near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
return max(near_pool, key=lambda s: s.free_capacity()).id if near_pool else None
```

**HPA** عمداً مکان‌ناآگاه است و فقط از روی ظرفیت آزاد انتخاب می‌کند:

```python
candidates = [s for s in active_servers if s.can_host(svc, cpu)]
return max(candidates, key=lambda s: s.free_capacity()).id if candidates else None
```

### مرکز ثقل تقاضا (demand centroid)

هر سرویس یک بافر با حداکثر ۳۰ موقعیت اخیر از درخواست‌های ورودی خودش نگه می‌دارد. مرکز ثقل، نقطه‌ی medoid این مجموعه است — یعنی همان نقطه‌ی واقعی‌ای از میان نقاط که کمترین مجموع فاصله تا بقیه‌ی نقاط را دارد (نه میانگین حسابی مختصات):

```python
def medoid(points):
    if not points: return None
    best, best_cost = points[0], float("inf")
    for p in points:
        cost = sum(haversine_km(p[0], p[1], q[0], q[1]) for q in points)
        if cost < best_cost:
            best, best_cost = p, cost
    return best
```

---

## ۱۲. متریک‌ها و ممیزی تصمیم

مجموعه‌ی معیارهایی که در طول شبیه‌سازی جمع‌آوری می‌شود: زمان‌های پاسخ، تأخیرهای شبکه، فاصله‌ها، تعداد نقض deadline، تعداد کل و کامل‌شده‌ی درخواست‌ها، تعداد رد‌شده‌ها به تفکیک دلیل، تعداد روشن/خاموش شدن سرور، تعداد ساخت/حذف رپلیکا، تعداد افزایش/کاهش مقیاس، و صحت تصمیم‌ها.

```python
def record_request(req):
    total_requests += 1
    if status == COMPLETED:
        completed_requests += 1; response_times.append(response_time_sec)
        distances.append(distance_km); network_delays.append(network_delay_ms)
        if deadline_violated: deadline_violations += 1
    elif status == REJECTED_QUEUE_FULL:
        rejected_queue_full += 1; deadline_violations += 1
    elif status == REJECTED_NO_REPLICA:
        rejected_no_replica += 1; deadline_violations += 1
```

هر تیک، وضعیت لحظه‌ای سرورها هم ثبت می‌شود؛ ضریب تعادل بار (`load_balance_cv`) به‌صورت انحراف‌معیار تقسیم بر میانگین بهره‌وری‌های سرورهای فعال محاسبه می‌شود (اگر فقط یک سرور فعال باشد، صفر است).

در پایان اجرا، خروجی نهایی شامل میانگین‌ها و صدک‌های ۹۵ و ۹۹ زمان پاسخ و تأخیر شبکه، نرخ نقض deadline، انرژی مصرفی تجمعی، میانگین فاصله، میانگین ضریب تعادل بار، تعداد روشن/خاموش‌شدن‌ها، تعداد رد‌شده‌ها به تفکیک دلیل، و صحت تصمیم‌های provisioning است.

برای ممیزی مستقل تصمیم‌ها، آستانه‌های جدا از آستانه‌ی داخلی هر الگوریتم استفاده می‌شود (`DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD=0.85`، `DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD=0.2`)، تا سنجش «آیا این تصمیم واقعاً لازم بود» بی‌طرفانه بماند و به آستانه‌ی خودِ همان الگوریتم وابسته نباشد.

---

## ۱۳. موتور شبیه‌سازی

موتور شبیه‌سازی از نوع discrete-event است: رویدادها در یک صف اولویت‌دار (min-heap) بر اساس زمان مرتب می‌شوند و یکی‌یکی پردازش می‌شوند. انواع رویداد: ورود درخواست، مسیریابی‌شدن درخواست، تیک تصمیم‌گیری دوره‌ای، پایان بوت سرور، پایان تخلیه‌ی سرور، آماده‌شدن رپلیکا، حذف‌شدن رپلیکا، همگام‌سازی انرژی.

روند کلی اجرا:
1. تمام رویدادهای ورود درخواست از داده‌ی خام بارگذاری و در صف قرار می‌گیرند.
2. جای‌گذاری اولیه (بخش ۹) انجام می‌شود و اولین تیک تصمیم‌گیری زمان‌بندی می‌شود.
3. تا زمانی که صف رویدادها خالی نشود یا از یک نقطه‌ی پایانی (کمی بعد از آخرین درخواست، برای اتمام درین‌های در حال انجام) عبور نکنیم، رویدادها یکی‌یکی برداشته و پردازش می‌شوند.
4. قبل از پردازش هر رویداد، انرژی مصرفی همه‌ی سرورها تا همان لحظه integrate می‌شود.
5. هر `DECISION_INTERVAL_SEC=30` ثانیه یک‌بار، تصمیم‌های اسکیل هر سرویس و provisioning سرور از الگوریتم فعال گرفته و اعمال می‌شوند.

هر تغییری که در منطق مشترک تصمیم‌گیری (مثل یک آستانه یا یک بررسی ظرفیت) اعمال شود، باید هم در مسیر شبیه‌سازی و هم در مسیر اجرای واقعی روی Kubernetes به یک شکل پیاده شود، چون هر دو مسیر از همان اشیاء الگوریتم استفاده می‌کنند و فقط حلقه‌ی رویداد (همزمان در برابر ناهمزمان) با هم فرق دارد.

---

## ۱۴. عامل PPO-DRL

### ۱۴.۱ محدوده‌ی تصمیم

هر ۳۰ ثانیه، عامل یک اکشن ترکیبی تولید می‌کند: تصمیم اسکیل برای هر ۱۵ سرویس به‌علاوه‌ی یک تصمیم provisioning برای یک سرور. مسیریابی لحظه‌ای درخواست‌ها خارج از این چرخه و مشترک با بقیه‌ی الگوریتم‌هاست.

### ۱۴.۲ فضای حالت

بردار حالت طولی برابر ۱۵۲ دارد (۱۰ سرور × ۶ ویژگی + ۱۵ سرویس × ۶ ویژگی + ۲ ویژگی سراسری):

```
برای هر سرور (به ترتیب شماره‌ی سرور):
  یک بردار one-hot از حالت (خاموش/بوت/فعال/تخلیه)
  بهره‌وری لحظه‌ای
  نسبت تعداد رپلیکاهای میزبانی‌شده به کل سرویس‌ها

برای هر سرویس (به ترتیب شماره‌ی سرویس):
  نسبت تعداد رپلیکا به تعداد کل سرورها
  نسبت اشغال صف (محدود و نرمال‌شده)
  نرخ نقض deadline اخیر
  نرخ ورود اخیر (نرمال‌شده با NORM_ARRIVAL_RATE)
  نرخ رد‌شدن اخیر
  نرخ نقض نزدیکی جغرافیایی اخیر

سراسری:
  میانگین زمان پاسخ اخیر (نرمال‌شده با NORM_RESPONSE_TIME_SEC)
  انرژی مصرفی اخیر (نرمال‌شده با NORM_ENERGY_JOULE)
```

ثابت‌های نرمال‌سازی با اجرای کامل Greedy روی داده‌ی آموزش و برداشتن صدک‌های آماری (۹۰ام یا ۹۵ام، نه بیشینه‌ی مطلق که یک نقطه‌ی پرت می‌تواند کل مقیاس را خراب کند) به‌دست آمده‌اند.

### ۱۴.۳ فضای اکشن

اکشن یک بردار چند-گسسته است: ۱۵ بعد اول (به ازای هر سرویس: بدون تغییر / افزایش / کاهش مقیاس) و ۱۰ بعد بعدی (به ازای هر سرور: بدون تغییر / روشن‌کردن / خاموش‌کردن). چون در هر تیک فقط یک عمل provisioning واقعی به موتور پاس داده می‌شود، از میان همه‌ی گزینه‌های غیر-«بدون تغییر» بخش سرور، اولویت با روشن‌کردن است؛ اگر چند سرور هم‌زمان انتخاب شده باشند، سروری با کمترین شماره برنده می‌شود.

### ۱۴.۴ محدودسازی اکشن (Action Masking)

قبل از نمونه‌گیری اکشن از توزیع احتمال مدل، گزینه‌های فیزیکاً ناممکن حذف می‌شوند (مثلاً افزایش مقیاس یک سرویسی که در cooldown است یا هیچ سروری با ظرفیت کافی ندارد، یا خاموش‌کردن سروری که تنها سرور فعال است):

```python
def compute_action_masks(engine_or_state, last_snapshot):
    masks = []
    for sid in sorted_service_ids:
        cooldown = (now - service_last_scale_time[sid]) < COOLDOWN_SEC
        cpu = SERVICES_INFO[sid]["resource_mips"]
        centroid = last_snapshot["services"][sid].get("demand_centroid") if last_snapshot else None
        bts_lat, bts_long = centroid if centroid else (None, None)
        can_up = (not cooldown) and any(
            s.state == ACTIVE and s.can_host(sid, cpu, bts_lat=bts_lat, bts_long=bts_long)
            for s in servers.values())
        ready = [r for r in replicas_by_service.get(sid, []) if r.state == READY]
        mature = [r for r in ready if (now - r.created_at) >= MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC]
        can_down = (not cooldown) and len(ready) > 1 and len(mature) > 0
        masks.extend([True, can_up, can_down])

    n_active = count(state == ACTIVE)
    for sid in sorted_server_ids:
        s = servers[sid]
        cooldown = s.in_cooldown(now, COOLDOWN_SEC)
        can_on = (s.state == OFF) and (not cooldown)
        can_off = (s.state == ACTIVE and not cooldown and n_active > 1
                   and (now - s.last_transition_time) >= MIN_ACTIVE_DURATION_SEC)
        masks.extend([True, can_on, can_off])
    return np.array(masks, dtype=bool)
```

این منطق باید در مسیر آموزش، مسیر inference و هر جای دیگری که مدل تصمیم می‌گیرد، دقیقاً یکسان پیاده شود.

### ۱۴.۵ پاداش

پاداش هر تیک، منفیِ یک جریمه‌ی وزن‌دار از چند مؤلفه است:

```python
w = PPO_REWARD_WEIGHTS
active_svcs = [s for s in snapshot["services"].values() if s["recent_arrivals"] > 0]
if active_svcs:
    total_arrivals = sum(s["recent_arrivals"] for s in active_svcs)
    weighted_dv_rate = sum(s["deadline_violation_rate"] * s["recent_arrivals"] for s in active_svcs) / total_arrivals
    unweighted_dv_rate = sum(s["deadline_violation_rate"] for s in active_svcs) / len(active_svcs)
    alpha = PPO_DEADLINE_FAIRNESS_ALPHA   # 0.7
    avg_dv_rate = alpha * weighted_dv_rate + (1 - alpha) * unweighted_dv_rate
else:
    avg_dv_rate = 0.0

active_utils = [s["utilization"] for s in snapshot["servers"].values() if s["state"] == ACTIVE]
load_cv = std(active_utils) / mean(active_utils) if len(active_utils) >= 2 and mean(active_utils) > 0 else 0.0

norm_rt = min(g["avg_response_time_recent"] / NORM_RESPONSE_TIME_SEC, 2.0)
norm_energy = min(g["energy_recent_joule"] / NORM_ENERGY_JOULE, 2.0)
norm_lb = min(load_cv, 2.0)
norm_rejected = min(g["num_rejected_recent"] / NORM_REJECTED_PER_TICK, 2.0)

penalty = (w["w1_response_time"]*norm_rt + w["w2_deadline"]*avg_dv_rate + w["w3_energy"]*norm_energy
           + w["w4_load_balance"]*norm_lb + w["w5_rejected"]*norm_rejected)
penalty += PPO_PENALTY_PER_ACTION * n_actions_applied_this_tick
reward = -penalty
```

مؤلفه‌ی deadline از میانگین‌گیری وزن‌دار (بر اساس تعداد ورودی هر سرویس) و ساده (هر سرویس با وزن یکسان) با ضریب ۰.۷/۰.۳ ترکیب می‌شود، تا هم سرویس‌های پرتردد و هم سرویس‌های کم‌تردد در پاداش دیده شوند.

### ۱۴.۶ محیط آموزش (Gymnasium)

محیط استاندارد Gymnasium با `observation_space` با ابعاد ۱۵۲ و `action_space` چند-گسسته پیاده می‌شود. در هر گام، اکشن مدل به یک تصمیم اسکیل برای هر سرویس و یک تصمیم provisioning ترکیبی تبدیل می‌شود، به موتور شبیه‌سازی پاس داده می‌شود، و پاداش از روی دلتای تعداد اکشن‌های واقعاً اعمال‌شده محاسبه می‌شود.

برای این‌که تصمیم‌های جای‌گذاری و مهاجرت در حین آموزش با همان منطق مسیر inference واقعی هماهنگ بمانند، از همان قاعده‌ی مرکز-ثقل‌محور بخش ۱۱ استفاده می‌شود؛ فقط تصمیم اسکیل/provisioning از بیرون (اکشن مدل) می‌آید.

### ۱۴.۷ تصمیم‌گیرنده در حالت استنتاج (Inference)

```python
class PPOAlgorithm(AlgorithmBase):
    name = "ppo"

    def __init__(self, model_path, deterministic=True, latency_aware_routing=False,
                 use_solver_placement=True, placement_weights=None):
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic
        self._cached_tick_key = None
        self._cached_scale = {}
        self._cached_provision = ProvisionAction(NO_CHANGE)
        self._helper = GreedyAlgorithm()   # برای منطق مهاجرت
        self.latency_aware_routing = latency_aware_routing
        self.use_solver_placement = use_solver_placement

    def _predict_and_cache(self, servers, metrics_snapshot, now):
        if self._cached_tick_key == now:
            return   # جلوگیری از پیش‌بینی تکراری در یک تیک
        obs = build_state_vector(metrics_snapshot, servers)
        action_masks = self._build_action_masks(servers, metrics_snapshot, now)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=self.deterministic)
        self._cached_scale = {sid: SCALE_MAP[action[i]] for i, sid in enumerate(sorted_service_ids)}
        non_noop = [(sid, PROVISION_MAP[action[15+j]]) for j, sid in enumerate(sorted_server_ids)
                    if PROVISION_MAP[action[15+j]] != NO_CHANGE]
        provision = ProvisionAction(NO_CHANGE)
        if non_noop:
            turn_ons = sorted((sid, pt) for sid, pt in non_noop if pt == TURN_ON)
            turn_offs = sorted((sid, pt) for sid, pt in non_noop if pt == TURN_OFF)
            chosen_sid, chosen_ptype = (turn_ons or turn_offs)[0]
            provision = ProvisionAction(chosen_ptype, chosen_sid)
        self._cached_provision = provision
        self._cached_tick_key = now

    def scale_decision(self, service_id, metrics_snapshot):
        return self._cached_scale.get(service_id, NO_CHANGE)

    def provision_decision(self, servers, metrics_snapshot, now):
        self._predict_and_cache(servers, metrics_snapshot, now)
        return self._cached_provision

    def migration_decision(self, draining_server, servers):
        return self._helper.migration_decision(draining_server, servers)
```

منطق جای‌گذاری اولیه از بخش ۹.۲ (solver ILP، با fallback به پوشش حریصانه) استفاده می‌کند. منطق مهاجرت هنگام تخلیه، همان منطق Greedy است اما بر اساس مرکز ثقل تقاضای واقعیِ همان لحظه محاسبه می‌شود.

### ۱۴.۸ آموزش

جریان آموزش سه مرحله دارد:

1. **جمع‌آوری دموی Greedy**: موتور شبیه‌سازی با الگوریتم Greedy روی داده‌ی آموزش اجرا می‌شود و هر تیک، ترکیب (مشاهده، ماسک اکشن، اکشن واقعاً اعمال‌شده) ثبت می‌شود؛ حداکثر ۱۰٬۰۰۰ تیک.
2. **پیش‌آموزش با تقلید رفتار (Behavior Cloning)**: شبکه‌ی سیاست مدل مستقیماً با آنتروپی متقاطع روی این دموها آموزش می‌بیند — ۵۰ epoch، نرخ یادگیری ۵×۱۰⁻⁵، دسته‌های ۶۴تایی.
3. **تنظیم دقیق با یادگیری تقویتی (MaskablePPO)**: ۸ محیط موازی، هرکدام با seed جدا و یک پنجره‌ی تصادفی ۲۴ساعته از تایم‌لاین سه‌روزه‌ی آموزش. تنظیمات: `n_steps=2048، batch_size=256، gamma=0.99، learning_rate=3e-4، ent_coef=0.01`، معماری شبکه با دو لایه‌ی پنهان ۲۵۶تایی برای هر دو شاخه‌ی سیاست و ارزش، و مجموع گام‌های آموزشی پیش‌فرض ۳٬۰۰۰٬۰۰۰.

مدل نهایی و آمار نرمال‌سازی محیط بعد از آموزش ذخیره می‌شوند.

### ۱۴.۹ اجرای استنتاج

روی داده‌ی تست، به‌صورت قطعی (deterministic) و بدون یادگیری بیشتر اجرا می‌شود. کش هر تیک از فراخوانی تکراری مدل جلوگیری می‌کند.

---

## ۱۵. اجرای واقعی روی Kubernetes

### ۱۵.۱ معماری کلی

- **شبیه‌ساز BTS**: دیتاست را با زمان‌بندی واقعی wall-clock بازپخش می‌کند؛ برای هر رکورد ابتدا یک درخواست مسیریابی به دیسپچر مرکزی می‌فرستد، و اگر مسیریابی موفق بود، درخواست پردازش را مستقیماً به آدرس پاد انتخاب‌شده می‌فرستد.
- **دیسپچر مرکزی** (FastAPI، پورت ۹۰۰۰): endpoint اصلی مسیریابی درخواست‌ها را انجام می‌دهد؛ یک endpoint گزارش هم به‌عنوان مسیر جایگزین وجود دارد (مسیر اصلی گزارش تکمیل، صف Redis است).
- **سرویس Worker**: یک ایمیج داکر مشترک برای هر ۱۵ سرویس (تفاوت فقط از طریق متغیرهای محیطی)، با یک نقطه‌ی سلامت و یک نقطه‌ی پردازش که هم‌زمان فقط یک درخواست را می‌پذیرد.
- **موتور بلادرنگ**: معادل ناهمزمان موتور شبیه‌سازی، با چند تسک موازی: چرخه‌ی تصمیم‌گیری هر ۳۰ ثانیه، تخلیه‌ی صف تکمیل هر ۰.۲ ثانیه، نمونه‌برداری بهره‌وری/انرژی هر ۵ ثانیه، جاروب رزروهای منقضی هر ۱۰ ثانیه.
- **Redis**: هماهنگی وضعیت لحظه‌ای بین چرخه‌ی تصمیم‌گیری و دیسپچر.
- **کلاینت Kubernetes**: ساخت/حذف واقعی Deployment و مسدود/بازکردن schedule نودها.

### ۱۵.۲ کلیدهای Redis

```
edge:server:{id}:state              -> وضعیت سرور
edge:replica:{svc}:{srv}:state      -> وضعیت رپلیکا
edge:replica:{svc}:{srv}:pod_ip     -> آدرس IP پاد
edge:replica:{svc}:{srv}:queue      -> شمارنده‌ی اشغال صف
edge:service:{svc}:ready_replicas   -> مجموعه‌ی سرورهای آماده‌ی آن سرویس
edge:reservations:{svc}:{srv}       -> رزروهای صف با زمان انقضا
edge:metrics:completions            -> صف پیام‌های تکمیل
service:{svc}:server:{srv}:busy_seconds_acc -> شمارنده‌ی دقیق ثانیه‌های اشغال برای محاسبه‌ی انرژی
```

### ۱۵.۳ Provisioning و اسکیل در حالت بلادرنگ

منطق روشن/خاموش‌کردن سرور و مهاجرت سرویس، معادل ناهمزمان همان منطق شبیه‌سازی است، با این تفاوت‌ها:
- روشن‌کردن سرور یعنی بازکردن schedule نود، تنظیم وضعیت در Redis، و ثبت انرژی بوت.
- خاموش‌کردن سرور یعنی اجرای مهاجرت (تعیین مقصد، اعتبارسنجی ظرفیت، تشخیص سرویس‌های تک‌نسخه‌ای، روشن‌کردن اضطراری در صورت نیاز)، ساخت Deployment جدید برای رپلیکای مقصد، انتظار تا آماده‌شدن آن، و سپس حذف Deployment رپلیکای قدیمی بعد از خالی‌شدن صفش یا رسیدن به سقف زمانی انتظار.
- تمام تسک‌های پس‌زمینه (حذف رپلیکا، پایش آماده‌شدن) در یک مجموعه‌ی سراسری ثبت می‌شوند تا در پایان اجرا، قبل از نهایی‌کردن متریک‌ها، منتظر تکمیل همه‌شان بمانیم.

### ۱۵.۴ بازیابی اطلاعات جغرافیایی برای متریک نهایی

چون پیام تکمیل از طریق صف Redis می‌آید (بدون دسترسی مستقیم به مختصات BTS در آن لحظه)، فاصله و تأخیر شبکه از یک دیکشنری موقت که هنگام مسیریابی پر شده بازیابی می‌شود. یک جاروب دوره‌ای، ورودی‌های قدیمی این دیکشنری را (برای درخواست‌هایی که هرگز تکمیل گزارش نشدند، مثلاً به‌خاطر کرش یک پاد) پاک می‌کند تا نشتی حافظه رخ ندهد.

### ۱۵.۵ مدیریت Kubernetes

```python
NAMESPACE = "edge-rl"
NODE_LABEL_KEY = "edge-server-id"

def worker_port(service_id): return 8000 + service_id

def resource_mips_to_millicpu(resource_mips):
    return round(resource_mips / REFERENCE_MIPS_PER_CORE * 1000)

def create_deployment(service_id, server_id): ...
def delete_deployment(service_id, server_id): ...
def is_deployment_ready(service_id, server_id) -> bool: ...
def get_pod_ip(service_id, server_id): ...
def cordon_node(server_id): ...     # معادل خاموش‌کردن سرور
def uncordon_node(server_id): ...   # معادل روشن‌کردن سرور
```

هر پاد با `node_selector` به سرور مشخص خودش سنجاق می‌شود، از شبکه‌ی میزبان استفاده می‌کند، و یک probe سلامت روی همان پورت اختصاصی‌اش دارد. هر فراخوانی API که ممکن است خطای گذرا بدهد، با تلاش مجدد و backoff نمایی پوشش داده می‌شود.

### ۱۵.۶ Dockerfile سرویس Worker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${SERVICE_PORT:-8000} --workers 1"]
```

محدودیت «فقط یک درخواست هم‌زمان» داخل خودِ کد اپلیکیشن (با یک semaphore) پیاده می‌شود، نه با تنظیمات سطح سرور وب — چون محدودیت در آن سطح، حتی مسیر بررسی سلامت پاد را هم مسدود می‌کند. پورت از یک متغیر محیطی خوانده می‌شود، چون چند سرویس مختلف ممکن است روی یک نود اجرا شوند و هرکدام باید پورت اختصاصی خودش را داشته باشد.

---

## ۱۶. لاگ ساخت‌یافته

هر رویداد مهم سیستم (ورود درخواست، تکمیل، رد‌شدن، تغییر حالت سرور یا رپلیکا، تصمیم اسکیل/provisioning، مهاجرت، روشن‌کردن اضطراری، جاروب رزرو و غیره) به‌صورت یک رکورد JSON مستقل در یک فایل خط‌به‌خط (JSONL) ثبت می‌شود. هر رکورد شامل نوع رویداد، نام الگوریتم، زمان شبیه‌سازی، زمان واقعی، و فیلدهای اختصاصی همان رویداد است. خروجی هر اجرا شامل یک فایل لاگ رویدادها و یک فایل خلاصه‌ی نتایج نهایی است.

---

## ۱۷. بارگذاری داده

داده‌ی خام شامل چهار فایل CSV است (سه روز برای آموزش، یک روز مستقل برای تست) با ستون‌های شناسه، شناسه‌ی BTS، عرض/طول جغرافیایی، شناسه‌ی سرویس و زمان شروع. هنگام بارگذاری، رکوردهای خارج از محدوده‌ی جغرافیایی معتبر و سرویس‌های غیرفعال حذف می‌شوند، و یک ستون زمان سراسری (بر اساس شماره‌ی روز ضرب در ۸۶۴۰۰ ثانیه به‌علاوه‌ی زمان درون‌روزی) ساخته می‌شود تا چند روز به یک تایم‌لاین پیوسته تبدیل شوند.

---

## ۱۸. رابط مشترک الگوریتم‌ها

هر الگوریتم تصمیم‌گیری از یک کلاس پایه‌ی مشترک ارث‌بری می‌کند که متدهای زیر را تعریف می‌کند:

- `initial_placement` — جای‌گذاری اولیه (پیاده‌سازی مشترک، بخش ۹)
- `select_replica` — مسیریابی لحظه‌ای (پیاده‌سازی مشترک، بخش ۸)
- `select_scale_down_victim` — انتخاب رپلیکای قربانی هنگام کاهش مقیاس (پیش‌فرض: کمترین اشغال)
- `scale_decision` — تصمیم اسکیل هر سرویس (اختصاصی هر الگوریتم، بخش ۱۱)
- `provision_decision` — تصمیم روشن/خاموش‌کردن سرور (اختصاصی هر الگوریتم، بخش ۱۰)
- `select_placement_server` — انتخاب سرور برای رپلیکای جدید (اختصاصی هر الگوریتم، بخش ۱۱)
- `migration_decision` — تعیین مقصد مهاجرت سرویس‌ها هنگام تخلیه‌ی سرور (بخش ۱۰)

برای اضافه‌کردن یک الگوریتم جدید، کافی است کلاسی از این پایه ساخته و متدهای اختصاصی‌اش پیاده شود؛ نه موتور شبیه‌سازی و نه لایه‌ی اتصال به Kubernetes نیازی به تغییر ندارند.

---

## ۱۹. ابزارهای ارزیابی و کالیبراسیون

- **مقایسه‌ی چهارگانه**: هر چهار الگوریتم روی همان داده اجرا می‌شوند، هرکدام لاگ و نتیجه‌ی جدا تولید می‌کنند، و یک جدول خلاصه‌ی مقایسه‌ای ساخته می‌شود.
- **جمع‌بندی چند-seed**: میانگین، انحراف‌معیار، کمینه و بیشینه‌ی هر معیار کلیدی روی چند اجرای PPO با seedهای مختلف گزارش می‌شود.
- **کالیبراسیون ثابت‌ها**: یک اجرای کامل Greedy دنبال می‌شود و توزیع کمیت‌های مختلف (نرخ ورود، زمان پاسخ، انرژی، نرخ رد‌شدن) با میانگین، میانه و صدک‌های آماری خلاصه می‌شود؛ ثابت‌های نرمال‌سازی از صدک ۹۰ یا ۹۵ انتخاب می‌شوند (نه بیشینه‌ی مطلق، که با یک نقطه‌ی پرت می‌تواند کل مقیاس را خراب کند).
- **تحلیل کیفیت تصمیم**: افزایش/کاهش مقیاس‌های اعمال‌شده بر اساس این‌که در یک پنجره‌ی زمانی بعدی واقعاً لازم بودند یا نه دسته‌بندی می‌شوند؛ همچنین چرخه‌های روشن/خاموش کوتاه‌مدت هر سرور شناسایی می‌شوند.
- **تحلیل‌های تکمیلی**: توزیع افزایش مقیاس و نقض deadline به تفکیک سرویس، برای تشخیص این‌که آیا یک الگو ساختاری است (وابسته به سررسیدهای سفت یک سرویس خاص) یا واقعاً ناشی از رفتار الگوریتم.

---

## ۲۰. ساختار پوشه‌ها و نصب

```
edge_rl/
  common/            # مدل داده، پیکربندی، متریک، محاسبات جغرافیایی، لاگ، ساخت بردار حالت
  data/loader.py      # بارگذاری و پیش‌پردازش داده
  simulator/          # موتور discrete-event و رویدادها
  algorithms/
    base.py
    greedy/
    voila/
    hpa/
    ppo/              # محیط، شبکه‌ی سیاست، آموزش، استنتاج، بهینه‌سازی جای‌گذاری
  k8s_adapter/         # کلاینت Kubernetes، وضعیت Redis، موتور بلادرنگ، دیسپچر، سرویس worker
  evaluation/          # مقایسه‌ی چهارگانه، جمع‌بندی چند-seed
  run.py               # نقطه‌ی ورود اصلی
  requirements.txt
```

وابستگی‌های اصلی: `numpy، pandas، matplotlib، gymnasium، stable-baselines3، sb3-contrib، torch، kubernetes، redis، httpx، pulp`. سرویس worker وابستگی سبک‌تر خودش را دارد: `fastapi، uvicorn، pydantic، redis`.

نصب:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

متغیرهای محیطی قابل تنظیم: مسیر پوشه‌ی داده، seed اجرا (پیش‌فرض ۴۲)، و ضریب نرمال‌سازی نرخ رد‌شدن در پاداش PPO (پیش‌فرض ۲.۰).

---

## ۲۱. راهنمای اجرا

اجرای یک الگوریتم روی داده‌ی تست:

```bash
python run.py --algorithm greedy --data test
python run.py --algorithm voila  --data test
python run.py --algorithm hpa    --data test
python run.py --algorithm ppo    --data test
```

آرگومان‌های اصلی `run.py`:

| آرگومان | مقادیر | توضیح |
|---|---|---|
| `--algorithm` | greedy / voila / hpa / ppo | الگوریتم مورد استفاده |
| `--mode` | sim (پیش‌فرض) / k8s | شبیه‌سازی یا اجرای واقعی روی کلاستر |
| `--data` | test (پیش‌فرض) / train | مجموعه‌ی داده |
| `--output-dir` | مسیر دلخواه | محل ذخیره‌ی لاگ و نتایج |
| `--latency-aware-routing` | flag | برای PPO: مسیریابی بر پایه‌ی تخمین کل تأخیر |
| `--no-solver-placement` | flag | برای PPO: غیرفعال‌کردن solver ILP، بازگشت به پوشش حریصانه |

خروجی هر اجرا: یک فایل لاگ رویدادها و یک فایل خلاصه‌ی نتایج.

آموزش PPO:

```bash
python -m algorithms.ppo.train
```

ارزیابی مستقل PPO بعد از آموزش:

```bash
python -m algorithms.ppo.infer
python -m algorithms.ppo.infer --latency-aware-routing --w-energy 2.0
```

مقایسه‌ی هر چهار الگوریتم:

```bash
python -m evaluation.compare_runs --data test
```

جمع‌بندی چند-seed:

```bash
EOTCH_SEED=42 python3 -m algorithms.ppo.train
EOTCH_SEED=42 python3 -m evaluation.compare_runs --output-dir outputs/seed42
python3 -m evaluation.aggregate_seeds --seeds 42 43 44 45 --base-dir outputs



python analyze_decision_quality.py outputs/seed40/ppo_events.jsonl

python diagnose_violations_by_service.py outputs/seed40/ppo_events.jsonl

python analyze_scaleup_by_service.py outputs/seed40/ppo_events.jsonl

python analyze_necessity_by_service.py outputs/seed40/ppo_events.jsonl
python analyze_decision_quality.py outputs/seed40/voila_events.jsonl
python analyze_decision_quality.py outputs/seed40/hpa_events.jsonl


```

اجرای واقعی روی Kubernetes (دو ترمینال):

```bash
# ترمینال ۱ — کنترل‌پلین و موتور
uvicorn k8s_adapter.dispatcher_api:app --port 9000
# ترمینال ۲ — مولد ترافیک
python3 -m k8s_adapter.bts_simulator
```

یا معادل خودکار همه‌چیز در یک فرمان:

```bash
python run.py --algorithm greedy --mode k8s --data test
```

---

## ۲۲. ترتیب منطقی ساخت پروژه از صفر

1. **هسته‌ی مشترک**: مدل داده، پیکربندی مرکزی، محاسبات جغرافیایی، لاگ ساخت‌یافته؛ همین‌جا محاسبه‌ی زمان اجرا و منطق cold-start هم اضافه می‌شود.
2. **بارگذاری داده**: پیش‌پردازش دیتاست خام و ساخت تایم‌لاین پیوسته.
3. **رابط مشترک الگوریتم‌ها**: جای‌گذاری اولیه و مسیریابی مشترک.
4. **موتور شبیه‌سازی**: چرخه‌ی رویداد کامل، شامل چرخه‌ی کامل یک درخواست، provisioning/migration، و اسکیل خودکار. این بزرگ‌ترین و حیاتی‌ترین بخش است؛ بهتر است ابتدا با یک الگوریتم بسیار ساده (که همیشه «بدون تغییر» برمی‌گرداند) تست شود تا از سلامت چرخه‌ی رویداد پایه مطمئن شویم.
5. **جمع‌آوری متریک**.
6. **چهار الگوریتم، از ساده به پیچیده**: ابتدا Greedy، سپس HPA (فرمول ساده و ثابت)، سپس VOILA (منطق medoid/proximity)، و در آخر PPO (که نیازمند ساخت بردار حالت، محیط Gymnasium، و آموزش با warm-start است).
7. **ابزارهای ارزیابی**: بعد از این‌که هر چهار الگوریتم قابل‌اجرا هستند.
8. **لایه‌ی اتصال به Kubernetes**: در نهایت، برای پورت‌کردن همان اشیاء تصمیم‌گیری (بدون تغییر) به یک محیط بلادرنگ واقعی با هماهنگی Redis و دیسپچر HTTP واقعی.

اصل کلیدی در تمام این مسیر: هیچ منطق تصمیم‌گیری‌ای دو بار نوشته نمی‌شود. موتور شبیه‌سازی و موتور بلادرنگ هر دو از همان اشیاء الگوریتم استفاده می‌کنند؛ ساخت بردار حالت فقط یک‌بار تعریف می‌شود؛ ثابت‌های نرمال‌سازی با کالیبراسیون روی داده‌ی واقعی به‌دست می‌آیند، نه با حدس دستی. هر تغییری در یک منطق مشترک (مثل انتخاب رپلیکا، تصمیم مهاجرت، یا آستانه‌های provisioning) باید هم‌زمان در مسیر شبیه‌سازی و مسیر Kubernetes واقعی اعمال شود، وگرنه رفتار این دو مسیر کم‌کم از هم فاصله می‌گیرد و مقایسه‌ی نتایج بی‌اعتبار می‌شود.

---

## ۲۳. جدول مرجع ثابت‌های عددی

| ثابت | مقدار |
|---|---|
| `BOOT_DELAY_SEC` | 30.0 |
| `POD_STARTUP_DELAY_SEC` | 5.0 |
| `GRACEFUL_TERMINATION_DELAY_SEC` | 10.0 |
| `SERVER_DRAIN_GRACE_SEC` | 15.0 |
| `MIN_ACTIVE_DURATION_SEC` | 300.0 |
| `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC` | 120.0 |
| `COLD_START_PENALTY_FRACTION` | 0.20 |
| `COLD_START_PENALTY_CAP_SEC` | 0.500 |
| `COLD_START_WINDOW_RATIO` | 3.0 |
| `COLD_START_WINDOW_CAP_SEC` | 10.0 |
| `E_BOOT_SERVER_J` | 500.0 |
| `E_POD_CREATE_J` | 20.0 |
| `MONITOR_WINDOW_SEC` | 30.0 |
| `UTIL_SCALE_UP_THRESHOLD` | 0.95 |
| `UTIL_SCALE_DOWN_THRESHOLD` | 0.45 |
| `SUSTAIN_LOW_SEC` | 60.0 |
| `SUSTAIN_HIGH_SEC` | 30.0 |
| `COOLDOWN_SEC` | 60.0 |
| `DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD` | 0.85 |
| `DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD` | 0.2 |
| `DECISION_INTERVAL_SEC` | 30.0 |
| `BASE_LATENCY_MS` | 2.0 |
| `K_MS_PER_KM` | 0.02 |
| `L0_MS` | 20.0 |
| `DISPATCH_OVERHEAD_MS` | 0.3 |
| `PROXIMITY_L0_MS` | 7.0 |
| `LAT_MIN, LAT_MAX` | 30.5, 31.7 |
| `LON_MIN, LON_MAX` | 120.7, 122.0 |
| وزن پاداش — زمان پاسخ | 0.08 |
| وزن پاداش — نقض deadline | 0.35 |
| وزن پاداش — انرژی | 0.20 |
| وزن پاداش — تعادل بار | 0.12 |
| وزن پاداش — نرخ رد‌شدن | 0.25 |
| جریمه‌ی هر اکشن اعمال‌شده | 0.02 |
| ضریب انصاف deadline (α) | 0.7 |
| نرمال‌ساز زمان پاسخ (ثانیه) | 1.232 |
| نرمال‌ساز انرژی (ژول) | 4431.91 |
| نرمال‌ساز نرخ ورود | 3.0 |
| نرمال‌ساز نرخ رد‌شدن هر تیک | 2.0 |
| VOILA — آستانه‌ی افزایش مقیاس | 0.65 |
| VOILA — آستانه‌ی کاهش مقیاس | 0.20 |
| VOILA — صبر برای کاهش مقیاس (تیک) | 3 |
| VOILA — پایداری نزدیکی (تیک) | 2 |
| VOILA — محافظت بعد از نزدیکی (تیک) | 5 |
| HPA — هدف بهره‌وری | 0.70 |
| Greedy — آستانه‌ی افزایش مقیاس | 0.7 |
| Greedy — آستانه‌ی کاهش مقیاس | 0.1 |
| شعاع استخر «نزدیک» در انتخاب رپلیکا | +5.0 کیلومتر از نزدیک‌ترین |
| طول بردار حالت PPO | 152 (=۱۰×۶ + ۱۵×۶ + ۲) |
| PPO — `n_steps` | 2048 |
| PPO — `batch_size` | 256 |
| PPO — `gamma` | 0.99 |
| PPO — نرخ یادگیری | 3e-4 |
| PPO — ضریب آنتروپی | 0.01 |
| PPO — معماری شبکه | دو لایه‌ی ۲۵۶تایی برای هر شاخه |
| PPO — کل گام‌های آموزش (پیش‌فرض) | 3,000,000 |
| PPO — تعداد محیط موازی | 8 |
| BC warm-start — تعداد epoch | 50 |
| BC warm-start — نرخ یادگیری | 5e-5 |
| BC warm-start — اندازه‌ی batch | 64 |
| BC warm-start — حداکثر تیک | 10,000 |
| بازه‌ی نمونه‌برداری بهره‌وری در k8s | 5.0 ثانیه |
| بازه‌ی جاروب رزرو در k8s | 10.0 ثانیه |
| مهلت اعتبار رزرو در k8s | مهلت سرویس + 5 ثانیه |
| بازه‌ی پایش صف تکمیل در k8s | 0.2 ثانیه |
| پورت دیسپچر | 9000 |
| پورت هر سرویس worker | 8000 + شناسه‌ی سرویس |

---

**جمع‌بندی:** این مستند تمام منابع، فرمول‌ها، ثابت‌ها، و منطق تصمیم‌گیری چهار الگوریتم (Greedy، HPA، VOILA، PPO-DRL)، هم در مسیر شبیه‌سازی discrete-event و هم در مسیر اجرای واقعی روی Kubernetes، را پوشش می‌دهد.
