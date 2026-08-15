# مشخصات فنی کامل پروژه `edge_rl` — سند بازسازی از صفر

> این سند طوری نوشته شده که یک هوش مصنوعی/توسعه‌دهنده بدون هیچ دسترسی دیگری به سورس یا تاریخچه‌ی پروژه، بتواند دقیقاً همین سیستم را از صفر پیاده‌سازی کند. هر فرمول، هر ثابت، هر رفتار لبه‌ای (edge case) و هر تصمیم طراحی که در کد نهایی وجود دارد اینجا صریح نوشته شده — نه فقط رفتار «نهایی درست»، بلکه دلیل و جایگزین‌های رد‌شده هم آورده شده تا دوباره پیاده‌سازی همان باگ‌ها تکرار نشود.

---

## بخش ۰: ایده‌ی کلی و اهداف

یک سیستم مدیریت منابع محاسباتی لبه (Edge Computing Resource Management) که:

- ۱۰ سرور لبه‌ی heterogeneous در نقاط مختلف یک شهر (بر پایه‌ی دیتاست واقعی BTS شانگهای) دارد.
- ۱۵ نوع سرویس مختلف (بار CPU متفاوت، deadline متفاوت) باید روی این سرورها اجرا شوند.
- درخواست‌ها از BTSهای مختلف با مختصات جغرافیایی می‌رسند و باید در کمترین زمان به یک replica مناسب مسیریابی شوند.
- چهار الگوریتم تصمیم‌گیری متفاوت (Greedy baseline، Kubernetes-HPA، VOILA، PPO-DRL) باید با معیارهای یکسان قابل مقایسه باشند.
- دو مسیر اجرا وجود دارد: **شبیه‌سازی discrete-event سریع** (برای آموزش/ارزیابی) و **اجرای واقعی روی یک خوشه‌ی Kubernetes** (با Redis برای هماهنگی state و FastAPI برای دیسپچر).
- اصل طراحی حیاتی: **هیچ منطق تصمیم‌گیری دو بار نوشته نمی‌شود.** هر دو مسیر (شبیه‌سازی و real-time) از همان اشیاء الگوریتم (`AlgorithmBase` subclasses) استفاده می‌کنند؛ فقط حلقه‌ی رویداد فرق دارد (sync heap-based در برابر async/asyncio). هر تغییری در منطق مشترک (مثل یک بررسی ظرفیت یا یک آستانه) باید در **هر دو** پیاده‌سازی موازی (`simulator/engine.py` و `k8s_adapter/realtime_dispatcher.py`) اعمال شود.

---

## بخش ۱: منابع سیستم — تعاریف دقیق

### ۱.۱ پروفایل‌های سرور

```python
SERVER_PROFILES = {
    "edge_small": {"n_cores": 8,  "mips_per_core": 2520, "capacity_mips": 20160, "p_idle": 28, "p_max": 113},
    "medium":     {"n_cores": 12, "mips_per_core": 2760, "capacity_mips": 33120, "p_idle": 30, "p_max": 130},
    "large":      {"n_cores": 16, "mips_per_core": 2520, "capacity_mips": 40320, "p_idle": 33, "p_max": 148},
}
REFERENCE_MIPS_PER_CORE = SERVER_PROFILES["medium"]["mips_per_core"]  # = 2760
```

`capacity_mips = n_cores * mips_per_core` (این رابطه فقط توضیحی است؛ در کد این مقادیر مستقیم به‌صورت literal نوشته می‌شوند، نه محاسبه‌شده).

### ۱.۲ سرورها (۱۰ عدد، id از ۱ تا ۱۰)

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
```

پروفایل هر سرور با نگاشت معکوس از `capacity_mips` به نام پروفایل استخراج می‌شود (یک بار در زمان بارگذاری config):
```python
_CAPACITY_TO_PROFILE = {p["capacity_mips"]: name for name, p in SERVER_PROFILES.items()}
for sid, info in SERVER_INFO.items():
    info["profile"] = _CAPACITY_TO_PROFILE[info["capacity_mips"]]
N_SERVERS = 10
```

قید سخت جای‌گذاری (همیشه اجرا می‌شود در `can_host`):
```
sum(resource_mips[svc] for svc in services on server) <= capacity_mips[server]   # با تبدیل واحد؛ بخش ۱.۵
```

### ۱.۳ سرویس‌ها (۱۵ عدد، id از ۱ تا ۱۵)

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
ACTIVE_SERVICES = tuple(sorted(SERVICES_INFO.keys()))   # (1,2,...,15)
```

اعتبارسنجی هنگام بارگذاری ماژول config:
```python
_max_single_service = max(s["resource_mips"] for s in SERVICES_INFO.values())
_min_server_capacity = min(p["capacity_mips"] for p in SERVER_PROFILES.values())
assert _max_single_service <= _min_server_capacity
```

### ۱.۴ محاسبه‌ی زمان اجرا (heterogeneity-aware)

```python
def compute_exec_time_sec(service_id, host_mips_per_core):
    svc = SERVICES_INFO[service_id]
    speed_factor = host_mips_per_core / REFERENCE_MIPS_PER_CORE
    effective_mips = svc["resource_mips"] * speed_factor
    return svc["task_length_mi"] / effective_mips
```

این یعنی زمان اجرا روی سرور `large` کمتر (سریع‌تر) از روی `edge_small` است (چون `mips_per_core` هر دو ۲۵۲۰ است ولی... **دقت**: `large` و `edge_small` هر دو `mips_per_core=2520` دارند، پس `speed_factor` برایشان برابر است؛ فقط `medium` سرعت متفاوت (۲۷۶۰) دارد. تفاوت اصلی بین `large` و `edge_small` در ظرفیت کل (تعداد هسته) است، نه سرعت هر هسته).

### ۱.۵ تبدیل واحد MIPS خام به effective/host-specific (نکته‌ی حیاتی، منبع باگ‌های رایج)

`resource_mips` هر سرویس نسبت به یک **سرور مرجع** (`medium`, `mips_per_core=2760`) تعریف شده. برای این‌که ظرفیت واقعی یک سرور مشخص (که ممکن است `edge_small` یا `large` باشد) درست محاسبه شود، هر جا `resource_mips` خام با `free_capacity()`/`used_cpu()` یک سرور مشخص مقایسه می‌شود، **باید** ابتدا با `_speed_factor()` همان سرور تبدیل شود:

```python
def _speed_factor(server):
    return SERVER_PROFILES[server.profile]["mips_per_core"] / REFERENCE_MIPS_PER_CORE

def _cpu_of(server, replica):
    resource_mips = SERVICES_INFO[replica.service_id]["resource_mips"]
    return round(resource_mips * _speed_factor(server))
```

**این تبدیل باید در همه‌ی مکان‌های زیر رعایت شود** (فراموش‌کردنش یک کلاس کامل باگ تولید می‌کند که فقط روی سرورهای `edge_small`/`large` بروز می‌کند، نه `medium`، چون فقط `medium` سرعت مرجع دارد و `speed_factor=1.0` می‌شود):

1. `Server.used_cpu()` = `sum(_cpu_of(r) for r in hosted_replicas.values())`
2. `Server.free_capacity()` = `capacity - used_cpu()`
3. `Server.can_host(...)`: قبل از مقایسه با `free_capacity()`، `cpu_demand` باید با `round(cpu_demand * _speed_factor())` به `effective_demand` تبدیل شود، و **این effective_demand** (نه خام) با `free_capacity()` مقایسه شود.
4. `Server.instantaneous_utilization(now)`: مجموع effective_mips رپلیکاهای واقعاً busy تقسیم بر capacity.
5. هر جای دیگری که یک مقدار CPU روی یک سرور مشخص با ظرفیت آن سرور مقایسه می‌شود (مثلاً pre-check‌های دستی قبل از drain/emergency-boot، شمارنده‌های utilization لحظه‌ای در realtime dispatcher).

**قاعده‌ی طلایی برای پیاده‌سازی صحیح:** هر تابعی که ظرفیت آزاد یک سرور را با نیاز CPU یک سرویس مقایسه می‌کند باید همیشه از `can_host()`/`free_capacity()`/`_cpu_of()` استفاده کند، نه این‌که خودش دوباره `resource_mips` خام را مستقیماً با `free_capacity()` مقایسه کند (این یک ناهم‌خوانی واحد است که منجر به رد نادرست عملیات مجاز می‌شود).

---

## بخش ۲: مدل‌های داده (`common/models.py`)

### ۲.۱ Enumها

```python
class ServerState(Enum): OFF = auto(); BOOTING = auto(); ACTIVE = auto(); DRAINING = auto()
class ReplicaState(Enum): STARTING = auto(); READY = auto(); DRAINING = auto(); TERMINATED = auto()
class RequestStatus(Enum): PENDING = auto(); COMPLETED = auto(); REJECTED_QUEUE_FULL = auto(); REJECTED_NO_REPLICA = auto()
```

ماشین حالت:
```
سرور:  OFF --boot trigger--> BOOTING --boot_delay_sec--> ACTIVE
       ACTIVE --drain trigger--> DRAINING --drain کامل--> OFF
رپلیکا: (ساخته‌شدن) --> STARTING --pod_startup_delay_sec--> READY
        READY --حذف/migrate/drain--> DRAINING --graceful_termination_delay_sec--> TERMINATED
```

### ۲.۲ `Replica` (dataclass)

فیلدها: `service_id, server_id, queue_len, exec_time, state=STARTING, created_at=0.0, ready_since=None, drain_started_at=None, available_at=0.0, departures=deque()`

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

این معادل دقیق یک صف FIFO تک‌سرور M/D/1/K با K=queue_len است — بدون شبیه‌ساز event-driven جداگانه، فقط با یک `deque` از زمان خروج‌ها.

### ۲.۳ `Server` (dataclass)

فیلدها: `id, profile, lat, long, capacity, p_idle, p_max, state=OFF, hosted_replicas={}, boot_started_at=None, drain_started_at=None, last_transition_time=-1e18, cumulative_energy_joule=0.0, cumulative_busy_cpu_seconds=0.0, num_boots=0, num_shutdowns=0`

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
        return False   # هر سرور حداکثر ۱ رپلیکا از هر سرویس
    effective_demand = round(cpu_demand * self._speed_factor())
    if self.free_capacity() < effective_demand:
        return False
    return is_sla_feasible(service_id, self.lat, self.long,
                            SERVER_PROFILES[self.profile]["mips_per_core"],
                            bts_lat=bts_lat, bts_long=bts_long)

def in_cooldown(self, now, cooldown_sec):
    return (now - self.last_transition_time) < cooldown_sec

def instantaneous_utilization(self, now):
    busy_cpu = 0
    for r in self.hosted_replicas.values():
        if r.state in (READY, DRAINING) and not r.is_idle(now):
            busy_cpu += self._cpu_of(r)
    return busy_cpu / self.capacity if self.capacity > 0 else 0.0

def instantaneous_power_w(self, now):
    if self.state == OFF: return 0.0
    if self.state == BOOTING: return self.p_idle
    util = self.instantaneous_utilization(now)
    return self.p_idle + (self.p_max - self.p_idle) * util
```

نکات کلیدی `can_host`:
- اگر سرویس از قبل روی این سرور هست، `False` (حداکثر یک رپلیکا از هر سرویس روی هر سرور).
- `cpu_demand` که پاس داده می‌شود همیشه `resource_mips` **خام** سرویس است؛ خودِ تابع آن را به effective تبدیل می‌کند.
- اگر `bts_lat`/`bts_long` داده نشود، `is_sla_feasible` به مسیر محافظه‌کارانه (بدترین فاصله‌ی ممکن) می‌رود.

### ۲.۴ `Request` (dataclass)

فیلدها: `id, bts_lat, bts_long, service_id, arrival_time, assigned_server_id=None, queue_enter_time=None, service_start_time=None, service_end_time=None, network_delay_ms=0.0, routing_delay_sec=0.0, wait_time_sec=0.0, response_time_sec=0.0, deadline_violated=False, status=PENDING`

(یک فیلد غیررسمی `_distance_km` هم روی instance ست می‌شود، نه بخشی از dataclass تعریف‌شده — با `getattr(req, "_distance_km", 0.0)` خوانده می‌شود.)

---

## بخش ۳: محاسبات جغرافیایی و شبکه (`common/geo.py`)

```python
EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r = radians(lat1), radians(lon1)
    lat2_r, lon2_r = radians(lat2), radians(lon2)
    dlat = lat2_r - lat1_r; dlon = lon2_r - lon1_r
    a = sin(dlat/2)**2 + cos(lat1_r)*cos(lat2_r)*sin(dlon/2)**2
    c = 2 * asin(min(1.0, sqrt(a)))
    return EARTH_RADIUS_KM * c

def network_delay_ms(distance_km, base_latency_ms, k_ms_per_km):
    return base_latency_ms + k_ms_per_km * distance_km
```

ثابت‌های شبکه:
```python
BASE_LATENCY_MS = 2.0
K_MS_PER_KM = 0.02
L0_MS = 20.0                # آستانه‌ی پوشش initial placement
DISPATCH_OVERHEAD_MS = 0.3  # سربار پردازش داخلی دیسپچر
PROXIMITY_L0_MS = 7.0       # آستانه‌ی proximity violation — روی RTT سنجیده می‌شود، نه یک‌طرفه!
LAT_MIN, LAT_MAX = 30.5, 31.7
LON_MIN, LON_MAX = 120.7, 122.0
```

**نکته‌ی حیاتی proximity:** با محدوده‌ی جغرافیایی داده (بیشینه‌ی فاصله‌ی ممکن BTS↔سرور ≈ ۱۸۲ کیلومتر → بیشینه‌ی `network_delay_ms` یک‌طرفه ≈ ۵.۶ms)، اگر آستانه‌ی proximity روی مقدار یک‌طرفه سنجیده شود (نه رفت‌وبرگشت ×۲)، هرگز نقض نمی‌شود و سیگنال VOILA کاملاً خاموش می‌ماند. **همیشه** با `2 * network_delay_ms >= PROXIMITY_L0_MS` مقایسه شود.

**نکته‌ی L0_MS:** با همین محدوده‌ی جغرافیایی، آستانه‌ی `L0_MS=20.0` عملاً هیچ‌وقت رد نمی‌شود (هر سرور هر BTS معتبر را «می‌پوشاند»). یعنی در حلقه‌ی پوشش حریصانه‌ی initial placement (بخش ۵)، عملاً انتخاب سرور بر پایه‌ی ظرفیت باقی‌مانده پیش می‌رود، نه پوشش واقعی متفاوت. این رفتار **باید حفظ شود**، نه اصلاح — بخشی از رفتار تعریف‌شده‌ی سیستم است.

---

## بخش ۴: بررسی شدنی‌بودن SLA (`is_sla_feasible`)

```python
DISPATCHER_LAT = mean(info["lat"] for info in SERVER_INFO.values())   # ≈ 31.185
DISPATCHER_LON = mean(info["long"] for info in SERVER_INFO.values())  # ≈ 121.431

def is_sla_feasible(service_id, server_lat, server_long, server_mips_per_core,
                     bts_lat=None, bts_long=None):
    svc = SERVICES_INFO[service_id]
    et = compute_exec_time_sec(service_id, server_mips_per_core)

    if bts_lat is not None and bts_long is not None:
        dist_to_dispatcher_km = haversine_km(bts_lat, bts_long, DISPATCHER_LAT, DISPATCHER_LON)
        dist_to_server_km = haversine_km(bts_lat, bts_long, server_lat, server_long)
    else:
        # محافظه‌کارانه: بدترین فاصله از هر ۴ گوشه‌ی محدوده‌ی جغرافیایی
        corners = [(LAT_MIN,LON_MIN),(LAT_MIN,LON_MAX),(LAT_MAX,LON_MIN),(LAT_MAX,LON_MAX)]
        dist_to_dispatcher_km = max(haversine_km(c[0],c[1],DISPATCHER_LAT,DISPATCHER_LON) for c in corners)
        dist_to_server_km = max(haversine_km(c[0],c[1],server_lat,server_long) for c in corners)

    dispatcher_rtt_sec = 2 * (BASE_LATENCY_MS + K_MS_PER_KM*dist_to_dispatcher_km + DISPATCH_OVERHEAD_MS) / 1000.0
    server_rtt_sec = 2 * (BASE_LATENCY_MS + K_MS_PER_KM*dist_to_server_km) / 1000.0
    min_response_sec = dispatcher_rtt_sec + server_rtt_sec + et
    return min_response_sec <= svc["deadline"]
```

نکته: `DISPATCHER_LAT`/`LON` **همان** موقعیتی است که موتور شبیه‌سازی هنگام محاسبه‌ی `routing_delay_sec` واقعی استفاده می‌کند (بخش ۷) — این دو باید همیشه از یک ثابت مشترک بیایند، وگرنه چک SLA و رفتار واقعی از هم واگرا می‌شوند.

---

## بخش ۵: cold-start

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

قاعده‌ی اعمال (در `_handle_routed` موتور شبیه‌سازی، بخش ۷): وقتی رپلیکای انتخاب‌شده `ready_since` دارد و `(now - ready_since) <= window_sec` است، مقدار `compute_cold_start_penalty_sec(...)` به `exec_time` همان درخواست اضافه می‌شود (به‌عنوان `cold_start_extra` در `try_admit`).

---

## بخش ۶: پارامترهای پیکربندی کامل (`common/config.py`)

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

# Initial placement
MONITOR_WINDOW_SEC = 30.0     # فقط برای t=0؛ کاملاً مستقل از DECISION_INTERVAL_SEC با وجود مقدار تصادفاً یکسان

# Scaling/Provisioning
UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_LOW_SEC = 60.0
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0
DECISION_INTERVAL_SEC = 30.0

# ممیزی مستقل تصمیم (باید با آستانه‌ی داخلی هیچ الگوریتمی برابر نباشد!)
DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.85
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2

# PPO Reward
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

ثابت‌های نرمال‌سازی state (کالیبره‌شده با `calibrate_constants.py` روی داده‌ی واقعی، در `common/state_builder.py`):
```python
NORM_RESPONSE_TIME_SEC = 1.232
NORM_ENERGY_JOULE      = 4431.91
NORM_ARRIVAL_RATE      = 3.0
```
و در `algorithms/ppo/env.py`:
```python
NORM_REJECTED_PER_TICK = float(os.environ.get("EOTCH_NORM_REJECTED_PER_TICK", "2.0"))
```
(نکته: مقدار پیش‌فرض در `env.py` واقعاً ۲.۰ است، نه ۶.۰ که در جای دیگری از مستندات به‌عنوان مقدار پیشنهادی کالیبراسیون آمده — برای پیاده‌سازی، از env var با پیش‌فرض `2.0` استفاده کن.)

پیکربندی همه‌ی این‌ها باید در یک آبجکت `frozen dataclass` به نام `Config` جمع شود و یک نمونه‌ی سراسری `CFG = Config()` ساخته شود که همه‌ی ماژول‌های دیگر از آن import می‌کنند (`from common.config import CFG`).

---

## بخش ۷: چرخه‌ی کامل یک درخواست (مسیر شبیه‌سازی)

### ۷.۱ ورود و مسیریابی

1. رویداد ورود در `global_start_sec` می‌رسد؛ یک `Request` با `bts_lat/long`, `service_id`, `arrival_time=now` ساخته می‌شود.
2. محاسبه‌ی تأخیر مسیریابی (رفت‌وبرگشت واقعی BTS↔دیسپچر):
```python
dispatcher_lat = mean(s.lat for s in servers.values())   # میانگین موقعیت هر ۱۰ سرور — دقیقاً همان DISPATCHER_LAT
dispatcher_lon = mean(s.long for s in servers.values())
distance_to_dispatcher_km = haversine_km(req.bts_lat, req.bts_long, dispatcher_lat, dispatcher_lon)
one_way_dispatch_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM*distance_to_dispatcher_km + DISPATCH_OVERHEAD_MS
routing_delay_sec = 2 * one_way_dispatch_delay_ms / 1000.0
```
3. یک رویداد `REQUEST_ROUTED` در `now + routing_delay_sec` زمان‌بندی می‌شود (نه بلافاصله).
4. وقتی این رویداد رخ داد: **Instance Selector** (بخش ۸) روی رپلیکاهای READY همان سرویس اجرا می‌شود.
5. اگر رد شد (نه رپلیکا موجود، نه صف خالی) → وضعیت `REJECTED_NO_REPLICA`/`REJECTED_QUEUE_FULL`؛ ثبت در متریک به‌عنوان یک نقض deadline هم (`_tick_violated` افزایش می‌یابد).
6. اگر پذیرفته شد: فاصله و تأخیر شبکه‌ی BTS↔سرور محاسبه می‌شود؛ proximity violation چک می‌شود (بخش ۳)؛ اگر رپلیکا در پنجره‌ی cold-start است، `cold_start_extra` محاسبه می‌شود؛ `try_admit` صدا زده می‌شود.
7. اگر `try_admit` هم شکست خورد (race نظری، در عمل نباید رخ دهد چون `select_replica` قبلاً چک کرده) → رد با دلیل `queue_full` و لاگ `unexpected_admit_race`.
8. فرمول نهایی زمان پاسخ:
```python
response_time_sec = (
    routing_delay_sec
    + 2 * network_delay_ms / 1000.0        # رفت‌وبرگشت BTS<->سرور
    + wait_time_sec
    + (service_end_time - service_start_time)   # = exec_time + cold_start_extra
)
deadline_violated = response_time_sec > SERVICES_INFO[service_id]["deadline"]
```
9. برای متریک گزارشی `network_delay_ms` فقط مقدار **یک‌طرفه** ثبت می‌شود (بدون ×۲ و بدون `routing_delay_sec`).
10. یک رویداد `ENERGY_RESYNC` بی‌اثر (no-op) در `service_end_time` زمان‌بندی می‌شود تا موتور مطمئن شود انرژی تا آن لحظه integrate می‌شود (چون `_advance_energy_to` هر بار قبل از پردازش رویداد بعدی صدا زده می‌شود).

---

## بخش ۸: مسیریابی/انتخاب نمونه (Instance Selection) — مشترک بین همه‌ی الگوریتم‌ها

پیاده‌سازی پیش‌فرض در `AlgorithmBase.select_replica(request, candidate_replicas, servers, now, admit_fn=None, occupancy_fn=None)`:

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

نکات حیاتی:
- **استخر «تقریباً هم‌فاصله»**: همه‌ی رپلیکاهایی که حداکثر ۵ کیلومتر بیشتر از نزدیک‌ترین فاصله دارند (`min_dist + 5.0`). در این استخر، معیار انتخاب **کمترین نسبت اشغال صف** است.
- اگر در استخر نزدیک هیچ گزینه‌ای admit نشد، fallback به ترتیب فاصله (نه فقط استخر نزدیک) روی بقیه‌ی کاندیدها ادامه می‌یابد.
- `occupancy_fn` (فقط-خواندنی) برای رتبه‌بندی همه‌ی کاندیدها استفاده می‌شود؛ `admit_fn` (که ممکن است side-effect واقعی داشته باشد، مثل رزرو Redis) **دقیقاً یک‌بار روی هر کاندیدا، به ترتیب رتبه، تا اولین موفقیت** صدا زده می‌شود — هرگز روی چند کاندیدا هم‌زمان برای مقایسه.
- در حالت شبیه‌سازی، `admit_fn`/`occupancy_fn` پیش‌فرض از خودِ `Replica.queue_occupancy`/`try_admit` منطق استفاده می‌کنند (بدون پارامتر صریح، `select_replica(req, candidates, servers, now)`).
- **VOILA هیچ متد اختصاصی `select_replica` ندارد** — از همین پیاده‌سازی پایه استفاده می‌کند. (یک نسخه‌ی Vivaldi-based در کد به‌صورت غیرفعال/کامنت باقی مانده اما override نمی‌شود — لازم نیست پیاده‌سازی شود، صرفاً کلاس `VivaldiNetwork` می‌تواند به‌عنوان کد مرده وجود داشته باشد یا اصلاً حذف شود.)

**قابلیت اختیاری PPO** (`latency_aware_routing=True`): به‌جای فاصله‌ی خام، رپلیکا با کمترین **تخمین کل تأخیر** انتخاب می‌شود:
```python
occ = occupancy_fn(r)
if occ >= r.queue_len: continue   # مطمئناً رد می‌شود
distance_km = haversine_km(request.bts_lat, request.bts_long, server.lat, server.long)
delay_ms = network_delay_ms(distance_km, BASE_LATENCY_MS, K_MS_PER_KM)
rtt_sec = 2 * delay_ms / 1000.0
est_wait_sec = occ * r.exec_time
est_total_latency = rtt_sec + est_wait_sec + r.exec_time
```
همه‌ی کاندیدها بر اساس `est_total_latency` مرتب می‌شوند؛ به ترتیب `admit_fn` روی هرکدام صدا زده می‌شود تا یکی موفق شود. **مهم:** `est_wait_sec` همیشه از `occupancy_fn` واقعی (نه `replica.available_at`) محاسبه می‌شود، چون `available_at` فقط در مسیر شبیه‌سازی (`try_admit`) به‌روزرسانی می‌شود، نه در مسیر k8s واقعی.

---

## بخش ۹: مقداردهی اولیه‌ی سیستم (t=0)

### ۹.۱ استراتژی پایه (Greedy/VOILA/HPA — پیاده‌سازی مشترک در `AlgorithmBase.initial_placement`)

```python
def initial_placement(self, servers, active_bts):
    # active_bts = لیست (lat, lon) یکتای BTSهایی که در MONITOR_WINDOW_SEC ثانیه‌ی
    # ابتدایی تایم‌لاین حداقل یک درخواست فرستاده‌اند
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

سپس برای هر یک از سرورهای انتخاب‌شده، `_start_server_boot(sid)` صدا زده می‌شود (وارد BOOTING می‌شود؛ در t=0 این معمولاً یعنی سرور فوراً بوت می‌شود ولی هنوز `boot_delay_sec` طول می‌کشد تا ACTIVE شود).

سپس برای هر یک از ۱۵ سرویس، نزدیک‌ترین سرور *از میان انتخاب‌شده‌ها* (`_nearest_capable_server`) که `can_host` آن را قبول کند پیدا و یک رپلیکا آنجا `_place_replica` می‌شود:
```python
def _nearest_capable_server(service_id, candidate_ids):
    cpu = SERVICES_INFO[service_id]["resource_mips"]
    candidates = [servers[sid] for sid in candidate_ids if servers[sid].can_host(service_id, cpu)]
    if not candidates: return None
    centroid_lat = mean(s.lat for s in candidates)
    centroid_lon = mean(s.long for s in candidates)
    candidates.sort(key=lambda s: haversine_km(centroid_lat, centroid_lon, s.lat, s.long))
    return candidates[0].id
```

### ۹.۲ استراتژی PPO (اختیاری، `use_solver_placement=True` پیش‌فرض): solver ILP

با کتابخانه‌ی `pulp` (solver CBC)، هدف چندجزئی:
```
minimize: w_count * (تعداد سرور روشن / کل سرورها)
        + w_energy * (مجموع p_idle سرورهای انتخاب‌شده / مجموع کل p_idle)
        + w_distance * (مجموع وزنی فاصله‌ی هر نقطه‌ی تقاضا تا نزدیک‌ترین سرور پوشش‌دهنده / نرمال‌ساز)
```
قیود: هر نقطه‌ی تقاضای پوشش‌پذیر (`delay <= L0_MS`) دقیقاً به یک سرور فعال منصوب می‌شود؛ منصوب فقط به سرور فعال‌شده مجاز؛ مجموع ظرفیت سرورهای انتخاب‌شده ≥ مجموع `resource_mips` هر ۱۵ سرویس.

پیاده‌سازی دقیق (`algorithms/ppo/optimal_placement.py`):
```python
def aggregate_training_demand(train_events_df):
    counts = train_events_df.groupby(["Lat", "Long"]).size()
    return [(float(lat), float(lon), int(w)) for (lat, lon), w in counts.items()]

def solve_optimal_server_selection(servers, demand_points, l0_ms=None, min_total_capacity=None,
                                    w_count=1.0, w_energy=1.0, w_distance=1.0, time_limit_sec=120.0):
    import pulp
    l0_ms = l0_ms or L0_MS
    min_total_capacity = min_total_capacity or sum(s["resource_mips"] for s in SERVICES_INFO.values())
    server_ids = list(servers.keys())

    coverage = {}; dist_km = {}
    for sid, s in servers.items():
        covered = set()
        for idx, (lat, lon, _w) in enumerate(demand_points):
            d_km = haversine_km(lat, lon, s.lat, s.long)
            delay = BASE_LATENCY_MS + K_MS_PER_KM * d_km
            if delay <= l0_ms:
                covered.add(idx); dist_km[(idx, sid)] = d_km
        coverage[sid] = covered

    coverable = set().union(*coverage.values()) if coverage else set()
    if not coverable: return []

    prob = pulp.LpProblem("optimal_initial_placement", pulp.LpMinimize)
    y = {sid: pulp.LpVariable(f"y_{sid}", cat="Binary") for sid in server_ids}
    x = {(idx, sid): pulp.LpVariable(f"x_{idx}_{sid}", cat="Binary")
         for idx in coverable for sid in server_ids if idx in coverage[sid]}

    for idx in coverable:
        prob += pulp.lpSum(x[(idx, sid)] for sid in server_ids if (idx, sid) in x) == 1
    for (idx, sid) in x:
        prob += x[(idx, sid)] <= y[sid]
    prob += pulp.lpSum(servers[sid].capacity * y[sid] for sid in server_ids) >= min_total_capacity

    n_servers_total = len(server_ids)
    total_p_idle = sum(servers[sid].p_idle for sid in server_ids)
    max_possible_dist = max(dist_km.values()) if dist_km else 1.0
    total_weight = sum(w for _, _, w in demand_points) or 1

    term_count = pulp.lpSum(y[sid] for sid in server_ids) / n_servers_total
    term_energy = pulp.lpSum(servers[sid].p_idle * y[sid] for sid in server_ids) / max(total_p_idle, 1e-9)
    term_distance = pulp.lpSum(demand_points[idx][2] * dist_km[(idx, sid)] * x[(idx, sid)]
                                for (idx, sid) in x) / (max_possible_dist * total_weight)

    prob += w_count*term_count + w_energy*term_energy + w_distance*term_distance
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Not Solved"): return []
    return [sid for sid in server_ids if pulp.value(y[sid]) and pulp.value(y[sid]) > 0.5]
```

اگر solver جواب پیدا نکرد یا `pulp` نصب نبود، fallback به همان `initial_placement` مشترک (بخش ۹.۱). نقاط تقاضا از **کل تایم‌لاین train سه‌روزه** استخراج می‌شوند، نه فقط پنجره‌ی ابتدایی.

`PPOAlgorithm.initial_placement` منطق زیر را دارد:
```python
def initial_placement(self, servers, active_bts):
    if not self.use_solver_placement:
        return super().initial_placement(servers, active_bts)
    if self._solver_selected_servers is None:
        try:
            train_events = load_train()
            demand_points = aggregate_training_demand(train_events)
            selected = solve_optimal_server_selection(servers, demand_points, **self._placement_weights)
            if selected:
                self._solver_selected_servers = selected
            else:
                self._solver_selected_servers = super().initial_placement(servers, active_bts)
        except Exception:
            self._solver_selected_servers = super().initial_placement(servers, active_bts)
    return self._ensure_sufficient_capacity(servers, list(self._solver_selected_servers))
```

`_ensure_sufficient_capacity`: اگر مجموع ظرفیت انتخاب‌شده کافی نبود (< مجموع همه‌ی `resource_mips`)، سرورهای باقی‌مانده به ترتیب نزدیکی به‌مجموعه‌ی فعلی اضافه می‌شوند تا کافی شود.

---

## بخش ۱۰: Provisioning پویا سرور و Migration

### ۱۰.۱ محاسبه‌ی utilization پنجره‌ای (برای تصمیم‌گیری)

هر سرور یک `cumulative_busy_cpu_seconds` دارد که هر بار موتور زمان جلو می‌رود به‌روز می‌شود:
```python
# در _advance_energy_to(t):
util_at_last = server.instantaneous_utilization(last_update_time)
power = server.instantaneous_power_w(last_update_time)
server.cumulative_energy_joule += power * (t - last_update_time)
server.cumulative_busy_cpu_seconds += util_at_last * server.capacity * (t - last_update_time)
```

در هر تیک تصمیم:
```python
avg_util[sid] = (server.cumulative_busy_cpu_seconds - util_at_window_start[sid]) / (capacity * window_elapsed_sec)
```
اگر `window_elapsed <= 1e-9` (اولین تیک)، fallback به `instantaneous_utilization(now)`.
بعد از هر تیک تصمیم، `util_at_window_start[sid] = cumulative_busy_cpu_seconds` و `util_window_start_time = now` ریست می‌شوند.

### ۱۰.۲ sustain-tracking

برای هر سرور دو ردیاب زمانی جدا: `_low_util_since[sid]`, `_high_util_since[sid]` (مقدار اولیه `None`).

```python
def update_sustain_tracking(snapshot, now):
    for sid, s in servers.items():
        if s.state != ACTIVE:
            low_util_since[sid] = None; high_util_since[sid] = None; continue
        util = snapshot["servers"][sid]["utilization"]
        if util < UTIL_SCALE_DOWN_THRESHOLD:
            if low_util_since[sid] is None: low_util_since[sid] = now
        else:
            low_util_since[sid] = None
        if util > UTIL_SCALE_UP_THRESHOLD:
            if high_util_since[sid] is None: high_util_since[sid] = now
        else:
            high_util_since[sid] = None
```

```python
def any_active_server_sustained_overloaded(now):
    return any(since is not None and (now - since) >= SUSTAIN_HIGH_SEC for since in high_util_since.values())

def any_active_server_sustained_underloaded(now):
    n_active = count(state==ACTIVE)
    if n_active <= 1: return False
    return any(since is not None and (now - since) >= SUSTAIN_LOW_SEC for since in low_util_since.values())

def was_turn_off_necessary(server_id, now):
    since = low_util_since.get(server_id)
    return since is not None and (now - since) >= SUSTAIN_LOW_SEC
```

### ۱۰.۳ capacity-starved detection

```python
def any_service_capacity_starved(snapshot):
    for svc_id in ACTIVE_SERVICES:
        if not was_scale_up_necessary(svc_id, snapshot):
            continue
        cpu = SERVICES_INFO[svc_id]["resource_mips"]
        centroid = snapshot["services"][svc_id].get("demand_centroid")
        bts_lat, bts_long = centroid if centroid else (None, None)
        if not any(s.state in (ACTIVE, BOOTING) and s.can_host(svc_id, cpu, bts_lat=bts_lat, bts_long=bts_long)
                   for s in servers.values()):
            return True
    return False
```
نکته: `BOOTING` هم «قابل‌اتکا» حساب می‌شود (چون به‌زودی ACTIVE می‌شود)، نه فقط `ACTIVE`.

### ۱۰.۴ معیار عینی و مستقل ممیزی (`_was_..._necessary`) — جدا از هر آستانه‌ی داخلی هر الگوریتم

```python
def was_scale_up_necessary(svc_id, snapshot):
    sv = snapshot["services"][svc_id]
    occ_ratio = sv["avg_queue_occupancy"]/sv["queue_len"] if sv["queue_len"] else 0.0
    return (occ_ratio > DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD   # 0.85
            or sv["rejection_rate"] > 0.0
            or sv["deadline_violation_rate"] > 0.0)

def was_scale_down_necessary(svc_id, snapshot):
    sv = snapshot["services"][svc_id]
    occ_ratio = sv["avg_queue_occupancy"]/sv["queue_len"] if sv["queue_len"] else 0.0
    return occ_ratio < DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD and sv["n_ready_replicas"] > 1  # < 0.2

def was_turn_on_necessary_audit(snapshot, now):
    overloaded_now = any(snapshot["servers"][sid]["utilization"] > UTIL_SCALE_UP_THRESHOLD
                          for sid, s in servers.items() if s.state == ACTIVE)
    return overloaded_now or any_service_capacity_starved(snapshot)

def was_turn_off_necessary_audit(server_id, snapshot):
    return snapshot["servers"][server_id]["utilization"] < UTIL_SCALE_DOWN_THRESHOLD
```

**این آستانه‌ها عمداً از آستانه‌ی داخلی هر ۴ الگوریتم فاصله دارند** (Greedy=۰.۷، HPA target=۰.۷، VOILA OCC_UP=۰.۶۵) تا ممیزی تاتولوژیک نباشد.

### ۱۰.۵ انتخاب پروفایل سرور خاموش برای روشن‌کردن (heterogeneity-aware)

```python
def pick_profile_for_overload(overloaded_servers, fallback_capacity):
    large_threshold = SERVER_PROFILES["large"]["capacity_mips"]      # 40320
    medium_threshold = SERVER_PROFILES["medium"]["capacity_mips"]    # 33120
    total = sum(s.capacity for s in overloaded_servers) if overloaded_servers else fallback_capacity
    if total >= large_threshold: return "large"
    if total >= medium_threshold: return "medium"
    return "edge_small"

def filter_by_profile_with_fallback(candidates, desired_profile):
    matching = [s for s in candidates if s.profile == desired_profile]
    return matching if matching else candidates
```

**نکته‌ی حیاتی:** به `pick_profile_for_overload` باید فقط لیست سرورهای *واقعاً overload* پاس داده شود، هرگز `overloaded or active` — اگر لیست overload خالی است و به‌جایش کل لیست active پاس داده شود، مجموع ظرفیت کل فلیت تقریباً همیشه از آستانه‌ی large می‌گذرد و تابع همیشه "large" برمی‌گرداند صرف‌نظر از شدت واقعی starvation. وقتی overloaded خالی است، `fallback_capacity` (مثلاً ظرفیت شلوغ‌ترین سرور فعال یا سروری که ارجاع starvation به آن اشاره دارد) باید استفاده شود.

### ۱۰.۶ اعمال provisioning (اشتراک منطق در sim/real)

```python
def apply_provisioning(action, snapshot, now):
    applied = False; skip_reason = None; via_capacity_starved_only = None
    turn_on_necessary = any_active_server_sustained_overloaded(now) or any_service_capacity_starved(snapshot)
    turn_off_opportunity = any_active_server_sustained_underloaded(now)
    bypass = getattr(algorithm, "bypass_sustain_gate", False)

    if action.action == TURN_ON and action.server_id is not None:
        s = servers[action.server_id]
        if s.state != OFF: skip_reason = "not_off"
        elif s.in_cooldown(now, COOLDOWN_SEC): skip_reason = "cooldown"
        elif not turn_on_necessary and not bypass: skip_reason = "overload_not_sustained"
        else:
            necessary_now = was_turn_on_necessary_audit(snapshot, now)
            via_capacity_starved_only = (any_service_capacity_starved(snapshot)
                                          and not any_active_server_sustained_overloaded(now))
            start_server_boot(action.server_id)   # یا در realtime: activate_server
            metrics.record_scale_action("TURN_ON"); applied = True
            metrics.record_decision_correctness("TURN_ON", necessary_now)

    elif action.action == TURN_OFF and action.server_id is not None:
        s = servers[action.server_id]
        n_active = count(state==ACTIVE)
        turn_off_necessary = was_turn_off_necessary(action.server_id, now)
        if s.state != ACTIVE: skip_reason = "not_active"
        elif not turn_off_necessary and not bypass: skip_reason = "low_util_not_sustained"
        elif n_active <= 1: skip_reason = "last_active_server"
        elif s.in_cooldown(now, COOLDOWN_SEC): skip_reason = "cooldown"
        elif (now - s.last_transition_time) < MIN_ACTIVE_DURATION_SEC: skip_reason = "min_active_duration"
        else:
            if start_server_drain(action.server_id):
                metrics.record_scale_action("TURN_OFF"); applied = True
                metrics.record_decision_correctness("TURN_OFF", was_turn_off_necessary_audit(action.server_id, snapshot))
            else:
                skip_reason = "migration_incomplete"

    # ثبت missed/blocked opportunities:
    proposed_turn_on = action.action==TURN_ON and action.server_id is not None
    proposed_turn_off = action.action==TURN_OFF and action.server_id is not None
    if turn_on_necessary and not applied:
        metrics.record_blocked_opportunity("TURN_ON") if proposed_turn_on else metrics.record_missed_opportunity("TURN_ON")
    if turn_off_opportunity and not (action.action==TURN_OFF and applied):
        metrics.record_blocked_opportunity("TURN_OFF") if proposed_turn_off else metrics.record_missed_opportunity("TURN_OFF")

    log("provision_decision", action=action.action.name, server_id=action.server_id, applied=applied,
        skip_reason=skip_reason, necessary_turn_on=turn_on_necessary, turn_off_opportunity=turn_off_opportunity,
        bypassed_sustain_gate=bypass,
        via_capacity_starved_only=(via_capacity_starved_only if applied and action.action==TURN_ON else None))
```

**Cooldown** بعد از هر boot/drain روی همان سرور، صرف‌نظر از هر چیز دیگری، `COOLDOWN_SEC` اعمال می‌شود.

### ۱۰.۷ `bypass_sustain_gate` — پرچم کلاسی

```python
class AlgorithmBase:
    bypass_sustain_gate: bool = True   # پیش‌فرض پایه (کلاس انتزاعی) True است
```
**هر چهار الگوریتم واقعی از این پیش‌فرض `True` ارث‌بری می‌کنند مگر این‌که صریحاً override کنند.** در کد نهایی هیچ‌کدام از Greedy/HPA/VOILA این را override نمی‌کنند (پس همه `True` هستند، یعنی sustain-gate برای هیچ‌کدام اجباری نیست جز از طریق cooldown/min_active_duration که همیشه صرف‌نظر از این پرچم اعمال می‌شوند). `PPOAlgorithm` هم صریحاً `bypass_sustain_gate = True` را ست می‌کند (هرچند این با پیش‌فرض یکی است).

> **توجه پیاده‌سازی:** این نکته‌ی مهم در سند اصلی (README) با کد نهایی متفاوت توضیح داده شده بود (README می‌گفت فقط PPO این پرچم را True می‌کند)، اما در آخرین نسخه‌ی کد `AlgorithmBase.bypass_sustain_gate` خودش `True` است و هیچ الگوریتمی آن را به `False` تغییر نمی‌دهد. برای پیاده‌سازی صحیح و نهایی، **پیش‌فرض کلاس پایه را `True` بگذار** — این یعنی در عمل، sustain-tracking صرفاً به‌عنوان سیگنال محاسباتی/گزارشی (`turn_on_necessary`, `turn_off_opportunity`) نگه داشته می‌شود اما هیچ الگوریتمی توسط آن مسدود نمی‌شود؛ تنها گیت‌های واقعاً مسدودکننده، cooldown و `min_active_duration_sec` هستند. ممیزی `decision_correctness` مستقل از این پرچم، همیشه با معیار بخش ۱۰.۴ سنجیده می‌شود.

### ۱۰.۸ Service Migration هنگام DRAIN

```python
def start_server_drain(server_id):
    s = servers[server_id]
    if s.state != ACTIVE: return False
    steps = algorithm.migration_decision(s, servers)   # لیست MigrationStep(service_id, target_server_id)

    # pre-validation ظرفیت مقصد: چند step هم‌زمان نباید مجموعاً از ظرفیت یک مقصد مشترک فراتر روند
    reserved_cpu = defaultdict(int); valid_steps = []
    for step in steps:
        target = servers[step.target_server_id]
        cpu = SERVICES_INFO[step.service_id]["resource_mips"]
        if target.free_capacity() - reserved_cpu[target.id] >= cpu:
            reserved_cpu[target.id] += cpu
            valid_steps.append(step)
        else:
            log("migration_step_dropped", ..., reason="target_capacity_overcommitted")
    steps = valid_steps

    migrated_services = {step.service_id for step in steps}
    # سرویس‌هایی که "sole hosted" هستند: تنها رپلیکای این سرویس در کل سیستم روی همین سرور در حال drain است
    sole_hosted = {svc_id for svc_id, r in s.hosted_replicas.items()
                   if r.state != TERMINATED and not any(
                       other.id != server_id and svc_id in other.hosted_replicas and
                       other.hosted_replicas[svc_id].state != TERMINATED for other in servers.values())}
    unmigrated = sole_hosted - migrated_services
    if unmigrated:
        trigger_emergency_boot(unmigrated, s)
        log("server_drain_aborted", reason="migration_incomplete", unmigrated=list(unmigrated))
        return False

    s.state = DRAINING; s.drain_started_at = now; s.last_transition_time = now
    log("server_drain_started")

    for step in steps:
        log("migration_started", ...)
        placed = place_replica(step.target_server_id, step.service_id)   # ساخت رپلیکای جدید (STARTING)
        if placed is None:
            log("migration_placement_failed", ...); continue
        pending_migrations[(step.target_server_id, step.service_id)] = server_id

    for r in list(s.hosted_replicas.values()):
        if r.service_id in migrated_services: continue   # این‌ها منتظر READY شدن مقصد می‌مانند (zero-downtime)
        start_replica_drain(r)   # رپلیکاهای چندنسخه‌ای فوراً drain می‌شوند

    schedule(now + SERVER_DRAIN_GRACE_SEC, SERVER_DRAIN_DONE, server_id)
    return True
```

**zero-downtime migration**: رپلیکای قدیمِ سرویس‌های sole-hosted فقط **بعد از READY شدن** رپلیکای جدید drain می‌شود (نگاه کن `_handle_replica_ready`: وقتی رپلیکای مقصد READY می‌شود، `pending_migrations` چک می‌شود و اگر مچ داشت، رپلیکای مبدأ آنجا drain می‌شود).

**رفع اضطراری (`trigger_emergency_boot`)**: اگر migration یک سرویس sole-hosted جواب نداد، نزدیک‌ترین سرور خاموش با ظرفیت کافی برای آن سرویس (با در نظر گرفتن رزرو موازی چند سرویس روی همان مقصد) بوت می‌شود؛ درین بی همان تیک متوقف/aborted می‌شود و بعداً دوباره تلاش خواهد شد (چون سرور همچنان ACTIVE می‌ماند و در تیک بعدی provision_decision دوباره TURN_OFF پیشنهاد می‌شود اگر هنوز لازم باشد).

`_handle_drain_done` (بعد از `SERVER_DRAIN_GRACE_SEC`): اگر همه‌ی رپلیکاهای روی سرور `TERMINATED` نشده باشند، رویداد دوباره به تعویق می‌افتد (خودش را re-schedule می‌کند)؛ در غیر این صورت سرور واقعاً به `OFF` می‌رود و `num_shutdowns` افزایش می‌یابد.

---

## بخش ۱۱: Auto Scaling رپلیکا

رابط مشترک:
```python
def scale_decision(self, service_id, metrics_snapshot) -> ScaleAction: ...   # {SCALE_UP, SCALE_DOWN, NO_CHANGE}
```
محافظ SCALE_DOWN: فقط رپلیکاهای «بالغ» (`(now - created_at) >= MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC`) کاندید حذف هستند.

### الگوریتم‌های اسکیل (منطق دقیق هر کدام)

**Greedy** (`scale_decision`):
```python
occ_ratio = avg_queue_occupancy / queue_len if queue_len else 0.0
if occ_ratio > 0.7 or rejection_rate > 0: return SCALE_UP
if occ_ratio < 0.1 and n_ready_replicas > 1: return SCALE_DOWN
return NO_CHANGE
```

**HPA** (`scale_decision`, `TARGET_UTILIZATION=0.70`):
```python
current_replicas = max(n_ready_replicas, 1)
current_util = avg_queue_occupancy / queue_len if queue_len else 0.0
if current_util <= 0 and rejection_rate <= 0:
    desired = 1
else:
    desired = ceil(current_replicas * (current_util / 0.70))
desired = max(1, desired)
if rejection_rate > 0:
    desired = max(desired, current_replicas + 1)
if desired > current_replicas: return SCALE_UP
if desired < current_replicas and current_replicas > 1: return SCALE_DOWN
return NO_CHANGE
```

**VOILA** (`scale_decision`, `OCC_UP_THRESHOLD=0.65`, `OCC_DOWN_THRESHOLD=0.20`, `SCALE_DOWN_PATIENCE_TICKS=3`, `PROXIMITY_SUSTAIN_TICKS=2`, `PROXIMITY_PROTECTION_TICKS=5`):
```python
occ_ratio = avg_queue_occupancy / queue_len if queue_len else 0.0
capacity_violation = occ_ratio > 0.65 or rejection_rate > 0.0
proximity_violation = (not capacity_violation) and proximity_violation_rate > 0.0

if capacity_violation:
    good_streak[svc]=0; proximity_violation_streak[svc]=0
    return SCALE_UP

if proximity_violation:
    proximity_violation_streak[svc] += 1
    good_streak[svc] = 0
    if proximity_violation_streak[svc] < PROXIMITY_SUSTAIN_TICKS:
        return NO_CHANGE
    proximity_violation_streak[svc] = 0
    proximity_recent[svc] = PROXIMITY_PROTECTION_TICKS
    return SCALE_UP

proximity_violation_streak[svc] = 0
good_streak[svc] += 1
proximity_recent[svc] = max(0, proximity_recent.get(svc,0) - 1)
if proximity_recent.get(svc,0) > 0:
    return NO_CHANGE
if good_streak[svc] >= 3 and occ_ratio < 0.20 and n_ready_replicas > 1:
    good_streak[svc] = 0
    return SCALE_DOWN
return NO_CHANGE
```

**PPO**: تصمیم از مدل آموزش‌دیده می‌آید (بخش ۱۳)، کش‌شده در `_predict_and_cache`.

### Placement رپلیکای جدید (هنگام SCALE_UP)

**Greedy**:
```python
centroid = demand_centroid(svc) or (mean_lat_of_active_servers, mean_long_of_active_servers)
candidates = [s for s in active_servers if s.can_host(svc, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
candidates.sort(key=lambda s: haversine_km(centroid[0], centroid[1], s.lat, s.long))
return candidates[0].id if candidates else None
```

**VOILA/PPO** (تفاوت با Greedy: بین نزدیک‌ترین‌ها به centroid، بیشترین ظرفیت آزاد انتخاب می‌شود):
```python
centroid = demand_centroid(svc) or (mean_lat_active, mean_long_active)
candidates = [s for s in active_servers if s.can_host(svc, cpu, bts_lat=centroid[0], bts_long=centroid[1])]
if not candidates: return None
distances = {s.id: haversine_km(centroid[0], centroid[1], s.lat, s.long) for s in candidates}
min_dist = min(distances.values())
near_pool = [s for s in candidates if distances[s.id] <= min_dist + 5.0]
return max(near_pool, key=lambda s: s.free_capacity()).id
```

**HPA** (عمداً location-unaware — بدون centroid):
```python
candidates = [s for s in active_servers if s.can_host(svc, cpu)]   # بدون bts_lat/long
return max(candidates, key=lambda s: s.free_capacity()).id if candidates else None
```

### `demand_centroid` — medoid موقعیت‌های اخیر تقاضا

هر سرویس یک بافر `deque(maxlen=30)` از آخرین موقعیت‌های (lat, long) درخواست‌های ورودی خودش نگه می‌دارد. در هر تیک تصمیم (اگر آن تیک حداقل یک ورود داشته)، مرکز medoid این نقاط محاسبه می‌شود:
```python
def medoid(points):
    if not points: return None
    best, best_cost = points[0], float("inf")
    for p in points:
        cost = sum(haversine_km(p[0], p[1], q[0], q[1]) for q in points)
        if cost < best_cost:
            best, best_cost = p, cost
    return best   # نقطه‌ی واقعی از میان نقاط، نه میانگین حسابی
```

---

## بخش ۱۲: متریک‌ها و ممیزی تصمیم

`MetricsCollector` باید موارد زیر را جمع‌آوری کند: `response_times[]`, `network_delays[]`, `distances[]`, `deadline_violations`, `total_requests`, `completed_requests`, `rejected_queue_full`, `rejected_no_replica`, `num_server_boots/shutdowns`, `num_pod_creates/deletes`, `num_scale_up/down`, `num_turn_on/off`, و `_decision_correctness: dict[kind -> {correct, incorrect, missed, blocked}]`.

```python
def record_decision_correctness(kind, necessary):
    _decision_correctness[kind]["correct" if necessary else "incorrect"] += 1

def record_missed_opportunity(kind):
    _decision_correctness[kind]["missed"] += 1   # الگوریتم اصلاً پیشنهاد نداد

def record_blocked_opportunity(kind):
    _decision_correctness[kind]["blocked"] += 1  # پیشنهاد داد، ولی گیت سیستمی جلویش را گرفت
```

`record_request(req)`:
```python
total_requests += 1
if status == COMPLETED:
    completed_requests += 1; response_times.append(response_time_sec)
    distances.append(getattr(req, "_distance_km", 0.0)); network_delays.append(network_delay_ms)
    if deadline_violated: deadline_violations += 1
elif status == REJECTED_QUEUE_FULL:
    rejected_queue_full += 1; deadline_violations += 1
elif status == REJECTED_NO_REPLICA:
    rejected_no_replica += 1; deadline_violations += 1
```

`record_snapshot(now, servers)`: هر تیک، لیست فعال‌ها را ذخیره؛ اگر `len(active)==1` → `load_balance_cv=0.0`؛ اگر `>=2` → `std(utilizations)/mean(utilizations)` (اگر mean>0 وگرنه ۰).

`finalize(servers)`: خروجی JSON نهایی شامل تمام میانگین‌ها/percentileها (۹۵ام و ۹۹ام با `numpy.percentile`) + `decision_correctness` با `correctness_rate_pct = 100*correct/(correct+incorrect)` (یا `None` اگر مخرج صفر).

فیلدهای دقیق خروجی نهایی:
```
algorithm, avg_response_time_sec, p95_response_time_sec, p99_response_time_sec,
deadline_violations, deadline_violation_rate_pct, cumulative_energy_joule,
avg_distance_km, avg_load_balance_cv, avg_network_delay_ms, p95_network_delay_ms, p99_network_delay_ms,
num_server_boots, num_server_shutdowns, num_pod_creates, num_pod_deletes,
num_requests_rejected_queue_full, num_requests_rejected_no_replica, avg_active_servers,
num_scale_up, num_scale_down, num_turn_on, num_turn_off, decision_correctness,
total_requests, completed_requests
```
`cumulative_energy_joule = sum(s.cumulative_energy_joule for s in servers.values())`.

---

## بخش ۱۳: موتور شبیه‌سازی (`SimulationEngine`) — حلقه‌ی رویداد کامل

### ۱۳.۱ رویدادها (heap با `heapq`, ترتیب بر اساس `(time, seq)`)
```python
class EventType(Enum):
    REQUEST_ARRIVAL, REQUEST_ROUTED, DECISION_TICK, SERVER_BOOT_DONE, SERVER_DRAIN_DONE,
    REPLICA_READY, REPLICA_TERMINATED, ENERGY_RESYNC
```
`Event(time, seq, type, payload)` — `order=True` روی `(time, seq)`، `seq` شمارنده‌ی صعودی برای شکستن تساوی و پایداری ترتیب.

### ۱۳.۲ `prime()`
```python
def prime():
    for row in events_df: push(row.global_start_sec, REQUEST_ARRIVAL, row)
    now = start_time = events_df.global_start_sec.min()
    energy_last_update = {sid: start_time for sid in servers}
    util_window_start_time = start_time
    initial_placement()
    push(start_time, DECISION_TICK)
    cutoff = max_time + 2*DISPATCH_OVERHEAD_MS/1000.0 + DECISION_INTERVAL_SEC + SERVER_DRAIN_GRACE_SEC
```

### ۱۳.۳ `step(external_actions=None)`
```python
def step(external_actions=None):
    while heap:
        ev = pop()
        if ev.time > cutoff: return None, True
        advance_energy_to(ev.time)   # برای همه‌ی سرورها: energy/busy_seconds integrate
        now = ev.time
        dispatch on ev.type:
            REQUEST_ARRIVAL   -> handle_arrival(payload)
            REQUEST_ROUTED     -> handle_routed(payload)
            DECISION_TICK      -> snapshot = handle_decision_tick(external_actions)
                                   push(now + DECISION_INTERVAL_SEC, DECISION_TICK)
                                   return snapshot, False    # کنترل به caller برمی‌گردد
            SERVER_BOOT_DONE   -> handle_boot_done(payload)
            SERVER_DRAIN_DONE  -> handle_drain_done(payload)
            REPLICA_READY      -> handle_replica_ready(payload)
            REPLICA_TERMINATED -> handle_replica_terminated(payload)
            ENERGY_RESYNC      -> pass (no-op, فقط باعث advance_energy_to می‌شود)
    return None, True
```
`run()` = `prime()` + حلقه‌ی `step()` تا `done`، سپس `metrics.finalize(servers)`.

### ۱۳.۴ `_handle_decision_tick(external_actions)`
```python
def handle_decision_tick(external_actions):
    metrics.record_snapshot(now, servers)
    snapshot = build_metrics_snapshot()          # پنجره‌ای دقیق (بخش ۱۰.۱)
    update_sustain_tracking(snapshot)
    annotate_provisioning_necessity(snapshot)    # snapshot["global"]["turn_on_necessary"] و هر سرور "turn_off_necessary"

    if external_actions is not None:             # مسیر PPO env (آموزش)
        apply_provisioning(external_actions["provision"], snapshot)
        for svc_id, decision in external_actions["scale"].items():
            apply_scale_decision(svc_id, decision, snapshot)
    else:
        action = algorithm.provision_decision(servers, snapshot, now)
        apply_provisioning(action, snapshot)
        for svc_id in ACTIVE_SERVICES:
            decision = algorithm.scale_decision(svc_id, snapshot)
            apply_scale_decision(svc_id, decision, snapshot)

    for sid, s in servers.items(): util_at_window_start[sid] = s.cumulative_busy_cpu_seconds
    util_window_start_time = now
    clear all tick_* counters (tick_total, tick_rejected, tick_violated, tick_response_times,
                                tick_lat_sum, tick_lon_sum, tick_proximity_violated)
    return snapshot
```

### ۱۳.۵ `apply_scale_decision(svc_id, decision, snapshot)`
```python
def apply_scale_decision(svc_id, decision, snapshot):
    applied = False; skip_reason = None
    necessary_up = was_scale_up_necessary(svc_id, snapshot)
    necessary_down = was_scale_down_necessary(svc_id, snapshot)

    if decision == NO_CHANGE: pass
    elif (now - service_last_scale_time[svc_id]) < COOLDOWN_SEC: skip_reason = "cooldown"
    elif decision == SCALE_UP:
        target = algorithm.select_placement_server(svc_id, servers)
        if target is not None:
            placed = place_replica(target, svc_id)
            if placed is not None:
                metrics.record_scale_action("SCALE_UP")
                service_last_scale_time[svc_id] = now
                applied = True
                metrics.record_decision_correctness("SCALE_UP", necessary_up)
            else: skip_reason = "placement_failed"
        else: skip_reason = "no_target_server"
    elif decision == SCALE_DOWN:
        ready = [r for r in replicas_by_service[svc_id] if r.state == READY]
        if len(ready) > 1:
            mature = [r for r in ready if (now - r.created_at) >= MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC]
            if not mature: skip_reason = "no_mature_replica"
            else:
                victim = algorithm.select_scale_down_victim(svc_id, mature, servers, now)
                start_replica_drain(victim)
                metrics.record_scale_action("SCALE_DOWN")
                service_last_scale_time[svc_id] = now
                applied = True
                metrics.record_decision_correctness("SCALE_DOWN", necessary_down)
        else: skip_reason = "only_one_replica_left"

    # missed/blocked opportunity (همان الگوی provisioning، بخش ۱۰.۶)
    if necessary_up and not (decision==SCALE_UP and applied):
        record_blocked("SCALE_UP") if decision==SCALE_UP else record_missed("SCALE_UP")
    if necessary_down and not (decision==SCALE_DOWN and applied):
        record_blocked("SCALE_DOWN") if decision==SCALE_DOWN else record_missed("SCALE_DOWN")

    log("scale_decision", service_id=svc_id, decision=decision.name, applied=applied,
        skip_reason=skip_reason, necessary_scale_up=necessary_up, necessary_scale_down=necessary_down)
```

`select_scale_down_victim` پیش‌فرض (`AlgorithmBase`): `min(ready_replicas, key=occupancy_fn)` (کمترین اشغال صف). **VOILA** override می‌کند: اگر centroid موجود و `len(ready_replicas)>1`، نیمی از رپلیکاها با کمترین بار انتخاب و از بین آن‌ها **دورترین از centroid** به‌عنوان قربانی انتخاب می‌شود (منطق: رپلیکای دور و کم‌بار، نه لزوماً کم‌بارترین مطلق).

### ۱۳.۶ توابع کمکی موتور (`_place_replica`, `_start_server_boot`, ...)

```python
def place_replica(server_id, service_id):
    s = servers[server_id]
    cpu = SERVICES_INFO[service_id]["resource_mips"]
    centroid = service_demand_centroid.get(service_id)
    bts_lat, bts_long = centroid if centroid else (None, None)
    if not s.can_host(service_id, cpu, bts_lat=bts_lat, bts_long=bts_long):
        return None
    r = Replica(service_id, server_id, queue_len=SERVICES_INFO[service_id]["queue_len"],
                exec_time=compute_exec_time_sec(service_id, SERVER_PROFILES[s.profile]["mips_per_core"]),
                created_at=now)
    s.hosted_replicas[service_id] = r
    replicas_by_service[service_id].append(r)
    metrics.record_transition("pod_create")
    s.cumulative_energy_joule += E_POD_CREATE_J
    log("pod_create_started", ...)
    if s.state == ACTIVE:
        schedule_replica_ready(r)   # push(now + POD_STARTUP_DELAY_SEC, REPLICA_READY, (server_id, service_id))
    return r

def start_server_boot(server_id):
    s = servers[server_id]
    if s.state != OFF: return
    s.state = BOOTING; s.boot_started_at = now; s.last_transition_time = now
    s.num_boots += 1
    s.cumulative_energy_joule += E_BOOT_SERVER_J
    metrics.record_transition("server_boot")
    log("server_boot_started", ...)
    push(now + BOOT_DELAY_SEC, SERVER_BOOT_DONE, server_id)

def handle_boot_done(server_id):
    s = servers[server_id]; s.state = ACTIVE; s.last_transition_time = now
    log("server_active", ...)
    for r in s.hosted_replicas.values():
        if r.state==STARTING and r.ready_since is None and r.created_at <= now:
            schedule_replica_ready(r)
    # rescue emergency boot bookkeeping (بخش ۱۰.۸)

def handle_replica_ready(key):
    server_id, service_id = key
    r = servers[server_id].hosted_replicas.get(service_id)
    if r is None or r.state != STARTING: return
    r.state = READY; r.ready_since = now
    log("pod_ready", ...)
    # اگر این رپلیکا مقصد یک migration بود، رپلیکای مبدأ حالا drain می‌شود (zero-downtime)
    src = pending_migrations.pop((server_id, service_id), None)
    if src is not None:
        old = servers[src].hosted_replicas.get(service_id)
        if old is not None and old.state not in (TERMINATED, DRAINING):
            start_replica_drain(old)
        log("migration_completed", ...)

def start_replica_drain(r):
    if r.state == TERMINATED: return
    r.state = DRAINING; r.drain_started_at = now
    log("pod_drain_started", ...)
    drain_wait = max(GRACEFUL_TERMINATION_DELAY_SEC, r.available_at - now)
    push(now + drain_wait, REPLICA_TERMINATED, (r.server_id, r.service_id))

def handle_replica_terminated(key):
    server_id, service_id = key
    s = servers[server_id]; r = s.hosted_replicas.get(service_id)
    if r is None: return
    r.state = TERMINATED
    metrics.record_transition("pod_delete")
    log("pod_terminated", ...)
    del s.hosted_replicas[service_id]
    replicas_by_service[service_id] = [x for x in replicas_by_service[service_id] if x is not r]
```

### ۱۳.۷ `handle_arrival(row)` و `handle_routed(req)`

دقیقاً طبق بخش ۷ بالا؛ نکات اضافه:
- `req.id` از یک شمارنده‌ی سراسری `request_seq` می‌آید (شروع از ۱).
- `_recent_positions[service_id].append((lat, long))` هنگام ورود انجام می‌شود (نه هنگام routed) — این تغذیه‌ی مستقیم `demand_centroid`.
- `_tick_total/_tick_lat_sum/_tick_lon_sum[service_id]` هنگام ورود افزایش/جمع می‌شوند.

---

## بخش ۱۴: عامل PPO-DRL

### ۱۴.۱ محدوده‌ی تصمیم

هر `DECISION_INTERVAL_SEC=30` ثانیه، یک اکشن ترکیبی: اسکیل هر ۱۵ سرویس + provisioning یک سرور. Instance selection لحظه‌ای خارج از این محدوده است (مشترک، بخش ۸).

### ۱۴.۲ فضای حالت (`common/state_builder.py`)

```
STATE_DIM = n_servers*6 + n_services*6 + 2 = 10*6 + 15*6 + 2 = 152
```
ترتیب بردار:
```python
for sid in sorted(server_info.keys()):   # ۱۰ سرور، هرکدام ۶ عدد
    one_hot = [1.0 if state==OFF else 0.0, ..BOOTING.., ..ACTIVE.., ..DRAINING..]  # ترتیب: OFF,BOOTING,ACTIVE,DRAINING
    n_replicas = len(servers[sid].hosted_replicas)
    parts += one_hot
    parts.append(utilization)
    parts.append(n_replicas / n_services)

for svc_id in active_services:   # ۱۵ سرویس، هرکدام ۶ عدد
    occ_ratio = avg_queue_occupancy/queue_len if queue_len else 0.0
    parts.append(n_replicas / n_servers)
    parts.append(min(occ_ratio, 2.0) / 2.0)
    parts.append(deadline_violation_rate)
    parts.append(min(recent_arrivals / NORM_ARRIVAL_RATE, 2.0) / 2.0)
    parts.append(rejection_rate)
    parts.append(proximity_violation_rate)

parts.append(min(avg_response_time_recent / NORM_RESPONSE_TIME_SEC, 2.0) / 2.0)
parts.append(min(energy_recent_joule / NORM_ENERGY_JOULE, 2.0) / 2.0)

vec = np.array(parts, dtype=float32)   # طول باید دقیقاً 152 باشد؛ assert کن
```
مقادیر نرمال‌سازی: `NORM_RESPONSE_TIME_SEC=1.232, NORM_ENERGY_JOULE=4431.91, NORM_ARRIVAL_RATE=3.0` — این‌ها با اجرای کامل Greedy روی داده‌ی train و برداشتن p90/p95 آماری از هر کمیت به‌دست آمده‌اند (اسکریپت کالیبراسیون در بخش ۱۸ توضیح داده می‌شود).

### ۱۴.۳ فضای اکشن

```python
action_space = MultiDiscrete([3]*15 + [3]*10)
# ۱۵ بعد اول: {0:NO_CHANGE, 1:SCALE_UP, 2:SCALE_DOWN} به ازای هر سرویس (به ترتیب sorted service ids)
# ۱۰ بعد بعدی: {0:NO_CHANGE, 1:TURN_ON, 2:TURN_OFF} به ازای هر سرور (به ترتیب sorted server ids)
```
برای بخش سرور: از میان تمام غیر-NO_CHANGE، فقط **یکی** واقعاً به موتور پاس داده می‌شود:
```python
turn_ons = sorted(sid for sid,ptype in server_actions.items() if ptype==TURN_ON)
turn_offs = sorted(sid for sid,ptype in server_actions.items() if ptype==TURN_OFF)
chosen_list = turn_ons or turn_offs   # اولویت با TURN_ON اگر هر دو موجود باشند
if chosen_list:
    chosen_sid = chosen_list[0]
    provision_action = ProvisionAction(server_actions[chosen_sid], chosen_sid)
else:
    provision_action = ProvisionAction(NO_CHANGE)
```

### ۱۴.۴ Action Masking (`compute_action_masks`) — باید دقیقاً یکی بین train و inference باشد

```python
def compute_action_masks(engine_or_state, last_snapshot):
    masks = []
    for sid in sorted_service_ids:
        cooldown = (now - service_last_scale_time[sid]) < COOLDOWN_SEC
        cpu = SERVICES_INFO[sid]["resource_mips"]
        centroid = last_snapshot["services"][sid].get("demand_centroid") if last_snapshot else None
        bts_lat, bts_long = centroid if centroid else (None, None)
        can_up = (not cooldown) and any(
            s.state==ACTIVE and s.can_host(sid, cpu, bts_lat=bts_lat, bts_long=bts_long)
            for s in servers.values())
        ready = [r for r in replicas_by_service.get(sid, []) if r.state==READY]
        mature = [r for r in ready if (now - r.created_at) >= MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC]
        can_down = (not cooldown) and len(ready) > 1 and len(mature) > 0
        masks.extend([True, can_up, can_down])

    n_active = count(state==ACTIVE)
    for sid in sorted_server_ids:
        s = servers[sid]
        cooldown = s.in_cooldown(now, COOLDOWN_SEC)
        can_on = (s.state==OFF) and (not cooldown)
        can_off = (s.state==ACTIVE and not cooldown and n_active > 1
                   and (now - s.last_transition_time) >= MIN_ACTIVE_DURATION_SEC)
        masks.extend([True, can_on, can_off])
    return np.array(masks, dtype=bool)
```
**این mask فقط امکان‌پذیری فیزیکی را چک می‌کند** — هیچ ربطی به sustain-tracking/necessity ندارد (چون `bypass_sustain_gate=True` است). این منطق باید **دقیقاً یکسان** در سه جا پیاده شود:
1. `algorithms/ppo/env.py:compute_action_masks` (مسیر training، به‌طور مستقیم به `engine` دسترسی دارد)
2. `PPOAlgorithm._build_action_masks` (مسیر inference/k8s، فقط `(servers, snapshot, now)` دارد — همان فرمول را با این پارامترها بازتولید می‌کند)
3. هر جای دیگری که مدل قرار است تصمیم بگیرد

### ۱۴.۵ Reward

```python
w = PPO_REWARD_WEIGHTS
active_svcs = [s for s in snapshot["services"].values() if s["recent_arrivals"] > 0]
if active_svcs:
    total_arrivals = sum(s["recent_arrivals"] for s in active_svcs)
    weighted_dv_rate = sum(s["deadline_violation_rate"]*s["recent_arrivals"] for s in active_svcs) / total_arrivals
    unweighted_dv_rate = sum(s["deadline_violation_rate"] for s in active_svcs) / len(active_svcs)
    alpha = PPO_DEADLINE_FAIRNESS_ALPHA   # 0.7
    avg_dv_rate = alpha*weighted_dv_rate + (1-alpha)*unweighted_dv_rate
else:
    weighted_dv_rate = unweighted_dv_rate = avg_dv_rate = 0.0

active_utils = [s["utilization"] for s in snapshot["servers"].values() if s["state"]==ACTIVE]
load_cv = 0.0
if len(active_utils) >= 2 and mean(active_utils) > 0:
    load_cv = std(active_utils) / mean(active_utils)

norm_rt = min(g["avg_response_time_recent"] / NORM_RESPONSE_TIME_SEC, 2.0)
norm_energy = min(g["energy_recent_joule"] / NORM_ENERGY_JOULE, 2.0)
norm_lb = min(load_cv, 2.0)
norm_rejected = min(g["num_rejected_recent"] / NORM_REJECTED_PER_TICK, 2.0)   # NORM_REJECTED_PER_TICK پیش‌فرض 2.0

penalty = (w["w1_response_time"]*norm_rt + w["w2_deadline"]*avg_dv_rate + w["w3_energy"]*norm_energy
           + w["w4_load_balance"]*norm_lb + w["w5_rejected"]*norm_rejected)
penalty += PPO_PENALTY_PER_ACTION * n_actions_applied_this_tick   # 0.02 * تعداد اکشن‌های واقعاً applied این تیک
reward = -penalty
```
`n_actions_applied_this_tick` از دلتای شمارنده‌های `metrics.num_scale_up/down/turn_on/off` بین قبل و بعد از `engine.step()` محاسبه می‌شود.

هر جزء (`response_time, deadline, energy, load_balance, rejected, action_penalty, deadline_weighted_raw, deadline_unweighted_raw`) باید جدا ذخیره شود (برای لاگ TensorBoard در آموزش).

### ۱۴.۶ محیط Gymnasium (`EdgeResourceEnv`)

```python
action_space = MultiDiscrete([3]*15 + [3]*10)
observation_space = Box(low=0.0, high=1.0, shape=(152,), dtype=float32)

def reset(seed=None, options=None):
    events_df = events_df_provider()   # تولیدکننده‌ی پنجره‌ی تصادفی از داده‌ی train
    engine = SimulationEngine(events_df, shared_algo, "ppo_train")
    if hasattr(shared_algo, "bind_engine"): shared_algo.bind_engine(engine)
    engine.prime()
    snapshot = engine.peek_snapshot()
    obs = build_state_vector(snapshot, engine.servers)
    return obs, {}

def step(action):
    service_actions = {sid: SCALE_MAP[action[i]] for i, sid in enumerate(sorted_service_ids)}
    server_actions = {sid: PROVISION_MAP[action[15+j]] for j, sid in enumerate(sorted_server_ids)}
    provision_action = combine_to_single(server_actions)   # بخش ۱۴.۳
    external = {"provision": provision_action, "scale": service_actions}

    before = (num_scale_up, num_scale_down, num_turn_on, num_turn_off)
    snapshot, done = engine.step(external_actions=external)
    if done:
        return zeros(152), 0.0, True, False, {}
    after = (...)
    n_actions_applied = sum(a2-a1 for a1,a2 in zip(before, after))
    obs = build_state_vector(snapshot, engine.servers)
    reward = compute_reward(snapshot, n_actions_applied)
    return obs, reward, False, False, {}

def action_masks():
    return compute_action_masks(engine, last_snapshot)
```

**نکته‌ی حیاتی هماهنگی train/inference در placement:** حین آموزش، `shared_algo` (یک `_MinimalSharedAlgorithm` که فقط برای گرفتن `select_placement_server`/`migration_decision` استفاده می‌شود، نه برای تصمیم اسکیل/provision که از خارج (اکشن مدل) می‌آید) باید همان منطق centroid-aware را برای placement استفاده کند — دقیقاً همان چیزی که `PPOAlgorithm.select_placement_server` (مسیر inference واقعی) استفاده می‌کند:
```python
def select_placement_server(service_id, servers):
    centroid = engine._service_demand_centroid.get(service_id)   # مستقیم از خودِ engine جاری
    bts_lat, bts_long = centroid if centroid else (None, None)
    candidates = [s for s in servers.values() if s.state==ACTIVE
                  and s.can_host(service_id, cpu, bts_lat=bts_lat, bts_long=bts_long)]
    if not candidates: return None
    if centroid is not None:
        distances = {s.id: haversine_km(centroid[0],centroid[1],s.lat,s.long) for s in candidates}
        min_dist = min(distances.values())
        near_pool = [s for s in candidates if distances[s.id] <= min_dist+5.0]
        return max(near_pool, key=lambda s: s.free_capacity()).id
    return max(candidates, key=lambda s: s.free_capacity()).id
```
`migration_decision` در این helper شیء را به یک `GreedyAlgorithm` تفویض می‌کند اما `_last_snapshot` آن دستی با centroidهای واقعی `engine._service_demand_centroid` پر می‌شود.

### ۱۴.۷ کلاس `PPOAlgorithm` (مسیر inference)

```python
class PPOAlgorithm(AlgorithmBase):
    name = "ppo"
    bypass_sustain_gate = True

    def __init__(self, model_path, deterministic=True, latency_aware_routing=False,
                 use_solver_placement=True, placement_weights=None):
        self.model = MaskablePPO.load(model_path)
        self.deterministic = deterministic
        self._cached_tick_key = None
        self._cached_scale = {}
        self._cached_provision = ProvisionAction(NO_CHANGE)
        self._last_snapshot = None
        self._helper = GreedyAlgorithm()   # برای migration_decision
        self.latency_aware_routing = latency_aware_routing
        self.use_solver_placement = use_solver_placement
        self._solver_selected_servers = None
        self._placement_weights = placement_weights or {"w_count":1.0,"w_energy":1.0,"w_distance":1.0}

    def _predict_and_cache(self, servers, metrics_snapshot, now):
        self._last_snapshot = metrics_snapshot
        if self._cached_tick_key == now:
            return   # کش: چون scale_decision ممکن است چند بار در همان تیک صدا زده شود
        obs = build_state_vector(metrics_snapshot, servers)
        action_masks = self._build_action_masks(servers, metrics_snapshot, now)
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=self.deterministic)
        self._cached_scale = {sid: SCALE_MAP[action[i]] for i, sid in enumerate(sorted_service_ids)}
        non_noop = [(sid, PROVISION_MAP[action[15+j]]) for j, sid in enumerate(sorted_server_ids)
                    if PROVISION_MAP[action[15+j]] != NO_CHANGE]
        provision = ProvisionAction(NO_CHANGE)
        if non_noop:
            turn_ons = sorted((sid,pt) for sid,pt in non_noop if pt==TURN_ON)
            turn_offs = sorted((sid,pt) for sid,pt in non_noop if pt==TURN_OFF)
            chosen_sid, chosen_ptype = (turn_ons or turn_offs)[0]
            provision = ProvisionAction(chosen_ptype, chosen_sid)
        self._cached_provision = provision
        self._cached_tick_key = now

    def scale_decision(self, service_id, metrics_snapshot):
        return self._cached_scale.get(service_id, NO_CHANGE)

    def provision_decision(self, servers, metrics_snapshot, now):
        self._predict_and_cache(servers, metrics_snapshot, now)
        return self._cached_provision

    def select_placement_server(self, service_id, servers):
        # همان منطق centroid-aware بخش ۱۴.۶، از self._last_snapshot می‌خواند
        ...

    def migration_decision(self, draining_server, servers):
        return self._helper.migration_decision(draining_server, servers)

    def select_replica(self, request, candidate_replicas, servers, now, admit_fn=None, occupancy_fn=None):
        if not self.latency_aware_routing:
            return super().select_replica(...)
        # منطق latency-aware بخش ۸
```
`initial_placement` طبق بخش ۹.۲.

### ۱۴.۸ آموزش (`algorithms/ppo/train.py`)

جریان کلی:
1. **جمع‌آوری دموی Greedy برای BC**: `SimulationEngine` با `GreedyAlgorithm` روی train اجرا می‌شود؛ هر تیک `(obs, action_mask, applied_action)` ثبت می‌شود — `applied_action` باید از `engine._last_tick_applied_actions` گرفته شود (چیزی که واقعاً اعمال شد، نه پیشنهاد خام که ممکن است رد شده باشد). `action_mask` **قبل از** `engine.step()` (با همان `last_snapshot` که در تیک قبلی/peek اولیه محاسبه شده) گرفته می‌شود. حداکثر `bc_max_ticks` تیک (پیش‌فرض ۱۰۰۰۰).
2. **BC warm-start** (`behavior_cloning_pretrain`): مستقیم روی `model.policy` (torch nn.Module واقعی sb3) با cross-entropy روی `MultiDiscrete` action distribution، با `action_masks` پاس داده‌شده به `model.policy.get_distribution(obs, action_masks=mask)`. `Adam(lr=5e-5)`, پیش‌فرض ۵۰ epoch، batch=64. loss هر epoch در `logs/bc_warmstart_loss.csv` ذخیره می‌شود.
3. **Fine-tune با MaskablePPO**: `n_envs=8` محیط موازی (`DummyVecEnv`)، هر کدام seed جدا (`CFG.seed + i`) و پنجره‌ی تصادفی ۲۴ساعته از تایم‌لاین سه‌روزه‌ی train (`make_random_window_provider`). هر env با `Monitor(ActionMasker(EdgeResourceEnv(provider), mask_fn), filename=...)` می‌شود. `VecNormalize(norm_obs=False, norm_reward=True, gamma=0.99)`. `MaskablePPO("MlpPolicy", vec_env, n_steps=2048, batch_size=256, gamma=0.99, learning_rate=3e-4, ent_coef=0.01, policy_kwargs={"net_arch": {"pi":[256,256], "vf":[256,256]}}, tensorboard_log=logs/tensorboard, seed=CFG.seed)`. آموزش با `model.learn(total_timesteps=3_000_000, callback=[CheckpointCallback(save_freq=200_000/n_envs), RewardComponentLoggingCallback])`.
4. مدل نهایی در `algorithms/ppo/ppo_model_seed{SEED}.zip` و `VecNormalize` state در همان مسیر با پسوند `_vecnormalize.pkl` ذخیره می‌شود (فقط برای resume، در inference لود نمی‌شود).

### ۱۴.۹ Inference-only (`algorithms/ppo/infer.py`)

روی `Data4.csv` (test)، `deterministic=True`، بدون یادگیری. کش هر تیک (بخش ۱۴.۷) از forward pass تکراری جلوگیری می‌کند. آرگومان‌های CLI: `--seed`, `--latency-aware-routing`, `--no-solver-placement`, `--w-count/--w-energy/--w-distance`, `--output-dir`.

---

## بخش ۱۵: اجرای واقعی روی Kubernetes

### ۱۵.۱ معماری کلی

- **BTS Simulator** (`bts_simulator.py`): دیتاست را با **زمان‌بندی واقعی wall-clock** replay می‌کند. برای هر رکورد: (۱) `POST /route` به دیسپچر مرکزی (فقط تصمیم مسیریابی، سبک)؛ (۲) اگر `status=ROUTED`، `POST /process` مستقیم به `ip:port` پاد (بدون واسطه‌ی دیسپچر). `sent_at_epoch` قبل از مرحله‌ی ۱ ثبت می‌شود.
- **Dispatcher API** (`dispatcher_api.py`, FastAPI, پورت ۹۰۰۰): دو endpoint — `POST /route` (فراخوانی `engine.route_request`) و `POST /report` (فراخوانی `engine.record_external_completion`، fire-and-forget — **در عمل استفاده نمی‌شود چون worker خودش گزارش را از طریق Redis می‌فرستد**).
- **Worker Service** (`worker_service/app.py`, یک ایمیج Docker برای هر ۱۵ سرویس، تفاوت فقط env vars): `/healthz`, `/process` (با `asyncio.Semaphore(1)` — دقیقاً یک درخواست هم‌زمان). بعد از پردازش: (۱) شمارنده‌ی صف Redis آزاد می‌شود، (۲) رزرو ZSET پاک می‌شود، (۳) یک رکورد completion در `edge:metrics:completions` push می‌شود، (۴) `busy_seconds_acc` افزایش می‌یابد.
- **RealtimeEngine** (`realtime_dispatcher.py`): معادل async این `SimulationEngine`، با تسک‌های موازی: `decision_loop` (هر ۳۰ ثانیه)، `drain_completion_queue` (هر ۰.۲ ثانیه)، `_utilization_energy_sampler_loop` (هر ۵ ثانیه)، `_reservation_sweeper_loop` (هر ۱۰ ثانیه)، `_lifetime_watcher` (توقف بعد از پایان بازه‌ی داده).
- **Redis** (`192.168.1.30:6379`): هماهنگی state لحظه‌ای بین `decision_loop` و `dispatcher_api`.
- **k8s_client**: ساخت/حذف Deployment واقعی، cordon/uncordon نود.

### ۱۵.۲ کلیدهای Redis

```
edge:server:{id}:state              -> "OFF"|"BOOTING"|"ACTIVE"|"DRAINING"
edge:replica:{svc}:{srv}:state      -> "STARTING"|"READY"|"DRAINING"|"TERMINATED"
edge:replica:{svc}:{srv}:pod_ip     -> IP پاد
edge:replica:{svc}:{srv}:queue      -> شمارنده‌ی اشغال صف (INCR/DECR اتمیک)
edge:service:{svc}:ready_replicas   -> SET سرورهای READY آن سرویس
edge:reservations:{svc}:{srv}       -> ZSET رزروهای صف، score=زمان انقضا
edge:metrics:completions            -> LIST پیام‌های تکمیل (fire-and-forget)
service:{svc}:server:{srv}:busy_seconds_acc -> شمارنده‌ی دقیق busy-seconds برای energy
```

توابع کلیدی (`redis_state.py`):
```python
def try_reserve_queue_slot(service_id, server_id, queue_len, request_id=None, ttl_sec=None):
    key = f"edge:replica:{service_id}:{server_id}:queue"
    new_val = INCR(key)
    if new_val > queue_len:
        DECR(key); return False
    if request_id is not None and ttl_sec is not None:
        ZADD(f"edge:reservations:{service_id}:{server_id}", {str(request_id): time.time()+ttl_sec})
    return True

def release_queue_slot(service_id, server_id, request_id=None):
    key = f"edge:replica:{service_id}:{server_id}:queue"
    if int(GET(key) or 0) > 0: DECR(key)
    if request_id is not None: ZREM(f"edge:reservations:{service_id}:{server_id}", str(request_id))

def sweep_expired_reservations(n_servers, n_services):
    # برای هر (svc,srv): ZRANGEBYSCORE با upper=now؛ هرکدام expired را ZREM کن و
    # release_queue_slot صدا بزن؛ تعداد آزادشده را برگردان
    ...

def pop_busy_seconds_acc(service_id, server_id):
    return float(GETSET(f"service:{service_id}:server:{server_id}:busy_seconds_acc", 0.0) or 0.0)
```
**رزرو صف بدون نشتی:** یک شمارنده‌ی INCR/DECR ساده کافی نیست — اگر پاد بین رزرو و پردازش کرش کند، شمارنده برای همیشه بالاتر از واقعی می‌ماند. راه‌حل: هر رزرو موفق هم در شمارنده و هم در یک ZSET با `score=now+ttl_sec` (`ttl_sec = deadline_sec + 5`) ثبت می‌شود؛ worker موفق خودش رزرو را از ZSET پاک می‌کند؛ یک تسک دوره‌ای (`RESERVATION_SWEEP_INTERVAL_SEC=10.0`) رزروهای منقضی‌شده‌ی جامانده را می‌یابد و شمارنده را آزاد می‌کند.

### ۱۵.۳ `RealtimeEngine.route_request(request_id, service_id, bts_lat, bts_long)`

```python
async def route_request(request_id, service_id, bts_lat, bts_long):
    route_started = time.monotonic()
    tick_total[service_id] += 1
    log("request_arrived", ...)
    recent_positions[service_id].append((bts_lat, bts_long))   # برای demand_centroid

    deadline = SERVICES_INFO[service_id]["deadline"]
    reservation_ttl_sec = deadline + 5

    candidates = [r for r in replicas_by_service[service_id] if r.is_selectable()]
    request_obj = SimpleNamespace(bts_lat=bts_lat, bts_long=bts_long, service_id=service_id)

    chosen = algorithm.select_replica(
        request_obj, candidates, servers, time.monotonic(),
        occupancy_fn=lambda r: redis_state.get_queue_occupancy(service_id, r.server_id),
        admit_fn=lambda r: redis_state.try_reserve_queue_slot(
            service_id, r.server_id, r.queue_len, request_id=request_id, ttl_sec=reservation_ttl_sec))

    if chosen is None:
        status = "REJECTED_NO_REPLICA" if not candidates else "REJECTED_QUEUE_FULL"
        tick_rejected[service_id] += 1; tick_violated[service_id] += 1
        log("request_rejected", ...)
        rejected_req = Request(id=request_id, ..., status=matching_status)
        metrics.record_request(rejected_req)
        return {"status": status}

    server = servers[chosen.server_id]
    distance_km = haversine_km(bts_lat, bts_long, server.lat, server.long)
    delay_ms = network_delay_ms(distance_km, BASE_LATENCY_MS, K_MS_PER_KM)
    if 2*delay_ms >= PROXIMITY_L0_MS:
        tick_proximity_violated[service_id] += 1
    log("request_routed", request_id=request_id, server_id=server.id, distance_km=distance_km, network_delay_ms=delay_ms)
    request_geo[request_id] = (distance_km, delay_ms, time.monotonic())   # برای completion بعدی

    ip = redis_state.get_pod_ip(service_id, chosen.server_id)
    port = k8s_client.worker_port(service_id)   # = 8000 + service_id
    routing_elapsed = max(0.0, time.monotonic() - route_started)
    remaining_deadline = max(deadline - routing_elapsed, 0.1)
    return {"status": "ROUTED", "server_id": server.id, "ip": ip, "port": port, "deadline_sec": remaining_deadline}
```

### ۱۵.۴ Worker `app.py` — `/process` endpoint

```python
EXEC_TIME_SEC, SERVICE_ID, SERVER_ID from env vars

@app.post("/process")
async def process(req):
    async with process_semaphore:   # Semaphore(1)
        start = time.monotonic()
        await asyncio.sleep(EXEC_TIME_SEC)
        elapsed = time.monotonic() - start
    # آزادسازی صف + پاک کردن رزرو ZSET
    key = f"edge:replica:{SERVICE_ID}:{SERVER_ID}:queue"
    if int(redis.get(key) or 0) > 0: redis.decr(key)
    if req.request_id is not None:
        redis.zrem(f"edge:reservations:{SERVICE_ID}:{SERVER_ID}", str(req.request_id))
    response_time_sec = elapsed
    if req.sent_at_epoch is not None:
        response_time_sec = time.time() - req.sent_at_epoch
    redis.rpush("edge:metrics:completions", json.dumps({"request_id":..., "service_id":..., "server_id":...,
                                                          "success": True, "response_time_sec": response_time_sec}))
    redis.incrbyfloat(f"service:{SERVICE_ID}:server:{SERVER_ID}:busy_seconds_acc", elapsed)
    return {...}
```
محدودیت «هر رپلیکا هم‌زمان فقط ۱ درخواست» با `asyncio.Semaphore(1)` **دور خودِ endpoint `/process`** پیاده می‌شود، نه با `--limit-concurrency` سطح uvicorn (آن فلگ `/healthz` را هم می‌گیرد و پاد هیچ‌وقت Ready نمی‌شود).

### ۱۵.۵ `_utilization_energy_sampler_loop` (هر `UTIL_SAMPLE_INTERVAL_SEC=5.0` ثانیه)

```python
async def utilization_energy_sampler_loop():
    last_sample = time.monotonic()
    while running:
        await sleep(5.0)
        now = time.monotonic(); elapsed = now - last_sample; last_sample = now
        for sid, s in servers.items():
            if s.state == OFF: continue
            busy_mips_seconds = 0.0
            if s.state in (ACTIVE, DRAINING):
                for svc_id, r in s.hosted_replicas.items():
                    if r.state not in (READY, DRAINING): continue
                    exact_busy_sec = min(redis_state.pop_busy_seconds_acc(svc_id, sid), elapsed)
                    # *** الزامی: باید با _speed_factor تبدیل شود، نه resource_mips خام ***
                    busy_mips_seconds += exact_busy_sec * round(SERVICES_INFO[svc_id]["resource_mips"] * s._speed_factor())
            s.cumulative_busy_cpu_seconds += busy_mips_seconds
            if s.state == BOOTING: power = s.p_idle
            elif s.state in (ACTIVE, DRAINING):
                avg_util = (busy_mips_seconds/elapsed)/s.capacity if s.capacity>0 and elapsed>0 else 0.0
                power = s.p_idle + (s.p_max - s.p_idle) * avg_util
            else: power = 0.0
            s.cumulative_energy_joule += power * elapsed
```
**تذکر حیاتی برای پیاده‌سازی صحیح (رفع یک کلاس باگ رایج)**: `busy_mips_seconds` باید با `resource_mips` **effective** (یعنی `round(resource_mips * server._speed_factor())`) محاسبه شود، نه با `resource_mips` خام. اگر خام استفاده شود، utilization/انرژی برای سرورهای `edge_small` و `large` سیستماتیک اشتباه محاسبه می‌شود (برای `medium` صحیح می‌ماند چون `speed_factor=1.0`). این خطا مستقیماً روی PPO state vector (utilization هر سرور و `energy_recent_joule`) اثر می‌گذارد و کیفیت آموزش را خراب می‌کند.

### ۱۵.۶ Provisioning/Scale در RealtimeEngine

منطق `_apply_provisioning`/`_apply_scale_decision` باید **دقیقاً معادل async** نسخه‌ی sim باشد (بخش ۱۰.۶ و ۱۳.۵)، با این تفاوت‌ها:
- `TURN_ON` → `await activate_server(server_id)`: `k8s_client.uncordon_node(server_id)`, `redis_state.set_server_state(server_id, "ACTIVE")`, `s.state=ACTIVE`, `s.last_transition_time=now`, `s.num_boots+=1`, **`s.cumulative_energy_joule += E_BOOT_SERVER_J`** (این خط الزامی است — بدون آن انرژی boot در حالت k8s گزارش نمی‌شود)، `metrics.record_transition("server_boot")`.
- `TURN_OFF` → `await drain_server(server_id)`: معادل async `_start_server_drain` (بخش ۱۰.۸)، شامل: `migration_decision`، pre-validation ظرفیت، تشخیص sole-hosted، `trigger_emergency_boot` اگر لازم، ساخت رپلیکای جدید با `create_replica` (که `k8s_client.create_deployment` + `redis_state.set_replica_state(...,"STARTING")` + **`s.cumulative_energy_joule += E_POD_CREATE_J`** را صدا می‌زند)، انتظار `_wait_specific_ready` تا رپلیکاهای مقصد READY شوند (پولینگ `k8s_client.is_deployment_ready`)، سپس حذف رپلیکاهای قدیمی با `_delete_replica` (صبر تا `queue_occupancy<=0` یا `max_wait` timeout، سپس `k8s_client.delete_deployment`).
- `_create_replica` یک تسک پس‌زمینه‌ی `_poll_until_ready` را spawn می‌کند که هر ۱ ثانیه `k8s_client.is_deployment_ready` را چک می‌کند تا `timeout=120.0` ثانیه؛ وقتی ready شد، `pod_ip` را از `k8s_client.get_pod_ip` می‌گیرد و در Redis + شیء `Replica` ثبت می‌کند.
- همه‌ی تسک‌های پس‌زمینه‌ی fire-and-forget (حذف رپلیکا هنگام SCALE_DOWN، poll-until-ready) باید در یک `set` جهانی ثبت شوند (`_spawn_background_task`) و در پایان `run()` با `asyncio.gather(*background_tasks, return_exceptions=True)` منتظر ماند — وگرنه ممکن است `metrics.finalize()` قبل از تکمیل واقعی این تسک‌ها اجرا شود.

### ۱۵.۷ `_build_metrics_snapshot` در RealtimeEngine

باید تمام کلیدهایی که `PPOAlgorithm._build_action_masks` بدون `.get` می‌خواند را حتماً داشته باشد: `snapshot["servers"][sid]["provision_cooldown_active"]`, `["min_active_duration_met"]`, `["is_last_active_server"]`. و `snapshot["global"]["avg_response_time_recent"]`/`["energy_recent_joule"]`/`["num_rejected_recent"]` باید واقعی محاسبه شوند (نه هاردکد صفر) — از `_tick_response_times` (پر شده در `record_external_completion`) و دلتای `cumulative_energy_joule` بین دو تیک.

### ۱۵.۸ `record_external_completion` — بازیابی geo برای متریک نهایی

چون completion از طریق صف Redis async می‌آید (بدون دسترسی مستقیم به مختصات BTS)، `distance_km`/`network_delay_ms` باید از یک دیکشنری موقت `_request_geo[request_id] = (distance_km, delay_ms, timestamp)` (که در `route_request` پر شده) بازیابی و به `Request` منتقل شود. اگر completion هرگز نرسید (پاد کرش کرد)، یک sweep دوره‌ای (`_prune_stale_request_geo`، در همان حلقه‌ی `_reservation_sweeper_loop`) ورودی‌های قدیمی‌تر از `max(deadline) + RESERVATION_SWEEP_INTERVAL_SEC + 5.0` را پاک می‌کند تا نشتی حافظه رخ ندهد.

### ۱۵.۹ `k8s_client.py`

```python
NAMESPACE = "edge-rl"
WORKER_IMAGE = "192.168.1.30:5000/edge-worker:latest"
NODE_LABEL_KEY = "edge-server-id"

def worker_port(service_id): return 8000 + service_id

def resource_mips_to_millicpu(resource_mips):
    return round(resource_mips / REFERENCE_MIPS_PER_CORE * 1000)   # همیشه نسبت به مرجع medium، نه پروفایل میزبان

def build_deployment_manifest(service_id, server_id):
    # container با env EXEC_TIME_SEC (=compute_exec_time_sec برای همان سرور)، SERVICE_ID, SERVER_ID, SERVICE_PORT
    # node_selector={NODE_LABEL_KEY: str(server_id)}, host_network=True, dns_policy=ClusterFirstWithHostNet
    # readiness_probe: GET /healthz روی همان پورت، initial_delay=1, period=2, failure_threshold=3
    # resources.requests/limits: cpu=f"{millicpu}m", memory=svc["memory"]
    ...

def create_deployment(service_id, server_id): ...   # 409 => از قبل موجود، نادیده بگیر
def delete_deployment(service_id, server_id): ...   # 404 => نادیده بگیر
def is_deployment_ready(service_id, server_id) -> bool: ...  # ready_replicas >= 1
def get_pod_ip(service_id, server_id) -> Optional[str]: ...
def cordon_node(server_id): ...    # unschedulable=True  (معادل server->OFF)
def uncordon_node(server_id): ...  # unschedulable=False (معادل server->ACTIVE)
```
هر فراخوانی API که ممکن است خطای transient بدهد باید با `_call_with_retry` (۳ تلاش، backoff نمایی، فقط برای status ۵۰۰/۵۰۳/۴۲۹) پیچیده شود.

---

## بخش ۱۶: لاگ ساخت‌یافته (`common/logger.py`)

```python
class EventLogger:
    def __init__(self, path, algorithm, enabled=True):
        self.algorithm=algorithm; self.enabled=enabled
        if enabled:
            os.makedirs(dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def log(self, event_type, sim_time=None, **fields):
        if not self.enabled: return
        record = {"event_type": event_type, "algorithm": self.algorithm,
                   "sim_time_sec": sim_time, "wall_time": datetime.now(timezone.utc).isoformat(), **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self): ...
```
هر رکورد یک خط JSON مستقل (JSONL). خروجی نهایی هر اجرا: `<algorithm>_events.jsonl` + `<algorithm>_result.json`.

انواع event type که باید لاگ شوند: `request_arrived, request_routed, request_queued, request_completed, request_rejected, unexpected_admit_race, server_boot_started, server_active, server_drain_started, server_off, server_drain_aborted, pod_create_started, pod_ready, pod_drain_started, pod_terminated, pod_ready_timeout, pod_drain_timeout_forced, scale_decision, provision_decision, migration_started, migration_completed, migration_placement_failed, migration_ready_timeout, migration_step_dropped, emergency_boot_triggered, emergency_boot_completed, reservation_sweep, initial_placement_ready_timeout, realtime_run_finished`.

---

## بخش ۱۷: بارگذاری داده (`data/loader.py`)

```python
COLUMNS = ["id", "BTSID", "Lat", "Long", "ServiceID", "startSec"]

def load_one_day(filename, day_index):
    df = read_csv(path, usecols=COLUMNS)
    df = df[df.Lat.between(LAT_MIN, LAT_MAX) & df.Long.between(LON_MIN, LON_MAX)]
    df = df[df.ServiceID.isin(ACTIVE_SERVICES)]
    df["day_index"] = day_index
    df["global_start_sec"] = day_index * 86400 + df["startSec"]
    return df

def load_timeline(filenames):
    frames = [load_one_day(f, i) for i, f in enumerate(filenames)]
    events = concat(frames).sort_values("global_start_sec").reset_index(drop=True)
    return events

def load_train(): return load_timeline(["Data1.csv","Data2.csv","Data3.csv"])  # 3 روز پیوسته
def load_test(): return load_timeline(["Data4.csv"])                          # 1 روز مستقل
```
مسیر داده با `EOTCH_DATA_DIR` (پیش‌فرض `<project_root>/data/raw`) قابل تنظیم است.

---

## بخش ۱۸: رابط مشترک `AlgorithmBase` (خلاصه‌ی نهایی)

```python
class AlgorithmBase(ABC):
    name: str = "base"
    bypass_sustain_gate: bool = True

    def initial_placement(self, servers, active_bts) -> List[int]: ...            # پیاده‌سازی مشترک، بخش ۹.۱
    def select_replica(self, request, candidate_replicas, servers, now,
                        admit_fn=None, occupancy_fn=None) -> Optional[Replica]: ... # پیاده‌سازی مشترک، بخش ۸
    def select_scale_down_victim(self, service_id, ready_replicas, servers, now,
                                  occupancy_fn=None) -> Replica: ...               # پیش‌فرض: کمترین اشغال؛ VOILA override
    @staticmethod
    def _pick_profile_for_overload(overloaded_servers, fallback_capacity) -> str: ...   # بخش ۱۰.۵
    @staticmethod
    def _filter_by_profile_with_fallback(candidates, desired_profile) -> List[Server]: ...
    @staticmethod
    def _capacity_starved_services(metrics_snapshot, servers, occ_threshold=0.7) -> List[int]: ...

    @abstractmethod
    def scale_decision(self, service_id, metrics_snapshot) -> ScaleAction: ...
    @abstractmethod
    def provision_decision(self, servers, metrics_snapshot, now) -> ProvisionAction: ...
    @abstractmethod
    def select_placement_server(self, service_id, servers) -> Optional[int]: ...
    @abstractmethod
    def migration_decision(self, draining_server, servers) -> List[MigrationStep]: ...
```
برای اضافه‌کردن یک الگوریتم جدید: کلاسی از `AlgorithmBase` بسازید، متدهای انتزاعی را پیاده کنید، و آن را در نقاط ورودی (`run.py`, `evaluation/compare_runs.py`) اضافه کنید — موتور شبیه‌سازی و adapter کلاستر نباید تغییر کنند.

---

## بخش ۱۹: ابزارهای ارزیابی و کالیبراسیون

### `evaluation/compare_runs.py`
هر ۴ الگوریتم را روی همان داده (train/test) اجرا می‌کند، هرکدام `<name>_events.jsonl` + `<name>_result.json` جدا می‌نویسد، و یک `comparison_summary.csv` (بدون ستون `decision_correctness` که تودرتوست) تولید می‌کند. اگر ساخت یک الگوریتم (مثلاً مدل PPO seed مشخص پیدا نشد) شکست خورد، آن را رد و ادامه می‌دهد.

### `evaluation/aggregate_seeds.py`
میانگین/std/min/max هر معیار کلیدی را روی چند اجرای seed مختلف PPO گزارش می‌دهد (خواندن `outputs/seed{N}/{algorithm}_result.json`).

### `calibrate_constants.py`
یک اجرای کامل Greedy را دنبال می‌کند، توزیع `recent_arrivals` (هر سرویس هر تیک)، `avg_response_time_recent` (فقط تیک‌های غیرصفر)، `energy_recent_joule`، `num_rejected_recent` (هم خام هم فقط-غیرصفر) را جمع و mean/median/p90/p95/p99/max چاپ می‌کند. روش انتخاب ثابت نرمال‌سازی: p90 یا p95 (نه max مطلق، چون یک outlier کل مقیاس را خراب می‌کند).

### `analyze_decision_quality.py`
دو تحلیل: (۱) طبقه‌بندی SCALE_UP/DOWN «غیرضروری طبق ممیزی لحظه‌ای» به anticipatory/noise (SCALE_UP) یا harmless_early/risky (SCALE_DOWN) با نگاه به `LOOKAHEAD_TICKS=15` تیک بعدی و/یا رد شدن واقعی طی `REJECTION_LOOKAHEAD_SEC=90` ثانیه. (۲) تحلیل flapping سرور: چرخه‌های `TURN_ON→TURN_OFF` هر سرور، `dwell = off_time - on_time`، آستانه‌ی `FLAPPING_DWELL_SEC=300`.

### `analyze_scaleup_by_service.py`, `analyze_necessity_by_service.py`, `diagnose_violations_by_service.py`
اسکریپت‌های تحلیل توزیع SCALE_UP/نقض deadline بر حسب سرویس، برای تشخیص این‌که آیا رفتار سیستم ساختاری (وابسته به deadlineهای سفت خاص) است یا رفتار واقعی الگوریتم. جزئیات دقیق منطق شمارش هرکدام در فایل‌های اصلی مستند شده و از نظر عملکردی جزو «هسته‌ی» سیستم نیستند — بازتولیدشان اختیاری است اما فرمول‌های شمارش (بالا در همین اسکریپت‌ها گفته شد) باید دقیق رعایت شود اگر بازسازی می‌شوند.

---

## بخش ۲۰: ساختار پوشه‌ها و وابستگی‌ها

```
edge_rl/
  common/{models.py, config.py, metrics.py, geo.py, logger.py, state_builder.py, network_coordinates.py}
  data/loader.py
  simulator/{engine.py, events.py}
  algorithms/
    base.py
    greedy/greedy_algorithm.py
    voila/voila_algorithm.py
    hpa/hpa_algorithm.py
    ppo/{env.py, policy_network.py, train.py, infer.py, ppo_algorithm.py, optimal_placement.py}
  k8s_adapter/
    {k8s_client.py, redis_state.py, realtime_dispatcher.py, dispatcher_api.py, smoke_test.py}
    worker_service/{Dockerfile, app.py, requirements.txt, bts_simulator.py}
  evaluation/{compare_runs.py, aggregate_seeds.py}
  analyze_decision_quality.py
  analyze_scaleup_by_service.py
  analyze_necessity_by_service.py
  diagnose_violations_by_service.py
  calibrate_constants.py
  build_push_pull_worker.py
  run.py
  requirements.txt
```

```
# requirements.txt
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
gymnasium>=0.29
stable-baselines3>=2.2
sb3-contrib>=2.2
torch>=2.0
kubernetes>=29.0
redis>=5.0
httpx>=0.27
pulp>=2.7
# k8s_adapter/worker_service/requirements.txt (جدا، سبک‌تر)
fastapi==0.115.*
uvicorn==0.30.*
pydantic==2.*
redis>=5.0
```
برای اجرای واقعی روی k8s جداگانه لازم است: `paramiko` (برای `build_push_pull_worker.py`).

متغیرهای محیطی: `EOTCH_DATA_DIR` (مسیر CSVها)، `EOTCH_SEED` (seed آموزش/اجرا، پیش‌فرض ۴۲)، `EOTCH_NORM_REJECTED_PER_TICK` (پیش‌فرض ۲.۰).

---

## بخش ۲۱: نقشه‌ی ساخت پیشنهادی (ترتیب پیاده‌سازی از صفر)

1. **`common/`**: `models.py` (dataclasses + enums)، `config.py` (تمام ثابت‌ها دقیقاً طبق بخش ۶)، `geo.py`، `logger.py`. اینجا باید `is_sla_feasible` و `compute_exec_time_sec`/`compute_cold_start_*` هم اضافه شوند.
2. **`data/loader.py`** — فقط بعد از داشتن ۴ فایل CSV با ستون‌های `id, BTSID, Lat, Long, ServiceID, startSec`.
3. **`algorithms/base.py`** — رابط مشترک + `select_replica`/`initial_placement` مشترک.
4. **`simulator/engine.py` + `events.py`** — موتور discrete-event کامل، شامل چرخه‌ی کامل درخواست، provisioning/migration، auto-scaling. این بزرگ‌ترین و حیاتی‌ترین ماژول است — قبل از هر چیز دیگری کامل و تست شود (مثلاً با یک الگوریتم بسیار ساده که همیشه NO_CHANGE برمی‌گرداند، برای اطمینان از این‌که چرخه‌ی رویداد پایه بدون خطا اجرا می‌شود).
5. **`common/metrics.py`** — `MetricsCollector`.
6. **چهار الگوریتم**: Greedy (baseline ساده)، سپس HPA (فرمول استاندارد ثابت)، سپس VOILA (medoid/proximity، پیچیده‌تر)، در آخر PPO (نیازمند `common/state_builder.py`، `algorithms/ppo/env.py`، آموزش با BC warm-start).
7. **`evaluation/`** — بعد از این‌که هر ۴ الگوریتم قابل‌اجرا هستند، برای مقایسه‌ی چهارگانه و تحلیل کیفیت تصمیم.
8. **`k8s_adapter/`** — در نهایت، برای پورت کردن همان اشیاء `AlgorithmBase` (بدون تغییر) به یک محیط real-time واقعی: Redis برای هماهنگی، FastAPI برای دیسپچر، Kubernetes client برای مدیریت واقعی پاد/نود، و worker service به‌عنوان بار واقعی محاسباتی.

**اصل نهایی که همیشه باید رعایت شود:** هر بار که یک منطق در `simulator/engine.py` تغییر می‌کند (مثلاً یک بررسی جدید، یک فرمول اصلاح‌شده، یک آستانه)، دقیقاً همان تغییر باید در `k8s_adapter/realtime_dispatcher.py` هم اعمال شود — این دو نباید هرگز از هم واگرا شوند، وگرنه مقایسه‌ی نتایج شبیه‌سازی با نتایج اجرای واقعی بی‌اعتبار می‌شود.

---

## بخش ۲۲: مرجع کامل خط فرمان (CLI) هر اسکریپت اجرایی

### `run.py` — نقطه‌ی ورود اصلی
```
python run.py --algorithm {greedy,voila,hpa,ppo} [--mode {sim,k8s}] [--data {train,test}]
              [--output-dir DIR] [--latency-aware-routing] [--no-solver-placement]
```
پیش‌فرض‌ها: `--mode sim`, `--data test`, `--output-dir outputs`. رفتار دقیق:
```python
def main():
    os.makedirs(output_dir, exist_ok=True)
    events = load_train() if data=="train" else load_test()
    algorithm = build_algorithm(algorithm_name, args)   # همان build_algorithm بخش‌های قبلی
    logger = EventLogger(f"{output_dir}/{algorithm_name}_events.jsonl", algorithm=algorithm_name)
    if mode == "k8s":
        result = asyncio.run(serve_control_plane(events, algorithm, algorithm_name, event_logger=logger))
    else:
        engine = SimulationEngine(events, algorithm, algorithm_name, event_logger=logger)
        result = engine.run()
    logger.close()
    json.dump(result, open(f"{output_dir}/{algorithm_name}_result.json","w"), indent=2, ensure_ascii=False, default=str)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
```
اگر `--algorithm ppo` و مدل seed فعلی (`CFG.seed`) پیدا نشود، خطای صریح با راهنمای اجرای `python -m algorithms.ppo.train` داده می‌شود (بدون کرش مبهم).

### `algorithms.ppo.train` (بدون CLI args — پارامترها داخل `main()` هاردکد هستند)
```
python -m algorithms.ppo.train
```
پارامترهای قابل تغییر در امضای `main()`: `total_timesteps=3_000_000, bc_epochs=50, window_hours=24.0, bc_max_ticks=10_000, n_envs=8`.

### `algorithms.ppo.infer`
```
python -m algorithms.ppo.infer [--seed N] [--latency-aware-routing] [--no-solver-placement]
                                [--w-count F] [--w-energy F] [--w-distance F] [--output-dir DIR]
```
اگر `--seed` داده شود، مسیر مدل `algorithms/ppo/ppo_model_seed{N}.zip` و خروجی در `<output-dir>/seed{N}/` می‌رود. همیشه روی `Data4.csv` (test) اجرا می‌شود؛ log در `<output-dir>/ppo_events.jsonl`، نتیجه در `<output-dir>/ppo_result.json`.

### `evaluation.compare_runs`
```
python -m evaluation.compare_runs [--data {train,test}] [--output-dir DIR] [--seed N]
                                    [--latency-aware-routing] [--no-solver-placement]
                                    [--w-count F] [--w-energy F] [--w-distance F]
```
هر ۴ الگوریتم (`greedy, voila, hpa, ppo`) را به ترتیب همین لیست اجرا می‌کند. اگر ساخت یکی (مثلاً `ImportError`/`NotImplementedError`/`FileNotFoundError` برای مدل PPO نبود) شکست بخورد، پیام `[رد شد] {name}: {error}` چاپ و رد می‌شود، بدون توقف بقیه. در پایان `comparison_summary.csv` (بدون ستون `decision_correctness`، چون تودرتوست و در CSV خوانا نیست) در `output_dir` نوشته می‌شود.

### `evaluation.aggregate_seeds`
```
python -m evaluation.aggregate_seeds --seeds N1 N2 N3 ... [--base-dir DIR] [--algorithm NAME]
```
پیش‌فرض `--base-dir outputs`, `--algorithm ppo`. برای هر seed فایل `<base-dir>/seed{N}/{algorithm}_result.json` را می‌خواند (اگر نبود رد می‌کند)؛ برای هر متریک در لیست زیر mean/std/min/max چاپ می‌کند:
```python
METRICS = ["avg_response_time_sec", "p95_response_time_sec", "p99_response_time_sec",
           "deadline_violation_rate_pct", "cumulative_energy_joule", "avg_distance_km",
           "avg_load_balance_cv", "num_requests_rejected_queue_full",
           "avg_active_servers", "completed_requests"]
```
سپس یک جدول خام CSV-style (هر سطر یک seed، ستون‌ها = `seed` + METRICS) هم چاپ می‌کند.

### اسکریپت‌های تحلیل (بدون argparse، آرگومان اول خط فرمان مسیر فایل است)
```
python analyze_decision_quality.py <path/to/*_events.jsonl>
python analyze_scaleup_by_service.py <path/to/*_events.jsonl>
python analyze_necessity_by_service.py <path/to/*_events.jsonl>
python diagnose_violations_by_service.py <path/to/*_events.jsonl>
```
اگر آرگومان داده نشود، پیام راهنما چاپ و با `sys.exit(1)` خارج می‌شود.

### `k8s_adapter.smoke_test`
```
python3 -m k8s_adapter.smoke_test
```
چهار تست به ترتیب اجرا می‌شوند (و اگر یکی از سه‌ی اول شکست بخورد، تست چهارم اصلاً اجرا نمی‌شود):
1. `test_redis()`: `ping()`، سپس نوشتن/خواندن یک کلید آزمایشی `edge:server:999:state`، سپس پاک‌سازی.
2. `test_k8s_connection()`: `_core_v1.list_node()` و چاپ تعداد نودها.
3. `test_node_labels()`: برای هر ۱۰ `server_id` در `CFG.server_info`، `_get_node_name(sid)` باید بدون خطا اجرا شود (یعنی یک نود با لیبل `edge-server-id=<sid>` وجود دارد)؛ اگر نبود، لیست سرورهای بدون لیبل چاپ می‌شود.
4. `test_deployment_roundtrip()`: `uncordon_node(1)` → `create_deployment(service_id=1, server_id=1)` → پولینگ حداکثر ۶۰ ثانیه (هر ۲ ثانیه) تا `is_deployment_ready(1,1)` و گرفتن IP → یک `POST /process` واقعی با `httpx` به `http://{ip}:{port}/process` با `{"request_id": 1}` و timeout=15 → در بلوک `finally` همیشه `delete_deployment(service_id=1, server_id=1)` صدا زده می‌شود (پاک‌سازی، حتی اگر مراحل قبلی شکست خورده باشند).
خروجی نهایی: `sys.exit(0)` اگر همه موفق، `sys.exit(1)` وگرنه.

### `build_push_pull_worker.py`
```
python3 build_push_pull_worker.py
```
بدون آرگومان CLI؛ تنظیمات به‌صورت ثابت در بالای فایل: `base_dir` (مسیر `k8s_adapter/worker_service` روی ماشین اجراکننده)، `docker_image = "192.168.1.30:5000/edge-worker:latest"`، `worker_nodes` (لیست IP هر ۱۰ نود worker، `192.168.1.11` تا `.20`؛ **master `.10` عمداً در این لیست نیست**)، `ssh_user`, `ssh_key_path`, `SUDO_PASS`. مراحل به ترتیب:
1. اگر ایمیج محلی از قبل موجود بود، حذف (`docker rmi -f`).
2. `docker build --network host -t <image> .` (در `base_dir`).
3. `docker push <image>` به رجیستری محلی.
4. برای هر نود worker (به ترتیب لیست، پشت‌سرهم نه موازی): از طریق SSH (با `paramiko`)، ابتدا تلاش حذف ایمیج قدیمی با `ctr -n k8s.io images remove` (خطا در این مرحله بی‌ضرر و نادیده گرفته می‌شود چون ممکن است قبلاً وجود نداشته)، سپس `ctr -n k8s.io images pull --plain-http <image>`.
5. موفقیت/شکست SSH با **exit status واقعی کانال** (`stdout.channel.recv_exit_status()`) تشخیص داده می‌شود، **نه** با خالی‌بودن `stderr` (چون `sudo -S` همیشه یک پیام «رمز عبور» روی stderr می‌نویسد حتی در حالت موفق — تشخیص بر پایه‌ی stderr همیشه یک False Negative تولید می‌کرد).
6. در پایان، لیست نودهای واقعاً شکست‌خورده (`exit_status != 0`) چاپ می‌شود؛ و راهنمای تنظیم `WORKER_IMAGE` در `k8s_client.py` نمایش داده می‌شود.

---

## بخش ۲۳: مدل‌های دقیق `dispatcher_api.py` (Pydantic + FastAPI)

```python
app = FastAPI(title="edge-rl control-plane dispatcher")
_engine = None   # نمونه‌ی RealtimeEngine؛ قبل از start با bind_engine(engine) ست می‌شود

class RouteRequest(BaseModel):
    request_id: int
    service_id: int
    bts_lat: float
    bts_long: float

class ReportRequest(BaseModel):
    request_id: int
    service_id: int
    server_id: int
    success: bool
    response_time_sec: float

@app.post("/route")
async def route(req: RouteRequest):
    if _engine is None: return {"status": "ENGINE_NOT_READY"}
    return await _engine.route_request(req.request_id, req.service_id, req.bts_lat, req.bts_long)

@app.post("/report")
async def report(req: ReportRequest):
    # در عمل مسیر اصلی گزارش completion، صف Redis (edge:metrics:completions) است؛
    # این endpoint به‌عنوان مسیر جایگزین/دستی نگه داشته شده، fire-and-forget.
    if _engine is None: return {"status": "ENGINE_NOT_READY"}
    _engine.record_external_completion(req.request_id, req.service_id, req.server_id,
                                        req.success, req.response_time_sec)
    return {"status": "OK"}
```
اجرای مستقل (برای دیباگ دستی، خارج از `run.py --mode k8s`):
```
uvicorn k8s_adapter.dispatcher_api:app --host 0.0.0.0 --port 9000
```
راه‌اندازی کامل production (دو ترمینال، بدون `run.py`):
```
# ترمینال ۱ — کنترل‌پلین + موتور
uvicorn k8s_adapter.dispatcher_api:app --port 9000
# ترمینال ۲ — مولد ترافیک (جای BTS واقعی)
python3 -m k8s_adapter.bts_simulator
```
یا معادل خودکار همه‌چیز در یک پروسه:
```
python run.py --algorithm greedy --mode k8s --data test
```
(`serve_control_plane` خودش `uvicorn.Server` را به‌صورت داخلی با `asyncio.create_task` اجرا می‌کند و همزمان `engine.run()` را با تسک‌های پس‌زمینه صدا می‌زند؛ وقتی `engine._running=False` شد -- یعنی `_lifetime_watcher` بازه‌ی داده تمام شده تشخیص داد -- سرور uvicorn هم با `server.should_exit=True` به‌آرامی خاموش می‌شود.)

---

## بخش ۲۴: Dockerfile دقیق worker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${SERVICE_PORT:-8000} --workers 1"]
```
نکات:
- `--workers 1` (نه بیشتر) — محدودیت «۱ درخواست هم‌زمان» با `asyncio.Semaphore(1)` **داخل خودِ `app.py`** پیاده می‌شود، نه با فلگ concurrency سطح uvicorn (چون آن فلگ `/healthz` را هم مسدود می‌کند و پاد هرگز Ready نمی‌شود).
- پورت از env var `SERVICE_PORT` خوانده می‌شود (نه ثابت ۸۰۰۰)، چون با `host_network=True` چند سرویس مختلف ممکن است روی یک نود اجرا شوند و باید پورت اختصاصی خودشان را بگیرند (`worker_port(service_id) = 8000 + service_id`، بخش ۱۵.۹).
- یک ایمیج واحد برای هر ۱۵ سرویس؛ تفاوت فقط از طریق env vars (`EXEC_TIME_SEC, SERVICE_ID, SERVER_ID, SERVICE_PORT`) در Deployment.

ساخت/push دستی (بدون اسکریپت کمکی):
```bash
docker build -t <REGISTRY>/edge-worker:latest .
docker push <REGISTRY>/edge-worker:latest
```

---

## بخش ۲۵: کد مرده / مؤلفه‌ی تعریف‌شده اما متصل‌نشده — `common/network_coordinates.py`

یک پیاده‌سازی کامل شبکه‌ی مختصات **Vivaldi** (`VivaldiCoordinate`, `VivaldiNetwork`) وجود دارد که برای تخمین RTT بین BTS و سرور بدون نیاز به فاصله‌ی جغرافیایی مستقیم طراحی شده، اما **به هیچ الگوریتمی متصل نیست** — نه در مسیریابی VOILA (که ادعای اولیه‌اش بود) و نه جای دیگری. برای پیاده‌سازی وفادار به رفتار نهایی سیستم:

- این ماژول باید وجود داشته باشد (برای کامل بودن کدبیس) اما **استفاده‌ی عملیاتی ندارد**.
- در `algorithms/voila/voila_algorithm.py` یک متد `select_replica` مبتنی بر Vivaldi به‌صورت **رشته‌ی docstring غیرفعال** (نه تابع واقعی) نگه داشته می‌شود — یعنی هرگز `class VoilaAlgorithm` را override نمی‌کند؛ VOILA برای مسیریابی از همان `AlgorithmBase.select_replica` مشترک (بخش ۸) استفاده می‌کند.
- اگر پیاده‌سازی از صفر انجام می‌شود و هدف صرفاً بازتولید **رفتار نهایی مشاهده‌شده** است، می‌توان این ماژول را کاملاً حذف کرد بدون تأثیر روی رفتار سیستم؛ اگر هدف بازتولید **کدبیس کامل** (شامل کد مرده) است، باید دقیقاً به شکل زیر باشد:

```python
class VivaldiCoordinate:
    DIM = 2
    def __init__(self, rng):
        self.vec = rng.uniform(-1.0, 1.0, size=self.DIM) * 0.01
        self.height = float(rng.uniform(0.05, 0.3))
        self.error_estimate = 2.0

    def predicted_rtt_ms(self, other):
        euclid = norm(self.vec - other.vec)
        return euclid + self.height + other.height

    def update(self, other, observed_rtt_ms, rng, ce=0.25, cc=0.5):
        predicted = self.predicted_rtt_ms(other)
        error = abs(predicted - observed_rtt_ms)
        w = self.error_estimate / (self.error_estimate + other.error_estimate + 1e-9)
        rel_error = min(error / max(observed_rtt_ms, 1e-6), 1.0)
        alpha = ce * w
        self.error_estimate = alpha*rel_error*self.error_estimate + (1-alpha)*self.error_estimate
        self.error_estimate = max(self.error_estimate, 0.05)
        delta = cc * w
        direction = self.vec - other.vec
        norm_d = norm(direction)
        unit = (direction/norm_d) if norm_d >= 1e-6 else normalize(rng.uniform(-1,1,size=self.DIM))
        self.vec = self.vec + delta * (observed_rtt_ms - predicted) * unit

class VivaldiNetwork:
    def __init__(self, servers, base_latency_ms, k_ms_per_km, seed=0, bootstrap_rounds=20):
        # هر سرور یک VivaldiCoordinate تصادفی می‌گیرد؛ در bootstrap_rounds دور،
        # با RTT واقعی محاسبه‌شده از haversine بین هر جفت سرور کالیبره می‌شوند.
        # BTSها هم به‌صورت lazy (اولین بار که دیده می‌شوند) یک coordinate می‌گیرند
        # و با observe() آپدیت می‌شوند.
    def estimate_rtt_ms(self, bts_lat, bts_lon, server_id): ...
    def observe(self, bts_lat, bts_lon, server_id, true_rtt_ms): ...
    def observation_count(self, bts_lat, bts_lon) -> int: ...
```

---

## بخش ۲۶: فرمول‌های دقیق باقی‌مانده‌ی اسکریپت‌های تحلیل

### `analyze_scaleup_by_service.py`
```
LOOKAHEAD_TICKS = 15
DECISION_INTERVAL_SEC = 30.0
CANDIDATE_ACTION_KEYS = ["action","decision","scale_action"]
CANDIDATE_TIME_KEYS = ["sim_time_sec","time","sim_time"]
CANDIDATE_SERVICE_KEYS = ["service_id","svc","svc_id"]
CANDIDATE_APPLIED_KEYS = ["applied","was_applied","executed"]
```
فقط رکوردهای `event_type=="scale_decision"` با `action در {"SCALE_UP","SCALE_DOWN"}` بارگذاری می‌شوند. برای هر سرویس، لیست `sim_time_sec` هر SCALE_UP که `applied` بوده (پیش‌فرض `applied=True` اگر فیلد نبود) جمع می‌شود. خروجی: به ازای هر سرویس، `count`, `share` (درصد از کل SCALE_UPهای اعمال‌شده)، `avg_gap_sec` و `min_gap_sec` (فاصله‌ی زمانی بین SCALE_UPهای متوالی همان سرویس).

### `analyze_necessity_by_service.py`
برای هر رکورد `scale_decision`: `total_ticks[sid] += 1`؛ اگر `necessary_scale_up==True` بود `necessary_ticks[sid]+=1` و اگر همزمان `applied and decision=="SCALE_UP"` بود `applied_when_necessary[sid]+=1`؛ اگر `necessary_scale_up==False` بود و `applied and decision=="SCALE_UP"` بود `applied_when_not_necessary[sid]+=1`. خروجی هر سرویس: `necessity_rate = 100*necessary_ticks/total_ticks`، `good_apply_rate = 100*applied_when_necessary/necessary_ticks`، `bad_apply_rate = 100*applied_when_not_necessary/(total_ticks-necessary_ticks)`، `bad_share = 100*applied_when_not_necessary[sid]/sum(applied_when_not_necessary.values())`.

### `diagnose_violations_by_service.py`
برای هر سرویس شمارنده‌های `total_by_svc` (از `request_arrived`)، `completed_by_svc`/`rt_sum_by_svc`/`rt_max_by_svc` (از `request_completed`)، `rejected_by_svc` (از `request_rejected`، که هم‌زمان به‌عنوان نقض هم شمرده می‌شود) جمع می‌شوند. برای هر `request_completed`: اگر رکورد خودش فیلد `deadline_violated` داشت همان معیار است؛ وگرنه `response_time_sec > deadline[service_id]`. خروجی هر سرویس: `violation_rate = 100*violated/arrivals`، `violation_share = 100*violated/کل_نقض‌ها`، `traffic_share = 100*arrivals/کل_ورودی‌ها`.

### `analyze_decision_quality.py`
```
LOOKAHEAD_TICKS = 15
REJECTION_LOOKAHEAD_SEC = 90
FLAPPING_DWELL_SEC = 300
```
برای هر سرویس، رکوردهای `scale_decision` به ترتیب زمان مرتب می‌شوند. برای هر رکورد `applied==True`:
- اگر `decision=="SCALE_UP"`: اگر `necessary_scale_up==True` → `correct_now`. وگرنه پنجره‌ی `LOOKAHEAD_TICKS` تیک بعدی همان سرویس بررسی می‌شود: اگر یکی از آن‌ها `necessary_scale_up==True` شد یا در بازه‌ی `(t, window_end_time]` رد شدن واقعی رخ داد → `anticipatory`؛ وگرنه → `noise`.
- اگر `decision=="SCALE_DOWN"`: اگر `necessary_scale_down==True` → `correct_now`. وگرنه اگر در پنجره‌ی بعدی نیاز واقعی (`necessary_scale_up`) یا رد شدن رخ داد → `risky`؛ وگرنه → `harmless_early`.

تحلیل flapping: از رکوردهای `provision_decision` با `applied==True` و `action in {"TURN_ON","TURN_OFF"}`، به تفکیک هر سرور مرتب‌شده به ترتیب زمان. هر جفت متوالی (یک `TURN_ON` و اولین `TURN_OFF` بعدی روی همان سرور) یک «چرخه» با `dwell = off_time - on_time` است. چرخه‌های با `dwell < FLAPPING_DWELL_SEC` به‌عنوان `flapping` علامت می‌خورند.

---

## بخش ۲۷: جدول مرجع سریع همه‌ی ثابت‌های عددی (چک‌لیست پیاده‌سازی)

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
| `PPO_REWARD_WEIGHTS.w1_response_time` | 0.08 |
| `PPO_REWARD_WEIGHTS.w2_deadline` | 0.35 |
| `PPO_REWARD_WEIGHTS.w3_energy` | 0.20 |
| `PPO_REWARD_WEIGHTS.w4_load_balance` | 0.12 |
| `PPO_REWARD_WEIGHTS.w5_rejected` | 0.25 |
| `PPO_PENALTY_PER_ACTION` | 0.02 |
| `PPO_DEADLINE_FAIRNESS_ALPHA` | 0.7 |
| `NORM_RESPONSE_TIME_SEC` | 1.232 |
| `NORM_ENERGY_JOULE` | 4431.91 |
| `NORM_ARRIVAL_RATE` | 3.0 |
| `NORM_REJECTED_PER_TICK` (env `EOTCH_NORM_REJECTED_PER_TICK`) | 2.0 |
| VOILA `OCC_UP_THRESHOLD` | 0.65 |
| VOILA `OCC_DOWN_THRESHOLD` | 0.20 |
| VOILA `SCALE_DOWN_PATIENCE_TICKS` | 3 |
| VOILA `PROXIMITY_SUSTAIN_TICKS` | 2 |
| VOILA `PROXIMITY_PROTECTION_TICKS` | 5 |
| HPA `TARGET_UTILIZATION` | 0.70 |
| Greedy scale-up occ threshold | 0.7 |
| Greedy scale-down occ threshold | 0.1 |
| Greedy/HPA/VOILA `select_replica` near-pool radius | +5.0 km از نزدیک‌ترین |
| STATE_DIM | 152 (=10×6 + 15×6 + 2) |
| PPO training: `n_steps` | 2048 |
| PPO training: `batch_size` | 256 |
| PPO training: `gamma` | 0.99 |
| PPO training: `learning_rate` | 3e-4 |
| PPO training: `ent_coef` | 0.01 |
| PPO training: `net_arch` | pi=[256,256], vf=[256,256] |
| PPO training: `total_timesteps` (پیش‌فرض) | 3,000,000 |
| PPO training: `n_envs` | 8 |
| BC warm-start: `epochs` | 50 |
| BC warm-start: `lr` | 5e-5 |
| BC warm-start: `batch_size` | 64 |
| BC warm-start: `bc_max_ticks` | 10,000 |
| k8s `UTIL_SAMPLE_INTERVAL_SEC` | 5.0 |
| k8s `RESERVATION_SWEEP_INTERVAL_SEC` | 10.0 |
| k8s reservation TTL | `deadline_sec + 5` |
| k8s `drain_completion_queue` poll interval | 0.2 sec |
| k8s dispatcher port | 9000 |
| k8s worker port | `8000 + service_id` |
| Redis host:port (پیش‌فرض) | 192.168.1.30:6379 |
| local registry (پیش‌فرض) | 192.168.1.30:5000 |

این جدول باید در پیاده‌سازی نهایی بدون هیچ انحرافی رعایت شود — هرگونه تغییر در این اعداد رفتار عددی سیستم (نه ساختار آن) را عوض می‌کند و مقایسه با نتایج مرجع را نامعتبر می‌سازد.

---

**پایان سند.** این مشخصات (بخش‌های ۰ تا ۲۷) به‌همراه جدول ثابت‌های بخش ۲۷، همه‌ی دانشی است که برای بازسازی کامل و بدون‌ابهام پروژه‌ی `edge_rl` — شامل هر دو مسیر شبیه‌سازی و اجرای واقعی روی Kubernetes، هر چهار الگوریتم تصمیم‌گیری، و کل خط‌لوله‌ی آموزش/ارزیابی PPO — لازم است.
