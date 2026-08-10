# مدیریت پویای منابع محاسباتی لبه (Edge Resource Management)

## Greedy / VOILA / Kubernetes-HPA / PPO-DRL — شبیه‌سازی + اجرای واقعی روی Kubernetes

این سند، مرجع کامل و یکپارچه‌ی پروژه است: از تعریف منابع و مدل داده تا چرخه‌ی کامل یک درخواست، الگوریتم‌های تصمیم‌گیری، عامل PPO، و اجرای زنده روی یک خوشه‌ی Kubernetes واقعی. هرکس بخواهد این پروژه را از صفر بسازد، باید دقیقاً همین ترتیب و همین مقادیر را پیاده کند.

---

## فهرست

1. [ایده‌ی کلی سیستم](#۱-ایده‌ی-کلی-سیستم)
2. [منابع سیستم](#۲-منابع-سیستم)
3. [دیتاست درخواست‌ها](#۳-دیتاست-درخواست‌ها-شانگهای)
4. [مدل موجودیت‌ها](#۴-مدل-موجودیت‌ها-domain-model)
5. [مقداردهی اولیه‌ی سیستم](#۵-مقداردهی-اولیه‌ی-سیستم-t0)
6. [چرخه‌ی کامل یک درخواست](#۶-چرخه‌ی-کامل-یک-درخواست)
7. [مسیریابی/انتخاب نمونه](#۷-مسیریابیانتخاب-نمونه-instance-selection)
8. [Provisioning پویا سرور و Migration](#۸-provisioning-پویا-سرور-و-service-migration)
9. [Auto Scaling رپلیکا](#۹-auto-scaling-رپلیکا)
10. [متریک‌های نهایی و ممیزی تصمیم](#۱۰-متریک‌های-نهایی-و-ممیزی-تصمیم)
11. [پارامترهای پیکربندی](#۱۱-پارامترهای-پیکربندی-commonconfigpy)
12. [معماری کد و ساختار پوشه‌ها](#۱۲-معماری-کد-و-ساختار-پوشه‌ها)
13. [عامل PPO-DRL](#۱۳-عامل-ppo-drl)
14. [اجرای زنده روی Kubernetes](#۱۴-اجرای-زنده-روی-kubernetes)
15. [ابزارهای ارزیابی و کالیبراسیون](#۱۵-ابزارهای-ارزیابی-و-کالیبراسیون)
16. [نصب و وابستگی‌ها](#۱۶-نصب-و-وابستگی‌ها)
17. [راهنمای اجرا](#۱۷-راهنمای-اجرا)
18. [نقشه‌ی ساخت پروژه از صفر](#۱۸-نقشه‌ی-ساخت-پروژه-از-صفر)

---

## ۱. ایده‌ی کلی سیستم

۱۰ سرور لبه (edge server) در نقاط مختلف یک شهر قرار دارند و باید به درخواست‌های ۱۵ نوع سرویس مختلف که از ایستگاه‌های پایه‌ی موبایل (BTS) می‌رسند پاسخ دهند. سؤال اصلی: **چه زمانی سرور روشن/خاموش شود، چند نمونه (replica) از هر سرویس اجرا شود، و درخواست‌ها به کدام سرور هدایت شوند** — طوری که هم‌زمان تأخیر پاسخ‌دهی پایین بماند، سررسیدها (deadline) نقض نشوند، و مصرف انرژی هم کنترل‌شده باشد؟

چهار استراتژی متفاوت برای پاسخ به این سؤال پیاده‌سازی و روی داده‌ی واقعی ترافیک BTS شانگهای، با معیارهای یکسان، مقایسه می‌شوند:

| الگوریتم | فلسفه |
|---|---|
| **Greedy** | آستانه‌ساده و مکان‌آگاه: بر اساس اشغال صف و نرخ رد شدن تصمیم می‌گیرد؛ placement/migration بر مبنای نزدیک‌ترین سرور به مرکز سرورهای فعال. baseline پروژه است. |
| **VOILA** | مبتنی بر مقاله‌ی VOILA؛ placement و migration را بر اساس **مرکز ثقل تقاضای واقعی** هر سرویس (medoid موقعیت درخواست‌های اخیر) انتخاب می‌کند، نه صرفاً نزدیک‌ترین به مرکز سرورهای فعال. علاوه بر نقض ظرفیت، نقض نزدیکی جغرافیایی (proximity violation) را هم به‌عنوان سیگنال scale-up در نظر می‌گیرد و برای scale-down به یک «streak» چند-تیکی از وضعیت سالم نیاز دارد (ضد نوسان). |
| **HPA** | معادل الگوریتم Kubernetes Horizontal Pod Autoscaler: کاملاً location-unaware، فقط بر اساس نسبت اشغال صف نسبت به یک هدف ثابت (۷۰٪) تعداد replica مطلوب را محاسبه می‌کند. |
| **PPO-DRL** | یک عامل یادگیری تقویتی (Proximal Policy Optimization، با `MaskablePPO` از `sb3-contrib`) که هر سه نوع تصمیم را هم‌زمان از یک بردار حالت یاد می‌گیرد. آموزش با **warm-start از دموی Greedy** (Behavior Cloning) شروع می‌شود و سپس با RL روی پاداشی وزن‌دار از زمان پاسخ، نقض سررسید، انرژی، توازن بار و نرخ رد شدن fine-tune می‌شود. |

هر چهار الگوریتم از همان منطق مشترک برای **جای‌گذاری اولیه** و **مسیریابی درخواست** استفاده می‌کنند تا مقایسه منصفانه بماند؛ فقط تصمیم‌های scale/provision/placement/migration فرق دارند.

---

## ۲. منابع سیستم

### ۲.۱ سرورها — ۱۰ عدد (heterogeneous، بر پایه‌ی سخت‌افزار واقعی)

سه پروفایل سرور دقیقاً روی مدل واقعی **HPE ProLiant DL360 Gen10** با سه CPU متفاوت تعریف شده‌اند. عدد `p_idle`/`p_max` (وات) با درون‌یابی نسبت به داده‌ی اندازه‌گیری‌شده‌ی SPECpower_ssj2008 (res2019q2-00916، سرور با CPU Xeon Platinum 8280، TDP=205W) نسبت به TDP هر CPU به‌دست آمده — یعنی این اعداد ادعای اندازه‌گیری مستقیم ندارند، تخمین کالیبره‌شده‌اند.

```python
SERVER_PROFILES = {
    # HPE ProLiant DL360 Gen10, 1x Xeon Silver 4110 (8c, 2.10GHz, 85W TDP)
    "edge_small": {"n_cores": 8,  "mips_per_core": 2520, "capacity_mips": 20160,
                   "p_idle": 28, "p_max": 113},
    # HPE ProLiant DL360 Gen10, 1x Xeon Gold 5118 (12c, 2.30GHz, 105W TDP)
    "medium":     {"n_cores": 12, "mips_per_core": 2760, "capacity_mips": 33120,
                   "p_idle": 30, "p_max": 130},
    # HPE ProLiant DL360 Gen10, 1x Xeon Gold 6130 (16c, 2.10GHz, 125W TDP)
    "large":      {"n_cores": 16, "mips_per_core": 2520, "capacity_mips": 40320,
                   "p_idle": 33, "p_max": 148},
}
REFERENCE_MIPS_PER_CORE = SERVER_PROFILES["medium"]["mips_per_core"]  # 2760، مرجع سرعت اجرا
```

`capacity_mips = n_cores * mips_per_core`. واحد ظرفیت MIPS (میلیون دستور در ثانیه) کل هسته‌های سرور است — نه یک عدد دلبخواهی.

ده سرور با موقعیت جغرافیایی ثابت، روی BTSهای واقعی دیتاست شانگهای (مکان‌یابی از قبل با یک solver انجام شده؛ این پروژه دوباره location-solving نمی‌کند):

```python
SERVER_INFO = {
    1: {"bts_id": 1498, "lat": 31.37, "long": 121.25, "capacity_mips": 20160},   # edge_small
    2: {"bts_id": 777,  "lat": 31.31, "long": 121.51, "capacity_mips": 20160},   # edge_small
    3: {"bts_id": 530,  "lat": 31.10, "long": 121.18, "capacity_mips": 33120},   # medium
    4: {"bts_id": 121,  "lat": 31.25, "long": 121.37, "capacity_mips": 33120},   # medium
    5: {"bts_id": 292,  "lat": 31.10, "long": 121.36, "capacity_mips": 33120},   # medium
    6: {"bts_id": 1344, "lat": 31.04, "long": 121.74, "capacity_mips": 33120},   # medium
    7: {"bts_id": 182,  "lat": 31.17, "long": 121.57, "capacity_mips": 40320},   # large
    8: {"bts_id": 505,  "lat": 31.15, "long": 121.41, "capacity_mips": 40320},   # large
    9: {"bts_id": 419,  "lat": 31.20, "long": 121.43, "capacity_mips": 40320},   # large
    10: {"bts_id": 609, "lat": 31.16, "long": 121.49, "capacity_mips": 40320},   # large
}
```

پروفایل هر سرور از روی `capacity_mips` استخراج می‌شود.

قید سخت جای‌گذاری (در تمام تصمیمات placement/scaling هر چهار الگوریتم رعایت می‌شود؛ نقض آن غیرمجاز است):

```
sum(resource_mips[svc] for svc in services deployed on server) <= capacity_mips[server]
```

برای PPO از **`MaskablePPO`** (کتابخانه‌ی `sb3-contrib`) استفاده می‌شود که قبل از sample کردن اکشن، گزینه‌های نامعتبر را از توزیع احتمال حذف می‌کند.

### ۲.۲ سرویس‌ها — ۱۵ عدد (بار CPU بر پایه‌ی MIPS/MI)

`exec_time` یک عدد ثابت در دیتاست نیست؛ از تقسیم طول کار سرویس (`task_length_mi`، بر حسب میلیون دستور — Million Instructions) بر توان مؤثر رپلیکای میزبان محاسبه می‌شود. یعنی زمان اجرای یک سرویس روی سرور `edge_small` بیشتر از همان سرویس روی سرور `large` است — رفتار واقعی هتروژن.

```python
SERVICES_INFO: Dict[int, dict] = {
    1 : {"resource_mips": 4800, "task_length_mi": 55,     "queue_len": 1,  "deadline": 0.030, "memory": "32Mi"},   # 5QI=84  Intelligent Transport Systems (V2X پایه)
    2 : {"resource_mips": 4900, "task_length_mi": 110,    "queue_len": 2,  "deadline": 0.050, "memory": "48Mi"},   # 5QI=3   Real-Time Gaming / V2X Messages
    3 : {"resource_mips": 5000, "task_length_mi": 140,    "queue_len": 2,  "deadline": 0.060, "memory": "48Mi"},   # 5QI=69  Mission-Critical Delay-Sensitive Signalling
    4 : {"resource_mips": 5100, "task_length_mi": 190,    "queue_len": 3,  "deadline": 0.075, "memory": "56Mi"},   # 5QI=65  Mission-Critical PTT Voice
    5 : {"resource_mips": 5200, "task_length_mi": 260,    "queue_len": 3,  "deadline": 0.100, "memory": "64Mi"},   # 5QI=7   Voice / Live Video / Interactive Gaming
    6 : {"resource_mips": 5300, "task_length_mi": 310,    "queue_len": 3,  "deadline": 0.100, "memory": "72Mi"},   # 5QI=66  Non-Mission-Critical PTT Voice
    7 : {"resource_mips": 5400, "task_length_mi": 420,    "queue_len": 4,  "deadline": 0.150, "memory": "96Mi"},   # 5QI=2   Conversational Video (Live Streaming)
    8 : {"resource_mips": 5400, "task_length_mi": 570,    "queue_len": 5,  "deadline": 0.200, "memory": "128Mi"},  # 5QI=70  Mission-Critical Data
    9 : {"resource_mips": 5500, "task_length_mi": 880,    "queue_len": 6,  "deadline": 0.300, "memory": "160Mi"},  # 5QI=8   Video Buffered Streaming (TCP-based)
    10: {"resource_mips": 5600, "task_length_mi": 1050,   "queue_len": 6,  "deadline": 0.300, "memory": "176Mi"},  # 5QI=9   Video Buffered Streaming / Default Bearer
    11: {"resource_mips": 5400, "task_length_mi": 3500,   "queue_len": 8,  "deadline": 1.0,   "memory": "192Mi"},  # فراتر از 5QI رسمی — IoT Batch Telemetry Aggregation
    12: {"resource_mips": 5400, "task_length_mi": 7000,   "queue_len": 10, "deadline": 2.0,   "memory": "256Mi"},  # فراتر از 5QI رسمی — Video Analytics / Object Detection Offload
    13: {"resource_mips": 5200, "task_length_mi": 10000,  "queue_len": 12, "deadline": 3.0,   "memory": "320Mi"},  # فراتر از 5QI رسمی — ML Inference Batch
    14: {"resource_mips": 5200, "task_length_mi": 50500,  "queue_len": 15, "deadline": 15.0,  "memory": "512Mi"},  # فراتر از 5QI رسمی — Large-Scale Data Aggregation
    15: {"resource_mips": 5800, "task_length_mi": 151000, "queue_len": 20, "deadline": 40.0,  "memory": "768Mi"},  # فراتر از 5QI رسمی — Heavy Batch Analytics / Retraining Offload
}
```

فرمول محاسبه‌ی زمان اجرا:

```python
def compute_exec_time_sec(service_id, host_mips_per_core):
    svc = SERVICES_INFO[service_id]
    speed_factor = host_mips_per_core / REFERENCE_MIPS_PER_CORE   # نسبت سرعت هسته‌ی میزبان به مرجع (medium)
    effective_mips = svc["resource_mips"] * speed_factor
    return svc["task_length_mi"] / effective_mips
```

هر بار که رپلیکای یک سرویس روی یک سرور جای‌گذاری می‌شود، `exec_time` مخصوص همان (سرویس, پروفایل سرور) محاسبه و در شیء `Replica` ذخیره می‌شود؛ زمان اجرا مقداری ثابتِ per-instance است، نه یک تابع تصادفی.

قید صحت‌سنجی هنگام بارگذاری config: بزرگ‌ترین `resource_mips` بین سرویس‌ها باید از کوچک‌ترین `capacity_mips` بین پروفایل‌ها کمتر باشد (وگرنه هیچ‌وقت روی هیچ سروری جا نمی‌شود).

**Deadline** بر حسب ثانیه است، از ۰.۰۱ ثانیه برای سرویس‌های سبک تا ۴۰ ثانیه برای سنگین‌ترین — این باعث می‌شود `deadline_violation` یک سیگنال معنادار و حساس باشد.

قواعد ثابت:

- حداکثر ۱ رپلیکا از یک سرویس روی هر سرور.
- هر رپلیکا یک صف FIFO واقعی با ظرفیت `queue_len` دارد (شبیه‌سازی صف واقعی M/D/1/K، نه تخمین آماری) — پیاده‌سازی با `deque` از زمان‌های خروج (`departures`)؛ هر بار admit، اشغال واقعی صف بر اساس مقایسه‌ی `now` با صف خروج‌های ثبت‌شده محاسبه می‌شود.
- هر رپلیکا هم‌زمان فقط ۱ درخواست پردازش می‌کند (server-side single-threaded per replica)؛ یک سرور می‌تواند هم‌زمان چند سرویس/رپلیکای مختلف را اجرا کند، هرکدام صف و پردازش مستقل خودشان را دارند.

---

## ۳. دیتاست درخواست‌ها (شانگهای)

ستون‌های استفاده‌شده: `id, BTSID, Lat, Long, ServiceID, startSec`. `Lat, Long` مختصات BTS مبدأ درخواست است (نه سرور). `startSec` ثانیه‌ی ورود درخواست از ابتدای همان روز (۰ تا ~۸۶۳۹۹)، هر روز از صفر شروع می‌شود.

- دیتای تست: `Data4.csv` (شنبه‌ی هفته‌ی ۴)، مستقل از train.
- دیتای آموزش: `Data1.csv, Data2.csv, Data3.csv` (شنبه‌های هفته‌ی ۱ تا ۳)، با فرمت ستونی یکسان.

فیلترهای پیش‌پردازش (`data/loader.py`):
- محدوده‌ی جغرافیایی معتبر: `LAT_MIN=30.5, LAT_MAX=31.7, LON_MIN=120.7, LON_MAX=122.0` — رکوردهای خارج از این باکس (نویز/خطای داده) قبل از شروع شبیه‌سازی حذف می‌شوند.
- فقط `ServiceID` های موجود در `SERVICES_INFO` (۱ تا ۱۵) نگه داشته می‌شوند.

اتصال زمانی برای ساخت یک تایم‌لاین پیوسته:

```
global_start_sec = day_index * 86400 + startSec
```

که `day_index ∈ {0, 1, 2}` برای ۳ فایل train (یک تایم‌لاین پیوسته‌ی سه‌روزه) و `day_index=0` برای فایل test (یک‌روزه، مستقل). این پیش‌پردازش قبل از شروع شبیه‌سازی انجام می‌شود و خروجی یک دیتافریم واحد با ستون `global_start_sec` مرتب‌شده است:

```python
def load_train():   # Data1.csv, Data2.csv, Data3.csv -> تایم‌لاین پیوسته‌ی ۳روزه
def load_test():    # Data4.csv -> تایم‌لاین یک‌روزه، مستقل از train
```

---

## ۴. مدل موجودیت‌ها (Domain Model)

### ۴.۱ Server

```python
@dataclass
class Server:
    id: int
    profile: str              # edge_small / medium / large
    lat: float
    long: float
    capacity: int              # MIPS
    p_idle: float
    p_max: float
    state: ServerState = OFF   # OFF | BOOTING | ACTIVE | DRAINING
    hosted_replicas: Dict[int, Replica]
    boot_started_at, drain_started_at: Optional[float]
    last_transition_time: float          # برای cooldown/anti-flapping
    cumulative_energy_joule: float
    cumulative_busy_cpu_seconds: float   # انتگرال زمانی busy_mips*ثانیه — پایه‌ی utilization دقیق
    num_boots, num_shutdowns: int
```

متدهای کلیدی:
- `used_cpu()` / `free_capacity()` / `can_host(service_id, cpu_demand)` — بر پایه‌ی MIPS.
- `in_cooldown(now, cooldown_sec)` — برای anti-flapping.
- `instantaneous_utilization(now)` — مجموع `resource_mips` رپلیکاهایی که **در همین لحظه واقعاً busy هستند** (`state in (READY, DRAINING)` و `not r.is_idle(now)`) تقسیم بر ظرفیت. فقط رپلیکاهای در حال پردازش شمرده می‌شوند، نه صرفاً deploy‌شده.
- `instantaneous_power_w(now)` — `power(t) = p_idle + (p_max - p_idle) * utilization(t)` برای ACTIVE/DRAINING؛ `p_idle` ثابت برای BOOTING؛ صفر برای OFF.

### ۴.۲ Replica

```python
@dataclass
class Replica:
    service_id, server_id: int
    queue_len: int
    exec_time: float                 # مقدار محاسبه‌شده مخصوص (سرویس, پروفایل سرور میزبان)
    state: ReplicaState = STARTING   # STARTING | READY | DRAINING | TERMINATED
    created_at: float
    ready_since: Optional[float]
    drain_started_at: Optional[float]
    available_at: float = 0.0        # زمانی که رپلیکا دوباره آزاد می‌شود
    departures: deque                # زمان‌های خروج درخواست‌های در صف/در حال پردازش
```

منطق صف واقعی:

```python
def queue_occupancy(now):
    while departures and departures[0] <= now: departures.popleft()
    return len(departures)

def try_admit(arrival_time, cold_start_extra=0.0):
    if queue_occupancy(arrival_time) >= queue_len: return None   # رد به‌دلیل صف پر
    start = max(arrival_time, available_at)
    finish = start + exec_time + cold_start_extra
    available_at = finish
    departures.append(finish)
    return {queue_enter_time, service_start_time=start, service_end_time=finish, wait_time_sec=start-arrival_time}
```

این معادل دقیق یک صف FIFO تک‌سرور (M/D/1/K با K=queue_len) است، بدون نیاز به شبیه‌ساز event-driven جداگانه برای هر رپلیکا.

### ۴.۳ Request

```python
@dataclass
class Request:
    id, bts_lat, bts_long, service_id: ...
    arrival_time: float
    assigned_server_id: Optional[int]
    queue_enter_time, service_start_time, service_end_time: Optional[float]
    network_delay_ms: float = 0.0        # یک‌طرفه (برای گزارش)
    routing_delay_sec: float = 0.0       # زمان رفت‌وبرگشت BTS<->دیسپچر
    wait_time_sec, response_time_sec: float
    deadline_violated: bool
    status: PENDING | COMPLETED | REJECTED_QUEUE_FULL | REJECTED_NO_REPLICA
```

### ۴.۴ ماشین حالت سرور و رپلیکا

```
OFF --(provision trigger)--> BOOTING --(boot_delay_sec)--> ACTIVE
ACTIVE --(drain trigger)--> DRAINING --(graceful drain کامل)--> OFF

(none) --(pod create)--> STARTING --(pod_startup_delay_sec)--> READY
READY --(scale-down/migrate/drain)--> DRAINING --(graceful_termination_delay_sec)--> TERMINATED
```

**cold-start:** جریمه‌ی cold-start به‌صورت پویا محاسبه می‌شود، نه یک عدد ثابت یکسان برای همه‌ی سرویس‌ها:

```python
COLD_START_PENALTY_FRACTION = 0.20
COLD_START_PENALTY_CAP_SEC  = 0.500
COLD_START_WINDOW_SEC       = 10.0

def compute_cold_start_penalty_sec(service_id, host_mips_per_core):
    dl = SERVICES_INFO[service_id]["deadline"]
    et = compute_exec_time_sec(service_id, host_mips_per_core)
    penalty = min(dl * COLD_START_PENALTY_FRACTION, et * 0.50)
    return min(penalty, COLD_START_PENALTY_CAP_SEC)
```

یعنی جریمه‌ی cold-start متناسب با «حساسیت» سرویس (نسبت به deadline خودش و نصف زمان اجرای خودش) است. این جریمه فقط زمانی اعمال می‌شود که درخواست ظرف `COLD_START_WINDOW_SEC=10` ثانیه‌ی بعد از READY شدن رپلیکا به آن برسد.

---

## ۵. مقداردهی اولیه‌ی سیستم (t=0)

### ۵.۱ استراتژی پایه (Greedy/VOILA/HPA): پوشش حریصانه (Set-Cover)

الگوریتم پوشش حریصانه (Set-Cover-Style، مشابه Procedure 3 مقاله‌ی VOILA): در هر تکرار، سروری انتخاب می‌شود که بیشترین BTS فعال بدون‌پوشش را (طبق آستانه‌ی `L0_MS` روی `network_delay_ms` یک‌طرفه) پوشش دهد، تا پوشش کامل یا اتمام سرورها. اگر پوشش اولیه کافی نبود (کل ظرفیت انتخاب‌شده کمتر از مجموع `resource_mips` هر ۱۵ سرویس)، سرورهای باقی‌مانده به ترتیب نزدیکی به مجموعه‌ی انتخاب‌شده اضافه می‌شوند تا ظرفیت کافی شود.

سپس برای هر ۱۵ سرویس، دقیقاً ۱ رپلیکا روی نزدیک‌ترین سرور فعال با ظرفیت کافی (که هنوز آن سرویس را ندارد) مستقر می‌شود.

### ۵.۲ استراتژی PPO: حل چندهدفه با ILP (اختیاری)

به‌جای پوشش حریصانه‌ی صرف، PPO (به‌طور پیش‌فرض، با پرچم `use_solver_placement=True`) از یک solver ILP واقعی (کتابخانه‌ی `pulp`، حل‌کننده‌ی CBC) برای انتخاب بهینه‌ی مجموعه‌ی سرورهای اولیه استفاده می‌کند، با تابع هدف سه‌جزئی وزن‌دار:

```
minimize:  w_count   * (تعداد سرور روشن / کل سرورها)
         + w_energy  * (مجموع p_idle سرورهای انتخاب‌شده / مجموع کل p_idle)
         + w_distance* (مجموع وزنی فاصله‌ی هر نقطه‌ی تقاضا تا نزدیک‌ترین سرور پوشش‌دهنده‌اش / نرمال‌ساز)
```

با قیود:
- هر نقطه‌ی تقاضای پوشش‌پذیر (در محدوده‌ی `L0_MS`) دقیقاً به یک سرور فعال منصوب می‌شود.
- منصوب‌کردن فقط به سرور *فعال*‌شده مجاز است.
- مجموع ظرفیت سرورهای انتخاب‌شده ≥ مجموع `resource_mips` کل ۱۵ سرویس.

نقاط تقاضا از تایم‌لاین سه‌روزه‌ی train (`aggregate_training_demand`) استخراج می‌شوند — یعنی این solver از **آمار واقعی تقاضای BTSها روی داده‌ی آموزش** استفاده می‌کند، نه فقط یک پنجره‌ی زمانی کوچک ابتدای اجرا. وزن‌های `w_count/w_energy/w_distance` قابل تنظیم از خط فرمان هستند. اگر solver جواب پیدا نکند یا `pulp` نصب نباشد، به همان پوشش حریصانه‌ی مشترک سقوط می‌کند (fallback امن).

---

## ۶. چرخه‌ی کامل یک درخواست

یک مرحله‌ی مسیریابی صریح از طریق دیسپچر بین ورود درخواست و رسیدن به رپلیکا وجود دارد که هزینه‌ی زمانی واقعی خودش را دارد و در `response_time` لحاظ می‌شود.

### ۶.۱ مرحله‌ی ورود و مسیریابی (Dispatch)

1. رویداد ورود در `global_start_sec` (از دیتاست) — درخواست با `service_id`، مختصات BTS مبدأ (`bts_lat`, `bts_long`) می‌رسد.
2. **دیسپچر مرکزی** (در شبیه‌سازی: یک نقطه‌ی مجازی روی میانگین جغرافیایی همه‌ی ۱۰ سرور؛ در اجرای واقعی: سرویس FastAPI روی ماشین `192.168.1.30:9000`) باید ابتدا آدرس این درخواست را دریافت کند. این یک **round-trip واقعی** بین BTS و دیسپچر است:

```python
one_way_dispatch_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM * distance_km(bts, dispatcher) + DISPATCH_OVERHEAD_MS
routing_delay_sec = 2 * one_way_dispatch_delay_ms / 1000.0     # رفت + برگشت پاسخ دیسپچر
```

در شبیه‌سازی، این تأخیر به‌صورت یک رویداد جداگانه (`REQUEST_ROUTED`) در `routing_delay_sec` ثانیه بعد از ورود، زمان‌بندی می‌شود؛ در اجرای k8s واقعی، این همان زمانی است که BTS منتظر پاسخ HTTP `POST /route` می‌ماند.

3. وقتی رویداد `REQUEST_ROUTED` رخ می‌دهد، **Instance Selector (Routing)** اجرا می‌شود (جزئیات کامل در بخش ۷).
4. اگر پذیرفته شد: درخواست وارد صف FIFO رپلیکا می‌شود (`try_admit`).
5. محاسبه‌ی نهایی:

```python
distance_km = haversine(bts, server)
network_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM * distance_km        # یک‌طرفه BTS<->سرور
wait_time_sec = service_start_time - queue_enter_time
cold_start_extra = compute_cold_start_penalty_sec(...) اگر شرایط برقرار باشد وگرنه ۰

response_time_sec = (
    routing_delay_sec                              # رفت‌وبرگشت BTS<->دیسپچر (مرحله‌ی مسیریابی)
    + 2 * network_delay_ms / 1000.0                 # رفت‌وبرگشت BTS<->سرور (خودِ داده)
    + wait_time_sec
    + (service_end_time - service_start_time)       # = exec_time + cold_start_extra
)
deadline_violated = response_time_sec > deadline[service_id]
```

`response_time_sec` شامل **دو** رفت‌وبرگشت جداست: یکی برای مرحله‌ی مسیریابی (BTS↔دیسپچر) و یکی برای مرحله‌ی داده (BTS↔سرور). این با معماری واقعی هم‌خوان است چون در اجرای زنده، BTS واقعاً دو فراخوانی HTTP جدا می‌زند (اول `/route` به دیسپچر، بعد `/process` مستقیم به IP:Port پاد).

برای متریک‌های گزارشی (`avg/p95/p99 network_delay_ms`)، فقط مقدار **یک‌طرفه**‌ی `network_delay_ms` (بدون ضرب در ۲ و بدون احتساب `routing_delay_sec`) ثبت می‌شود؛ ضرب‌در‌۲ فقط داخل فرمول `response_time_sec` اعمال می‌شود.

### ۶.۲ مدل تأخیر شبکه

```
distance_km = haversine(lat1, lon1, lat2, lon2)
network_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM * distance_km
```

ثابت‌ها:

```python
BASE_LATENCY_MS = 2.0        # سربار پردازش/route حداقلی
K_MS_PER_KM = 0.02           # ضریب جغرافیایی (کالیبره‌شده، نه صرفاً سرعت نور در فیبر)
DISPATCH_OVERHEAD_MS = 0.3   # سربار پردازش داخلی خودِ دیسپچر
L0_MS = 20.0                 # آستانه‌ی پوشش اولیه (initial placement / ILP solver)
PROXIMITY_L0_MS = 7.0        # آستانه‌ی RTT برای «نقض مجاورت» (VOILA proximity signal)
```

**نکته‌ی حیاتی:** آستانه‌ی `proximity_violation` باید روی **RTT** (یعنی `2 * network_delay_ms`) سنجیده شود، نه `network_delay_ms` یک‌طرفه؛ چون با محدوده‌ی جغرافیایی دیتاست (حداکثر فاصله‌ی ممکن BTS↔سرور ≈ ۱۸۲ کیلومتر) بیشینه‌ی `network_delay_ms` یک‌طرفه فقط ~۵.۶ میلی‌ثانیه است که همیشه کمتر از `PROXIMITY_L0_MS=7.0` می‌ماند و این شرط هیچ‌وقت True نمی‌شود. این سیگنال مبنای اصلی رفتار geo-aware الگوریتم VOILA است، پس باید حتماً روی RTT سنجیده شود:

```python
if 2 * network_delay_ms >= PROXIMITY_L0_MS:
    proximity_violated += 1
```

---

## ۷. مسیریابی/انتخاب نمونه (Instance Selection)

معیار پیش‌فرض (همه‌ی الگوریتم‌ها): فاصله‌ی جغرافیایی + وضعیت صف.

- الف) لیست رپلیکاهای READY آن سرویس گرفته می‌شود.
- ب) فاصله‌ی جغرافیایی BTS مبدأ تا سرور میزبان هر رپلیکا (haversine) محاسبه می‌شود.
- ج) یک **استخر «تقریباً هم‌فاصله»** ساخته می‌شود: همه‌ی رپلیکاهایی که فاصله‌شان حداکثر ۵ کیلومتر بیشتر از نزدیک‌ترین رپلیکاست (`min_dist + 5.0`)؛ در این استخر، رپلیکایی با **کم‌ترین نسبت اشغال صف** (`queue_occupancy / queue_len`) انتخاب می‌شود — یعنی وقتی چند سرور تقریباً هم‌فاصله‌اند، تصمیم نهایی بر پایه‌ی **ترافیک/بار لحظه‌ای** آن‌هاست، نه صرفاً فاصله‌ی خام.
- د) اگر در استخر نزدیک هیچ‌کدام صف خالی نداشتند، به ترتیب فاصله (نه فقط استخر نزدیک) ادامه می‌دهد تا اولین رپلیکای پذیرا پیدا شود.
- ه) اگر همه‌ی رپلیکاهای READY آن سرویس صف پر دارند → `REJECTED_QUEUE_FULL`.
- و) اگر اصلاً رپلیکای READY برای آن سرویس وجود ندارد → `REJECTED_NO_REPLICA`.

**قابلیت اختیاری برای PPO** (`latency_aware_routing=True`، پرچم CLI `--latency-aware-routing`): به‌جای فاصله‌ی خام، رپلیکایی انتخاب می‌شود که **کمترین تخمین کل تأخیر** (RTT شبکه + انتظار تخمینی صف + exec_time خودِ رپلیکا) را دارد:

```python
est_total_latency = (2 * network_delay_ms/1000) + max(0, replica.available_at - now) + replica.exec_time
```

این حالت پیش‌فرض نیست (معیار رسمی routing فاصله‌ی جغرافیایی است، نه latency واقعی اندازه‌گیری‌شده مثل مقاله‌ی اصلی VOILA که Vivaldi/Serf دارد)، اما به‌عنوان یک گزینه‌ی قابل‌مقایسه در ارزیابی نگه داشته شده است.

**یادداشت — `common/network_coordinates.py` (Vivaldi، غیرفعال):** یک پیاده‌سازی کامل مختصات شبکه‌ی Vivaldi برای تخمین RTT بدون اندازه‌گیری مستقیم وجود دارد (برای شبیه‌سازی سناریویی که VOILA واقعاً روی RTT اندازه‌گیری‌شده کار می‌کند)، اما چون معیار رسمی routing «فاصله‌ی جغرافیایی» تعریف شده، این مسیر در کد VOILA کامنت است و **استفاده نمی‌شود** — فقط برای مرجع/توسعه‌ی آینده نگه داشته شده.

---

## ۸. Provisioning پویا سرور و Service Migration

### ۸.۱ محاسبه‌ی utilization: پنجره‌ای دقیق

برای **تصمیم‌گیری** (نه برای محاسبه‌ی توان لحظه‌ای انرژی) از میانگین واقعی روی کل پنجره‌ی بین دو تیک تصمیم استفاده می‌شود — دقیق‌تر و بدون نویز لحظه‌ای:

```python
# در هر تیک تصمیم (هر DECISION_INTERVAL_SEC):
avg_util[server] = (cumulative_busy_cpu_seconds_now - cumulative_busy_cpu_seconds_at_window_start)
                    / (capacity * window_elapsed_sec)
```

`cumulative_busy_cpu_seconds` هر سرور هر بار که موتور زمان جلو می‌رود به‌روزرسانی می‌شود (`util_at_last * capacity * dt`) — یعنی این یک انتگرال‌گیری زمانی دقیق است، نه نمونه‌برداری گسسته. برای محاسبه‌ی توان لحظه‌ای انرژی از `instantaneous_utilization(now)` همان لحظه استفاده می‌شود — این دو مقصود عمداً جدا نگه داشته شده‌اند.

### ۸.۲ آستانه‌های provisioning با sustain-tracking

```python
UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_HIGH_SEC = 30.0     # مدت تداوم overload قبل از توجیه TURN_ON
SUSTAIN_LOW_SEC = 60.0
COOLDOWN_SEC = 60.0
MIN_ACTIVE_DURATION_SEC = 300.0                 # حداقل عمر یک سرور ACTIVE قبل از این‌که بتواند خاموش شود
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0   # حداقل عمر رپلیکا قبل از این‌که بتواند SCALE_DOWN شود
```

برای هر سرور، دو ردیاب جداگانه‌ی «از چه زمانی زیر/بالای آستانه‌ام» نگه‌داری می‌شود (`_low_util_since`, `_high_util_since`)؛ فقط وقتی تداوم به `SUSTAIN_HIGH_SEC`/`SUSTAIN_LOW_SEC` برسد، سیگنال «نیاز واقعی» فعال می‌شود — این از نوسان لحظه‌ای (flapping ناشی از یک spike کوتاه) جلوگیری می‌کند.

انتخاب پروفایل سرور خاموش برای روشن‌کردن (heterogeneity-aware): مجموع ظرفیت سرورهای overload شده (یا سرور با بیشترین utilization) تعیین می‌کند پروفایل مطلوب چیست (`large` اگر مجموع ≥ ظرفیت large، `medium` اگر ≥ ظرفیت medium، وگرنه `edge_small`)؛ بین سرورهای آن پروفایل، نزدیک‌ترین از نظر جغرافیایی انتخاب می‌شود؛ اگر هیچ سروری از پروفایل مطلوب موجود نبود، fallback به کل سرورهای خاموش.

Cooldown: بعد از هر boot/drain روی یک سرور خاص، حداقل `COOLDOWN_SEC` قبل از رویداد معکوس روی همان سرور اعمال نمی‌شود.

**رفع اضطراری:** اگر هنگام drain یک سرور، migration رپلیکای تک‌نسخه‌ای به هیچ سرور ACTIVE مناسبی ممکن نبود، به‌جای شکست بی‌صدا، یک boot اضطراری روی نزدیک‌ترین سرور خاموش صدا زده می‌شود (provisioning اضطراری) و drain تا تکمیل موفق migration به تعویق می‌افتد (لاگ می‌شود و بعداً دوباره تلاش می‌شود).

### ۸.۳ Service Migration هنگام DRAIN

وقتی سروری وارد DRAINING می‌شود:

- برای هر رپلیکای روی آن سرور که **تنها رپلیکای آن سرویس در کل سیستم** است: نزدیک‌ترین سرور ACTIVE دیگر با ظرفیت آزاد کافی و بدون آن سرویس پیدا می‌شود؛ رپلیکای جدید آنجا STARTING می‌شود (`pod_startup_delay` طی می‌شود)، و **فقط پس از READY شدن** رپلیکای جدید، رپلیکای قدیم به DRAINING می‌رود (zero-downtime، تا از قطع سرویس جلوگیری شود). اگر هیچ سرور ACTIVE مناسبی پیدا نشد، یک سرور OFF جدید Boot می‌شود (Provisioning اضطراری) و migration به محض ACTIVE شدنش انجام می‌شود.
- برای رپلیکاهای سرویس‌های چندرپلیکایی: فقط تگ drain می‌خورند (از لیست کاندید Router خارج می‌شوند)، صف موجود تخلیه می‌شود، سپس TERMINATED؛ نیازی به جایگزینی روی سرور دیگر نیست چون رپلیکای دیگر همان سرویس در سیستم فعال است.

---

## ۹. Auto Scaling رپلیکا

رابط مشترک:

```python
scale_decision(service_id, current_metrics) -> {SCALE_UP, SCALE_DOWN, NO_CHANGE}
```

معیارهای ورودی: نسبت اشغال صف (`avg_queue_occupancy / queue_len`)، نرخ رد به‌دلیل صف پر (`rejection_rate`)، نرخ نقض deadline اخیر، و (فقط VOILA) `proximity_violation_rate`. Cooldown مشابه بخش ۸ برای هر `service_id` اعمال می‌شود.

**محافظ SCALE_DOWN:** فقط رپلیکاهایی که حداقل `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC` عمر دارند («بالغ» — mature) کاندید حذف‌شدن هستند؛ این از حذف رپلیکایی که همین الان (برای پاسخ به یک spike) بالا آمده و هنوز فرصت اثبات لازم‌بودنش را نداشته، جلوگیری می‌کند.

### تفاوت سیاست هر الگوریتم

- **Greedy:** `occ_ratio > 0.7` یا `rejection_rate > 0` → `SCALE_UP`؛ `occ_ratio < 0.1` و بیش از ۱ رپلیکا آماده → `SCALE_DOWN`. Placement: نزدیک‌ترین سرور به مرکز‌ثقل سرورهای فعال با ظرفیت کافی.
- **K8s-HPA:** فرمول استاندارد HPA: `desired_replicas = ceil(current_replicas * (current_utilization / TARGET_UTILIZATION))` با `TARGET_UTILIZATION=0.70`؛ اگر رد شدن داشتیم، حداقل یک رپلیکا بیشتر از فعلی. Placement: سروری با بیشترین ظرفیت آزاد (`max free_capacity`)، بدون در نظر گرفتن جغرافیا — دقیقاً رفتار HPA واقعی که geo-aware نیست.
- **VOILA:** علاوه‌بر آستانه‌ی اشغال صف (`OCC_UP_THRESHOLD=0.65`, `OCC_DOWN_THRESHOLD=0.20`)، سیگنال دوم **proximity_violation_rate** دارد: اگر ظرفیت مشکلی ندارد ولی RTT بیش از حد است (نقض مجاورت)، بعد از `PROXIMITY_SUSTAIN_TICKS=2` تیک متوالی نقض، `SCALE_UP` جغرافیایی صادر می‌شود و بعد از آن `PROXIMITY_PROTECTION_TICKS=5` تیک، سرویس از SCALE_DOWN محافظت می‌شود (تا اثر رپلیکای جدید سنجیده شود). Placement/migration/scale-down-victim همگی بر پایه‌ی **مدوید (medoid)** موقعیت اخیر درخواست‌های همان سرویس هستند (`demand_centroid` — نقطه‌ی واقعی از میان نقاط اخیر که مجموع فاصله‌اش تا بقیه کمینه است، نه صرفاً میانگین حسابی lat/long).
- **PPO:** تصمیم از مدل آموزش‌دیده می‌آید (بخش ۱۳).

---

## ۱۰. متریک‌های نهایی و ممیزی تصمیم

خروجی هر اجرا یک دیکشنری/JSON با این فیلدها:

```
avg_response_time_sec, p95_response_time_sec, p99_response_time_sec,
deadline_violations, deadline_violation_rate_pct,
cumulative_energy_joule,
avg_distance_km,
avg_load_balance_cv,
avg_network_delay_ms, p95_network_delay_ms, p99_network_delay_ms,

num_server_boots, num_server_shutdowns,
num_pod_creates, num_pod_deletes,
num_requests_rejected_queue_full, num_requests_rejected_no_replica,
avg_active_servers,
num_scale_up, num_scale_down, num_turn_on, num_turn_off,
decision_correctness,
total_requests, completed_requests
```

### ممیزی درستی تصمیم‌ها (`decision_correctness`)

هر الگوریتم باید گزارش بدهد چند بار SCALE_UP/DOWN/TURN_ON/OFF زده و چند تا واقعاً لازم بود. این به‌صورت یک لایه‌ی ممیزی مستقل در موتور پیاده شده: هر بار که موتور یک تصمیم (SCALE_UP/DOWN/TURN_ON/TURN_OFF) را از الگوریتم می‌گیرد، **بی‌ربط به این‌که خودِ الگوریتم چه فکری می‌کند**، با یک معیار عینی جداگانه چک می‌کند که آیا آن اکشن واقعاً لازم بود:

- `necessary_up`: `occ_ratio > 0.7` یا `rejection_rate > 0` (آستانه‌ی ممیزی، مستقل از آستانه‌ی داخلی الگوریتم).
- `necessary_down`: `occ_ratio < 0.2` و بیش از ۱ رپلیکای آماده.
- `necessary_turn_on`: overload پایدار *یا* سرویسی «capacity-starved» (نیاز واقعی به ظرفیت ولی هیچ سرور فعال/در حال بوت‌شدنی جا ندارد).
- `necessary_turn_off`: utilization سرور زیر آستانه‌ی پایین.

خروجی برای هر نوع تصمیم:

```json
{
  "SCALE_UP": {"correct": N, "incorrect": N, "missed_opportunities": N, "correctness_rate_pct": X}
}
```

`missed_opportunities` یعنی چند بار یک تصمیم *لازم بود* ولی الگوریتم آن را نگرفت (یا cooldown/محدودیت جلوی اجرایش را گرفت). این معیار امکان مقایسه‌ی کیفیت تصمیم‌گیری هر الگوریتم را (نه فقط تعداد خام اکشن‌ها) فراهم می‌کند.

**ابزار مکمل — `analyze_decision_quality.py`:** روی فایل لاگ JSONL هر الگوریتم دو نوع تحلیل انجام می‌دهد:
1. طبقه‌بندی SCALE_UP/SCALE_DOWN «غیرضروری طبق ممیزی لحظه‌ای» به «پیش‌بینانه/موجّه» در برابر «نویز واقعی» (با نگاه به چند تیک بعدی، آیا نیاز واقعی رخ داد یا نه).
2. تحلیل نوسان (flapping) سرور: چرخه‌های `TURN_ON → TURN_OFF` هر سرور را می‌سنجد و آن‌هایی که مدت فعالیت‌شان (`dwell`) کمتر از یک آستانه (پیش‌فرض ۳۰۰ ثانیه) بوده را flapping علامت می‌زند.

انرژی کل: انتگرال زمانی توان لحظه‌ای در تمام حالت‌ها + `E_BOOT_SERVER_J=500.0` هر رویداد boot + `E_POD_CREATE_J=20.0` هر رویداد pod-create.

---

## ۱۱. پارامترهای پیکربندی (`common/config.py`)

همه‌ی پارامترهای عددی در یک فایل مرکزی قرار دارند و هیچ‌کجای دیگر کد hardcode نمی‌شوند:

```python
# Server Lifecycle
BOOT_DELAY_SEC = 30.0
POD_STARTUP_DELAY_SEC = 5.0
GRACEFUL_TERMINATION_DELAY_SEC = 10.0
SERVER_DRAIN_GRACE_SEC = 15.0
MIN_ACTIVE_DURATION_SEC = 300.0
MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC = 120.0

# Cold start
COLD_START_PENALTY_FRACTION = 0.20
COLD_START_PENALTY_CAP_SEC  = 0.500
COLD_START_WINDOW_SEC       = 10.0

# انرژی
E_BOOT_SERVER_J = 500.0
E_POD_CREATE_J = 20.0

# Scaling / Provisioning
UTIL_SCALE_UP_THRESHOLD = 0.95
UTIL_SCALE_DOWN_THRESHOLD = 0.45
MONITOR_WINDOW_SEC = 30.0
SUSTAIN_LOW_SEC = 60.0
SUSTAIN_HIGH_SEC = 30.0
COOLDOWN_SEC = 60.0
DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD = 0.7
DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD = 0.2
DECISION_INTERVAL_SEC = 30.0

# شبکه
BASE_LATENCY_MS = 2.0
K_MS_PER_KM = 0.02
L0_MS = 20.0
DISPATCH_OVERHEAD_MS = 0.3
PROXIMITY_L0_MS = 7.0

# محدوده‌ی جغرافیایی معتبر داده
LAT_MIN, LAT_MAX = 30.5, 31.7
LON_MIN, LON_MAX = 120.7, 122.0

# PPO Reward
PPO_REWARD_WEIGHTS = {
    "w1_response_time": 0.08,
    "w2_deadline": 0.35,
    "w3_energy": 0.20,
    "w4_load_balance": 0.12,
    "w5_rejected": 0.25,
}
PPO_PENALTY_PER_ACTION = 0.012
```

پارامترهای قابل‌کالیبراسیون (تخمین معقول اولیه، در طول توسعه با آزمایش دقیق‌تر می‌شوند؛ تغییرشان معماری را نمی‌شکند، فقط رفتار عددی سیستم را تنظیم می‌کند): مقدار دقیق `K_MS_PER_KM`, `BASE_LATENCY_MS`, `DISPATCH_OVERHEAD_MS`؛ طول `MONITOR_WINDOW_SEC`, `SUSTAIN_LOW_SEC`, `SUSTAIN_HIGH_SEC`, `COOLDOWN_SEC`, `MIN_ACTIVE_DURATION_SEC`, `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC`؛ وزن‌های reward PPO (`w1..w5`) و `PPO_PENALTY_PER_ACTION`؛ `DECISION_INTERVAL_SEC`؛ وزن‌های solver ILP (`w_count`, `w_energy`, `w_distance`)؛ انتخاب بین `select_replica` مبتنی بر فاصله‌ی خام در برابر `latency_aware_routing`.

---

## ۱۲. معماری کد و ساختار پوشه‌ها

```
edge_rl/
  common/
    models.py          # Server, Replica, Request, StateEnum
    config.py           # SERVER_PROFILES, SERVER_INFO, SERVICES_INFO, پارامترها
    metrics.py           # جمع‌آوری/محاسبه‌ی همه‌ی معیارها
    geo.py                # haversine, network_delay
    logger.py             # لاگ ساخت‌یافته‌ی JSON
    state_builder.py      # ساخت بردار state مشترک برای PPO
    network_coordinates.py # Vivaldi (فعلاً غیرفعال/مرجع)

  data/
    loader.py             # خواندن CSV، فیلتر جغرافیایی، آفست روزانه

  simulator/
    engine.py             # موتور discrete-event (heapq-based)
    events.py              # تعریف رویدادها

  algorithms/
    base.py                # کلاس انتزاعی AlgorithmBase
    greedy/greedy_algorithm.py
    voila/voila_algorithm.py
    hpa/hpa_algorithm.py
    ppo/
      env.py                 # محیط Gymnasium برای آموزش
      policy_network.py
      train.py                # آموزش (BC warm-start + PPO fine-tune)
      infer.py                # اجرای inference-only مدل آموزش‌دیده
      ppo_algorithm.py        # پیاده‌سازی AlgorithmBase با مدل آموزش‌دیده
      optimal_placement.py    # solver ILP جای‌گذاری اولیه

  k8s_adapter/               # اجرای واقعی روی کلاستر Kubernetes
    k8s_client.py              # ساخت/حذف Deployment، cordon/uncordon نود
    redis_state.py             # هماهنگی وضعیت لحظه‌ای روی Redis
    realtime_dispatcher.py     # معادل real-time موتور شبیه‌سازی
    dispatcher_api.py          # سرویس FastAPI کنترل‌پلین (پورت ۹۰۰۰)
    smoke_test.py               # تست اتصال Redis/K8s قبل از اجرای کامل
    worker_service/              # ایمیج Worker واقعی + BTS Simulator

  evaluation/
    compare_runs.py       # اجرای هر ۴ الگوریتم روی داده‌ی یکسان
    aggregate_seeds.py     # میانگین/انحراف‌معیار نتایج PPO روی چند seed

  analyze_decision_quality.py  # تحلیل flapping و کیفیت تصمیمات
  calibrate_constants.py        # کالیبراسیون ثابت‌های نرمال‌سازی state/reward
  build_push_pull_worker.py      # build/push/pull ایمیج Docker Worker
  run.py                          # نقطه‌ی ورود اصلی CLI
  requirements.txt
```

### قرارداد اینترفیس مشترک (`AlgorithmBase`)

```python
class AlgorithmBase(ABC):
    def initial_placement(self, servers, active_bts) -> List[int]: ...
    def select_replica(self, request, candidate_replicas, servers, now) -> Replica | None: ...
    def scale_decision(self, service_id, metrics_snapshot) -> ScaleAction: ...
    def provision_decision(self, servers, metrics_snapshot, now) -> ProvisionAction: ...
    def select_placement_server(self, service_id, servers) -> Optional[int]: ...
    def migration_decision(self, draining_server, servers) -> List[MigrationStep]: ...
    def select_scale_down_victim(self, service_id, ready_replicas, servers, now) -> Replica: ...
```

`initial_placement` و `select_replica` و `select_scale_down_victim` به‌طور پیش‌فرض در `AlgorithmBase` پیاده‌سازی شده‌اند و بین همه‌ی الگوریتم‌ها مشترک‌اند (قابل override در صورت نیاز، مثل VOILA). برای افزودن یک الگوریتم جدید کافی است کلاسی از `AlgorithmBase` ساخته شود و متدهای انتزاعی پیاده‌سازی شوند؛ سپس فقط الگوریتم در `run.py` و `evaluation/compare_runs.py` به تابع سازنده اضافه می‌شود — موتور شبیه‌سازی و adapter کلاستر هیچ تغییری نیاز ندارند.

### نکته‌ی طراحی مهم درباره‌ی موتور شبیه‌سازی و اجرای زنده

موتور شبیه‌سازی (`simulator/engine.py`) و `RealtimeEngine` (اجرای واقعی روی k8s) از **یک منطق تصمیم‌گیری مشترک** (خودِ اشیاء `AlgorithmBase`) استفاده می‌کنند، اما دو پیاده‌سازی جدا از حلقه‌ی رویداد دارند: یکی sync/heap-based برای شبیه‌سازی سریع (discrete-event)، یکی async/asyncio برای زمان واقعی (real-clock) — چون معناشناسی «زمان» در این دو کاملاً متفاوت است (فشرده‌شده در برابر واقعی). این دو باید همیشه دستی هم‌گام نگه داشته شوند؛ هر تغییری در منطق تصمیم در یکی، باید در دیگری هم اعمال شود.

### جریان اصلی موتور شبیه‌سازی

`run()` = `prime()` + حلقه‌ی `step()`:

1. `prime()`: تمام سطرهای دیتاست به رویداد `REQUEST_ARRIVAL` تبدیل و در heap قرار می‌گیرند؛ `_initial_placement()` اجرا می‌شود؛ اولین `DECISION_TICK` در `start_time` زمان‌بندی می‌شود؛ `_cutoff` (زمان توقف کامل شبیه‌سازی) محاسبه می‌شود = آخرین زمان ورود + یک حاشیه‌ی امن (۲×سربار دیسپچ + یک بازه‌ی تصمیم + مدت grace خاموشی) تا رویدادهای معلق (صف‌های در حال تخلیه، سرورهای در حال drain) هم فرصت اتمام داشته باشند.
2. `step()`: رویداد بعدی heap را pop می‌کند، انرژی همه‌ی سرورها را تا این لحظه انتگرال‌گیری می‌کند، سپس بر اساس نوع رویداد یکی از handlerها را صدا می‌زند. `DECISION_TICK` تنها رویدادی است که مقدار برمی‌گرداند (snapshot) و کنترل را به caller پس می‌دهد — این دقیقاً همان نقطه‌ای است که PPO env هر بار یک اکشن اعمال می‌کند.

هر تیک تصمیم:
1. `record_snapshot` (برای `avg_load_balance_cv`).
2. ساخت snapshot متریک — utilization پنجره‌ای دقیق، وضعیت هر سرویس (تعداد رپلیکا، اشغال صف، نرخ رد، نرخ نقض deadline، `demand_centroid`، `proximity_violation_rate`).
3. به‌روزرسانی sustain-tracking (`_low_util_since`/`_high_util_since`).
4. یا از `external_actions` (مسیر PPO env) یا از خودِ `self.algorithm` تصمیم گرفته می‌شود.
5. اعمال provisioning و scale decision — هرکدام قبل از اجرا واقعاً بررسی می‌کنند که آیا شرایط (cooldown، ظرفیت، last-active-server، min-active-duration، mature-replica) برقرار است؛ اگر نه، `skip_reason` در لاگ ثبت می‌شود و ممیزی درستی تصمیم به‌روز می‌شود.
6. شمارنده‌های هر تیک ریست می‌شوند.

---

## ۱۳. عامل PPO-DRL

### ۱۳.۱ محدوده‌ی تصمیمات

یک عامل PPO مرکزی که هر `DECISION_INTERVAL_SEC=30` ثانیه سه نوع تصمیم می‌گیرد: Auto-scaling رپلیکا، Provisioning سرور، Placement رپلیکای جدید. Instance Selection لحظه‌ای خارج از محدوده‌ی تصمیم PPO است (مشترک بین همه‌ی الگوریتم‌ها، بخش ۷) — چون این یک تصمیم روتینگ بلادرنگ برای هر درخواست است، نه یک تصمیم مدیریت منابع.

### ۱۳.۲ فضای حالت (State) — `common/state_builder.py`

```
STATE_DIM = n_servers * 6 + n_services * 6 + 2 = 10*6 + 15*6 + 2 = 152
```

- برای هر سرور (۶ بعد): one-hot وضعیت (۴ حالت: OFF/BOOTING/ACTIVE/DRAINING) + utilization پنجره‌ای + تعداد رپلیکای میزبانی‌شده (نرمال‌شده بر `n_services`).
- برای هر سرویس (۶ بعد): نسبت رپلیکای فعال به کل سرورها + نسبت اشغال صف (کلمپ‌شده روی ۲.۰ و نرمال‌شده) + نرخ نقض deadline + نرخ ورود اخیر (نرمال‌شده) + `rejection_rate` + `proximity_violation_rate`.
- +۲ بعد سراسری: `avg_response_time_recent` نرمال‌شده، `energy_recent_joule` نرمال‌شده.

ثابت‌های نرمال‌سازی حدس دستی نیستند — با اجرای `calibrate_constants.py` روی داده‌ی train واقعی (اجرای Greedy، برداشتن آمار p90/p95 هر کمیت از تیک‌های واقعی تصمیم) به‌دست می‌آیند؛ روش انتخاب: از p90 یا p95 به‌عنوان «حداکثر معمول» (norm=1.0) استفاده شود، نه بیشینه‌ی مطلق (چون یک outlier کل مقیاس را خراب می‌کند).

`build_state_vector()` در یک ماژول مستقل نوشته شده و هم توسط `simulator/engine.py` (از طریق `EdgeResourceEnv`/`PPOAlgorithm`)، هم توسط `k8s_adapter` فراخوانی می‌شود؛ منطق ساخت state هیچ‌جای دیگری تکرار نمی‌شود.

### ۱۳.۳ فضای اقدام (Action) — `MultiDiscrete`

```python
action_space = MultiDiscrete([3]*15 + [3]*10)   # 15 سرویس × {NO_CHANGE, SCALE_UP, SCALE_DOWN} + 10 سرور × {NO_CHANGE, TURN_ON, TURN_OFF}
```

برای بعد سرور: از میان همه‌ی TURN_ON/TURN_OFFهای غیر-NO_CHANGE در همان تیک، فقط **یکی** واقعاً اعمال می‌شود — موتور در هر تیک تصمیم فقط یک اقدام provisioning می‌پذیرد (`ProvisionAction` تکی)، این محدودیت طراحی عمدی است تا از تغییرات هم‌زمان انبوه جلوگیری شود.

Placement رپلیکای جدید (وقتی SCALE_UP انتخاب شد): سروری با بیشترین ظرفیت آزاد در نزدیک‌ترین خوشه به `demand_centroid` آن سرویس (یا مرکز سرورهای فعال اگر centroid نداریم) — همان منطق VOILA برای placement.

### ۱۳.۴ Action Masking — `MaskablePPO`

قبل از هر sample، اکشن‌های نامعتبر ماسک می‌شوند:

- `SCALE_UP` سرویس فقط اگر: سرویس در cooldown نیست **و** حداقل یک سرور ACTIVE با ظرفیت کافی برای این سرویس وجود دارد.
- `SCALE_DOWN` سرویس فقط اگر: در cooldown نیست، بیش از ۱ رپلیکای READY دارد، و حداقل یکی «بالغ» (`mature`) است.
- `TURN_ON` سرور فقط اگر: OFF است و در cooldown نیست.
- `TURN_OFF` سرور فقط اگر: ACTIVE است، در cooldown نیست، تنها سرور فعال نیست، و `MIN_ACTIVE_DURATION_SEC` گذشته.
- `NO_CHANGE` همیشه مجاز (برای هر بعد).

### ۱۳.۵ Reward

```python
penalty = (w1_response_time * norm(avg_response_time_recent)
         + w2_deadline * avg_deadline_violation_rate
         + w3_energy * norm(energy_recent_joule)
         + w4_load_balance * norm(load_balance_cv)
         + w5_rejected * norm(num_rejected_recent))
penalty += PPO_PENALTY_PER_ACTION * n_actions_applied_this_tick     # جریمه‌ی ثابت هر اکشن غیر-NO_CHANGE اعمال‌شده
reward = -penalty
```

وزن‌ها در بخش ۱۱ آمده‌اند. هر جزء reward جدا هم لاگ می‌شود و در TensorBoard زیر `reward_components/*` قابل مشاهده است — این برای تشخیص این‌که کدام جزء دارد بر بقیه غالب می‌شود، حیاتی است (نمونه‌ی خطرناک: وزن نامتعادل می‌تواند باعث شود عامل provisioning را کاملاً متوقف کند).

### ۱۳.۶ آموزش (`algorithms/ppo/train.py`)

- **BC Warm-Start از Greedy:** ابتدا Greedy روی داده‌ی train اجرا و در هر تیک `(state, action)` ثبت می‌شود (حداکثر `bc_max_ticks=10000` تیک به‌طور پیش‌فرض). سپس `behavior_cloning_pretrain` مستقیم روی `model.policy` (شبکه‌ی PyTorch واقعی sb3) با cross-entropy روی توزیع MultiDiscrete آموزش می‌دهد (`Adam`, `lr=5e-5`، پیش‌فرض ۵۰ epoch، batch=64)؛ loss هر epoch در `logs/bc_warmstart_loss.csv` ذخیره می‌شود.
- **Fine-tune با RL:** بعد از warm-start، `MaskablePPO` با `n_steps=2048, batch_size=256, gamma=0.99, lr=3e-4, ent_coef=0.01` و شبکه‌ی `net_arch=dict(pi=[256,256], vf=[256,256])` آموزش می‌بیند.
- **موازی‌سازی:** `n_envs=8` محیط موازی (`DummyVecEnv`). هر env یک provider تصادفی مستقل با seed جدا دارد — برای هر اپیزود، یک پنجره‌ی زمانی تصادفی ۲۴ساعته (`window_hours=24.0`، قابل تنظیم) از تایم‌لاین سه‌روزه‌ی train انتخاب می‌شود.
- **نرمال‌سازی reward:** `VecNormalize(norm_obs=False, norm_reward=True, gamma=0.99)` — آمار running mean/std خودکار محاسبه می‌شود.
- **لاگ منحنی یادگیری:** هر env جدا `logs/monitor/env_{i}.monitor.csv` می‌نویسد (reward per episode)؛ `tensorboard_log=logs/tensorboard` هم فعال است.
- **چک‌پوینت:** `CheckpointCallback` هر `200_000/n_envs` گام یک چک‌پوینت در `logs/checkpoints/` ذخیره می‌کند.
- مدل نهایی: `algorithms/ppo/ppo_model_seed{SEED}.zip` (و `..._vecnormalize.pkl` برای resume آموزش — **در inference لود نمی‌شود** چون مسیر inference مستقیم روی observation خام با `deterministic=True` عمل می‌کند).

اجرا:
```bash
python3 -m algorithms.ppo.train
tensorboard --logdir logs/tensorboard   # اختیاری
```

### ۱۳.۷ ارزیابی/Inference (`algorithms/ppo/infer.py`)

- روی `Data4.csv` (شنبه‌ی هفته‌ی ۴) — بدون یادگیری آنلاین (inference-only، `deterministic=True`) تا مقایسه با سه الگوریتم قاعده‌محور منصفانه باشد.
- پرچم‌های CLI: `--seed` (چند seed مختلف برای گزارش میانگین±انحراف‌معیار)، `--latency-aware-routing`، `--no-solver-placement`، `--w-count/--w-energy/--w-distance`.
- کش هر تیک: چون موتور شبیه‌سازی ممکن است بین دو `DECISION_TICK` چندین بار `scale_decision` را برای سرویس‌های مختلف صدا بزند، `PPOAlgorithm` فقط یک بار در هر `now` واقعی مدل را forward می‌کند و نتیجه را کش می‌کند — از inference تکراری غیرضروری جلوگیری می‌کند.

---

## ۱۴. اجرای زنده روی Kubernetes

### ۱۴.۱ معماری کنترل‌پلین/دیتاپلین جدا

الزام اصلی: «ارسال اطلاعات درخواست از BTS به دیسپچر انجام می‌شود و دیسپچر براساس موقعیت جغرافیایی BTS و فاصله و ترافیکش از سرورها پیشنهاد می‌دهد که درخواستش را به چه سروری ارسال کند.» پیاده‌سازی دقیقاً به این شکل است:

1. **BTS Simulator** (`k8s_adapter/worker_service/bts_simulator.py`) دیتاست را با **زمان‌بندی واقعی wall-clock** (نه فشرده‌شده) replay می‌کند — اگر دو رکورد ۵ ثانیه از هم فاصله دارند، اسکریپت واقعاً ۵ ثانیه صبر می‌کند قبل از ارسال دومی.
2. برای هر رکورد، **دو تماس HTTP واقعی و جدا** زده می‌شود:
   - **مرحله‌ی ۱ (کنترل‌پلین، سبک):** `POST http://<dispatcher-host>:9000/route` با بدنه‌ی `{request_id, service_id, bts_lat, bts_long}`. دیسپچر بر پایه‌ی فاصله‌ی جغرافیایی + وضعیت صف تصمیم می‌گیرد کدام سرور، و فقط **آدرس** (`ip`, `port`, `deadline_sec` باقی‌مانده) را پس می‌دهد — نه این‌که خودش payload را پردازش کند.
   - **مرحله‌ی ۲ (دیتاپلین، سنگین):** `POST http://<pod-ip>:<port>/process` مستقیماً به IP واقعی پاد (که در پاسخ مرحله‌ی ۱ گرفته شد)، **بدون واسطه‌ی دیسپچر**. یعنی ترافیک سنگین (payload واقعی) هرگز از ماشین دیسپچر رد نمی‌شود — فقط تصمیم مسیریابی از آن‌جا رد می‌شود.
3. `response_time_sec` واقعی از دید BTS، از لحظه‌ی *قبل از مرحله‌ی ۱* تا پایان مرحله‌ی ۲ اندازه‌گیری می‌شود — این با تعریف end-to-end در بخش ۶ سازگار است (شامل هر دو RTT).

### ۱۴.۲ هماهنگی state روی Redis (`192.168.1.30`)

چون دو فرآیند جدا داریم (۱: `decision_loop` که هر ۳۰ ثانیه scale/provision اجرا می‌کند، ۲: `dispatcher_api` که هر درخواست واقعی را مسیریابی می‌کند)، Redis دید مشترک و لحظه‌ای فراهم می‌کند:

```
edge:server:{id}:state              -> "OFF"|"BOOTING"|"ACTIVE"|"DRAINING"
edge:replica:{svc}:{srv}:state      -> "STARTING"|"READY"|"DRAINING"|"TERMINATED"
edge:replica:{svc}:{srv}:pod_ip     -> آدرس IP واقعی پاد
edge:replica:{svc}:{srv}:queue      -> شمارنده‌ی اشغال صف (INCR/DECR اتمیک)
edge:service:{svc}:ready_replicas   -> SET سرورهای READY آن سرویس
edge:reservations:{svc}:{srv}       -> ZSET رزروهای صف با score=زمان انقضا (رفع نشتی صف)
edge:metrics:completions            -> LIST پیام‌های تکمیل (fire-and-forget از worker)
service:{svc}:server:{srv}:busy_seconds_acc  -> شمارنده‌ی دقیق ثانیه‌ی busy برای محاسبه‌ی energy
edge:metrics:*                       -> شمارنده‌های زنده برای مانیتورینگ حین اجرا
```

**رفع نشتی صف:** یک شمارنده‌ی ساده‌ی INCR/DECR به‌تنهایی کافی نیست؛ اگر پاد worker بین رزرو صف (در دیسپچر) و پردازش واقعی کرش کند یا در دسترس نباشد، شمارنده‌ی صف *برای همیشه* یک واحد بالاتر از واقعیت می‌ماند — این نشتی به‌تدریج صف را «پر» نشان می‌دهد، درخواست‌های بعدی را غلط رد می‌کند، و تصمیمات scale/provision (که دقیقاً روی `avg_queue_occupancy` لحظه‌ای حساب می‌شوند) را گمراه می‌کند. راه‌حل: هر رزرو موفق هم در شمارنده‌ی سریع (hot path) و هم در یک `ZSET` جدا با score=زمان انقضا (`ttl = deadline_sec + 5`) ثبت می‌شود؛ پردازش موفق در worker خودش رزرو را از ZSET پاک می‌کند؛ یک تسک دوره‌ای (هر ۱۰ ثانیه) رزروهای منقضی‌شده‌ی جامانده را پیدا و شمارنده را برایشان آزاد می‌کند.

### ۱۴.۳ Worker Service واقعی (`k8s_adapter/worker_service/`)

- یک ایمیج Docker یکسان برای هر ۱۵ سرویس؛ تفاوت هر سرویس فقط از طریق متغیرهای محیطی (`EXEC_TIME_SEC`, `SERVICE_ID`, `SERVER_ID`, `SERVICE_PORT`) در Deployment مشخص می‌شود.
- محدودیت «هر رپلیکا هم‌زمان فقط ۱ درخواست» با `asyncio.Semaphore(1)` دور endpoint `/process` پیاده شده (نه با `--limit-concurrency` سطح uvicorn، چون آن فلگ `/healthz` را هم می‌گیرد و پاد هیچ‌وقت Ready نمی‌شود).
- پردازش با `await asyncio.sleep(EXEC_TIME_SEC)` شبیه‌سازی می‌شود (`EXEC_TIME_SEC` از `compute_exec_time_sec` بر اساس پروفایل واقعی سروری که Deployment رویش scheduleشده محاسبه و به‌عنوان env var پاس داده می‌شود).
- بعد از پردازش، پاد خودش (نه دیسپچر): (۱) شمارنده‌ی صف Redis را آزاد می‌کند، (۲) رزرو متناظر را از ZSET پاک می‌کند، (۳) یک رکورد سبک متریک (`edge:metrics:completions`) push می‌کند، (۴) `busy_seconds_acc` را برای محاسبه‌ی energy افزایش می‌دهد. **دیسپچر مرکزی هرگز منتظر پاسخ این پردازش نمی‌ماند** — فقط دوره‌ای (هر ۰.۲ ثانیه) این صف را می‌خواند. این جداسازی صریح باعث می‌شود دیسپچر هرگز bottleneck ترافیک داده نشود.

### ۱۴.۴ K8s Client (`k8s_adapter/k8s_client.py`)

- `create_deployment`/`delete_deployment`: معادل واقعی `_place_replica`/`_handle_replica_terminated` شبیه‌سازی؛ با `_call_with_retry` (backoff نمایی، ۳ تلاش، فقط برای خطاهای ۵۰۰/۵۰۳/۴۲۹ که transient هستند).
- تبدیل `resource_mips` به `millicpu` واقعی Kubernetes: `millicpu = round(resource_mips / server_profile["mips_per_core"] * 1000)` — درخواست CPU در مانیفست Deployment متناسب با سرعت واقعی سرور میزبان تنظیم می‌شود.
- `node_selector={"edge-server-id": str(server_id)}` + `host_network=True` — هر Deployment دقیقاً روی نود متناظر با همان `server_id` پین می‌شود (پیش‌نیاز: نودهای واقعی خوشه باید از قبل با لیبل `edge-server-id=<id>` برچسب بخورند).
- `cordon_node`/`uncordon_node`: معادل OFF/ACTIVE سرور — با `unschedulable=true/false` روی خودِ نود Kubernetes (نه حذف/ایجاد نود؛ نودها همیشه در خوشه هستند، فقط قابل‌زمان‌بندی بودنشان تغییر می‌کند).
- readiness probe روی `/healthz` هر پاد — قبل از این‌که رپلیکا READY اعلام شود چک می‌شود.

### ۱۴.۵ لاگ ساخت‌یافته

هر رکورد JSON خط‌به‌خط شامل: `event_type`, `algorithm`, `sim_time_sec` (یا زمان واقعی wall-clock در حالت k8s)، `wall_time` (ISO)، و فیلدهای مرتبط (`server_id`/`service_id`/`request_id`/متریک لحظه‌ای). انواع رویداد پوشش داده‌شده: `request_arrived, request_routed, request_queued, request_completed, request_rejected, server_boot_started, server_active, server_drain_started, server_off, pod_create_started, pod_ready, pod_drain_started, pod_terminated, scale_decision, provision_decision, migration_started, migration_completed` + رویدادهای تشخیصی اضافه (`emergency_boot_triggered`, `migration_step_dropped`, `server_drain_aborted`, `reservation_sweep`, `pod_ready_timeout`, ...).

خروجی نهایی هر اجرا (چه شبیه‌سازی چه واقعی): یک فایل `<algorithm>_events.jsonl` کامل + `<algorithm>_result.json` با همان قالب بخش ۱۰، تا مقایسه‌ی sim-vs-real و algorithm-vs-algorithm یکسان باشد.

### پیش‌نیازها برای اجرای واقعی

1. لیبل‌گذاری هر ۱۰ نود worker: `kubectl label node <نام‌نود> edge-server-id=<id>`
2. ساخت namespace: `kubectl create namespace edge-rl`
3. build و push ایمیج worker:
   ```bash
   python build_push_pull_worker.py
   ```
4. دسترسی معتبر `~/.kube/config` به کلاستر، و یک نمونه‌ی Redis در دسترس (روی `192.168.1.30:6379`).

پیش از اجرای کامل، حتماً تست اتصال اجرا شود:
```bash
python -m k8s_adapter.smoke_test
```

سپس اجرای واقعی:
```bash
python run.py --algorithm greedy --mode k8s --data test
```

در این حالت، `decision_loop` هر ۳۰ ثانیه‌ی واقعی تصمیمات scale/provision را اعمال می‌کند و `dispatch_loop`/`bts_simulator` به‌طور موازی، رویدادهای CSV را با زمان‌بندی واقعی (نه فشرده) به سرویس‌های واقعی مستقر روی worker nodeها ارسال می‌کند؛ هماهنگی وضعیت لحظه‌ای بین این دو حلقه از طریق Redis انجام می‌شود.

راه‌اندازی دو‌ترمینالی برای اجرای زنده:

```bash
# ترمینال ۱: control-plane + engine.run()
uvicorn k8s_adapter.dispatcher_api:app --port 9000

# ترمینال ۲: مولد ترافیک (جای BTS واقعی)
python3 -m k8s_adapter.bts_simulator
```

---

## ۱۵. ابزارهای ارزیابی و کالیبراسیون

- **`evaluation/compare_runs.py`**: هر ۴ الگوریتم را روی همان داده (train یا test) پشت‌سرهم اجرا می‌کند، هرکدام لاگ/نتیجه‌ی جدا می‌نویسد، و یک `comparison_summary.csv` (جدول کنار‌هم همه‌ی معیارهای بخش ۱۰) تولید می‌کند. اگر مدل PPO برای یک seed پیدا نشود یا کتابخانه‌ای نصب نباشد، آن الگوریتم را رد می‌کند و ادامه می‌دهد (بدون کرش کل مقایسه).
- **`evaluation/aggregate_seeds.py`**: چون یک اجرای PPO تک‌seed کافی برای نتیجه‌گیری آماری معتبر نیست، این اسکریپت نتایج چند seed مختلف (هرکدام باید از قبل با `algorithms/ppo/train.py` جدا آموزش دیده و با `compare_runs.py --seed N` ارزیابی شده باشد) را می‌خواند و میانگین±انحراف‌معیار/min/max هر معیار کلیدی را گزارش می‌دهد.
- **`calibrate_constants.py`**: یک اجرای کامل Greedy روی داده‌ی train را دنبال می‌کند و توزیع خام (`recent_arrivals`, `avg_response_time_recent`, `energy_recent_joule`, `num_rejected_recent`) هر تیک را جمع می‌کند، سپس mean/median/p90/p95/p99/max هرکدام را چاپ می‌کند — مبنای انتخاب ثابت‌های نرمال‌سازی state/reward.
- **`build_push_pull_worker.py`**: اسکریپت کمکی برای build و push ایمیج Docker پاد Worker به رجیستری خصوصی (`192.168.1.30:5000/edge-worker:latest`) قبل از استقرار روی خوشه‌ی واقعی.
- **`k8s_adapter/smoke_test.py`**: تست دودی سریع (health-check اتصال به Redis، اتصال به Kubernetes API، لیبل‌گذاری نودها، یک round-trip کامل ساخت/انتظار/فراخوانی/حذف یک Deployment آزمایشی) قبل از اجرای کامل روی خوشه‌ی واقعی.
- **`analyze_decision_quality.py`**: طبقه‌بندی SCALE_UP/SCALE_DOWN و تحلیل flapping سرور (بخش ۱۰).

اجرای مقایسه‌ی چهارگانه:

```bash
python -m evaluation.compare_runs --data test
```

اجرای جمع‌بندی چند-seed (بعد از آموزش/ارزیابی جدا هر seed):

```bash
EOTCH_SEED=42 python3 -m algorithms.ppo.train
EOTCH_SEED=42 python3 -m evaluation.compare_runs --output-dir outputs/seed42
EOTCH_SEED=43 python3 -m algorithms.ppo.train
EOTCH_SEED=43 python3 -m evaluation.compare_runs --output-dir outputs/seed43
python3 -m evaluation.aggregate_seeds --seeds 42 43 44 45 --base-dir outputs
```

---

## ۱۶. نصب و وابستگی‌ها

```bash
python -m venv venv
source venv/bin/activate        # ویندوز: venv\Scripts\activate
pip install -r requirements.txt
```

```
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
```

برای اجرای واقعی روی Kubernetes به‌طور جداگانه نیاز است: `kubernetes`, `redis`, `httpx`, `paramiko`.

برای اجزای سمت Worker (داخل ایمیج سبک): `fastapi==0.115.*`, `uvicorn==0.30.*`, `pydantic==2.*`, `redis>=5.0` (در `k8s_adapter/worker_service/requirements.txt` جدا).

مسیر پوشه‌ی داده با متغیر محیطی قابل تنظیم است (پیش‌فرض: `<ریشه‌ی پروژه>/data/raw`):

```bash
export EOTCH_DATA_DIR=/path/to/data      # لینوکس/مک
set EOTCH_DATA_DIR=D:\path\to\data       # ویندوز (cmd)
```

seed آموزش/اجرا با متغیر محیطی قابل تنظیم است تا بتوان چند مدل با seedهای مختلف آموزش داد:

```bash
export EOTCH_SEED=43
```

---

## ۱۷. راهنمای اجرا

اجرای یک الگوریتم روی داده‌ی تست (شبیه‌سازی):

```bash
python run.py --algorithm greedy --data test
python run.py --algorithm voila  --data test
python run.py --algorithm hpa    --data test
python run.py --algorithm ppo    --data test   # نیاز به مدل آموزش‌دیده دارد
```

آرگومان‌های `run.py`:

| آرگومان | مقادیر | توضیح |
|---|---|---|
| `--algorithm` | `greedy` / `voila` / `hpa` / `ppo` | الگوریتم مورد استفاده |
| `--mode` | `sim` (پیش‌فرض) / `k8s` | شبیه‌سازی یا اجرای واقعی روی کلاستر |
| `--data` | `test` (پیش‌فرض) / `train` | مجموعه‌ی داده |
| `--output-dir` | مسیر دلخواه (پیش‌فرض `outputs/`) | محل ذخیره‌ی لاگ‌ها و نتایج |
| `--latency-aware-routing` | flag | PPO: مسیریابی بر پایه‌ی تخمین کل تأخیر به‌جای صرفاً فاصله |
| `--no-solver-placement` | flag | PPO: غیرفعال‌کردن ILP، fallback به پوشش حریصانه |

خروجی هر اجرا:
- `outputs/<algorithm>_events.jsonl` — لاگ ساخت‌یافته‌ی هر رویداد.
- `outputs/<algorithm>_result.json` — خلاصه‌ی معیارهای نهایی.

آموزش PPO:

```bash
python -m algorithms.ppo.train
```

ارزیابی inference-only بعد از آموزش:

```bash
python -m algorithms.ppo.infer
python -m algorithms.ppo.infer --latency-aware-routing --w-energy 2.0
```

مقایسه‌ی هر چهار الگوریتم:

```bash
python -m evaluation.compare_runs --data test
```

تحلیل کیفیت تصمیم:

```bash
python analyze_decision_quality.py outputs/ppo_events.jsonl
```

---

## ۱۸. نقشه‌ی ساخت پروژه از صفر

ترتیب توسعه‌ای که منطقی‌ترین وابستگی‌ها را رعایت می‌کند:

1. **`common/`** — مدل داده (`models.py`)، پیکربندی مرکزی (`config.py`)، متریک (`metrics.py`)، محاسبات جغرافیایی (`geo.py`)، لاگ ساخت‌یافته (`logger.py`).
2. **`data/loader.py`** — پیش‌پردازش دیتاست BTS شانگهای و ساخت تایم‌لاین پیوسته.
3. **`simulator/engine.py`** + **`algorithms/base.py`** — موتور discrete-event و اینترفیس مشترک الگوریتم‌ها؛ در همین مرحله چرخه‌ی کامل یک درخواست (بخش ۶)، مقداردهی اولیه (بخش ۵)، provisioning/migration (بخش ۸) و auto-scaling (بخش ۹) پیاده می‌شوند.
4. **چهار الگوریتم** (`greedy`, `hpa`, `voila`, `ppo`) — از ساده به پیچیده: ابتدا Greedy به‌عنوان baseline، سپس HPA، سپس VOILA (با منطق medoid/proximity)، و در نهایت PPO (که نیازمند `common/state_builder.py`، محیط Gymnasium، و آموزش با BC warm-start است).
5. **`evaluation/`** — برای مقایسه‌ی چهارگانه و جمع‌بندی چند-seed، بعد از این‌که همه‌ی الگوریتم‌ها قابل‌اجرا هستند.
6. **`k8s_adapter/`** — در نهایت، برای پورت کردن همان منطق تصمیم‌گیری (بدون تغییر در `AlgorithmBase`) به یک خوشه‌ی Kubernetes واقعی، با هماهنگی Redis و دیسپچر HTTP واقعی (`dispatcher_api.py` + `realtime_dispatcher.py` + `k8s_client.py` + `worker_service/`).

نکته‌ی کلیدی در تمام این مسیر: **هیچ منطق تصمیم‌گیری‌ای دو بار نوشته نمی‌شود.** موتور شبیه‌سازی و موتور real-time هر دو از همان اشیاء `AlgorithmBase` استفاده می‌کنند؛ `build_state_vector` فقط یک بار در `common/state_builder.py` تعریف می‌شود؛ ثابت‌های نرمال‌سازی state/reward با کالیبراسیون روی داده‌ی واقعی به‌دست می‌آیند، نه حدس دستی. هر تغییری در یک منطق مشترک (مثل `select_replica`، `migration_decision`، یا آستانه‌های provisioning) باید همزمان در مسیر شبیه‌سازی و مسیر k8s واقعی اعمال شود، وگرنه رفتار دو مسیر به‌آرامی از هم واگرا می‌شود و مقایسه‌ی نتایج بی‌اعتبار می‌شود.
