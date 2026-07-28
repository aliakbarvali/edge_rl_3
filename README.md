# edge-resource-management (eotch)

شبیه‌سازی مدیریت پویای منابع لبه (Edge Resource Management) با ۴ الگوریتم
قابل‌سوییچ (Greedy / VOILA / Kubernetes-HPA / PPO-DRL)، طبق «سند معماری کامل
پروژه نسخه ۱.۰».

## وضعیت فعلی (فاز ۱ و ۲ از نقشه‌راه بخش ۰ سند)

| بخش | وضعیت |
|---|---|
| `common/` (config, models, geo, metrics, logger, state_builder) | ✅ کامل |
| `data/loader.py` (۴ فایل + آفست روزانه) | ✅ کامل |
| `simulator/engine.py` (موتور discrete-event) | ✅ کامل |
| `algorithms/base.py` (AlgorithmBase مشترک) | ✅ کامل |
| `algorithms/greedy/` | ✅ کامل و تست‌شده |
| `algorithms/voila/` | ✅ کامل و تست‌شده (بخش ۲ جدید) |
| `algorithms/hpa/` | ✅ کامل و تست‌شده (بخش ۲ جدید) |
| `algorithms/ppo/` (env, train, infer, ppo_algorithm) | ✅ کد کامل — **آموزش روی سیستم من اجرا نشد** (زیر را بخوانید) |
| `k8s_adapter/` (فاز ۳) | ⏳ هنوز شروع نشده |

## ⚠️ مهم‌ترین نکته قبل از هرچیز

**در محیط توسعه‌ی من (sandbox) دسترسی شبکه به `gymnasium`, `stable-baselines3`,
`sb3-contrib`, `torch`, `simpy` وجود نداشت** (pip نمی‌توانست این پکیج‌ها را
resolve کند، با اینکه دامنه‌های pypi در تنظیمات مجاز بودند). بنابراین:

1. `simulator/engine.py` را با یک موتور heapq-based **معادل** simpy نوشتم
   (مستند در بالای همان فایل). اگر simpy روی سیستم خودتان در دسترس است و
   ترجیحش می‌دهید، فقط همین فایل نیاز به بازنویسی دارد.
2. تمام کد PPO (`env.py`, `train.py`, `ppo_algorithm.py`) نوشته و تا حد ممکن
   تست شده (با یک ماژول `gymnasium` جعلی برای بررسی منطق `reset/step`/
   action masking بدون نیاز به نصب واقعی)، **اما هرگز واقعاً `train.py` را با
   torch/sb3-contrib واقعی اجرا نکردم**. لطفاً:
   ```bash
   pip install -r requirements.txt
   python -m algorithms.ppo.train
   ```
   را خودتان اجرا کنید و اگر خطایی داد (مثلاً ناسازگاری نسخه‌ی API
   `MaskablePPO`/`get_distribution`) برایم بفرستید تا دیباگ کنم.

همه‌ی بخش‌های دیگر (`common/`, `data/`, `simulator/engine.py`,
`algorithms/greedy/`, `algorithms/base.py`) با داده‌ی واقعی شما کاملاً اجرا و
تست شدند (نتایج زیر).

## نحوه‌ی اجرا

```bash
pip install -r requirements.txt

# یک الگوریتم را روی داده‌ی تست (Data4.csv) اجرا کن
python run.py --algorithm greedy --data test

# آموزش PPO (روی Data1-3.csv، با BC warm-start از Greedy)
python -m algorithms.ppo.train

# ارزیابی PPO آموزش‌دیده (روی Data4.csv، inference-only)
python -m algorithms.ppo.infer

# مقایسه‌ی همه‌ی الگوریتم‌های *آماده* (فعلاً greedy + ppo اگر آموزش دیده شود)
python -m evaluation.compare_runs --data test
```

## نتیجه‌ی واقعی هر سه الگوریتم روی Data4.csv (تست‌شده در sandbox من)

| متریک | Greedy | **Voila** | HPA |
|---|---|---|---|
| avg_response_time_sec | 50.89 | 51.50 | 52.72 |
| deadline_violation_rate_pct | 4.88 | **4.79** | 5.45 |
| deadline_violations | 1667 | **1634** | 1859 |
| avg_load_balance_cv | 0.302 | 0.286 | **0.246** |
| cumulative_energy_joule | 31.8M | 31.5M | **30.7M** |
| num_requests_rejected_no_replica | 68 | **22** | 19 |
| avg_active_servers | 2.17 | 2.13 | 2.02 |
| num_server_boots/shutdowns | 93/91 | 76/74 | 6/4 |
| completed | 33029 | **33079** | 32931 |

**تفسیر صادقانه:** Voila کمترین نقض deadline و کمترین رد به‌خاطر نبود replica
را دارد (دقیقاً چیزی که فلسفه‌ی location-aware آن پیش‌بینی می‌کند). یک نکته‌ی
جالب: `avg_distance_km` واقعی Greedy (۱۸.۱۵) از Voila (۲۰.۶۴) کمتر است -
چون Greedy در مجموع replica بیشتری ساخته (۳۶۵ در برابر ۲۸۳ pod-create) که
خودش فاصله‌ی متوسط تا نزدیک‌ترین replica را کم می‌کند، صرف‌نظر از هوشمندی
مکان‌یابی؛ این نشان می‌دهد `avg_distance_km` به‌تنهایی معیار کاملی برای
سنجش کیفیت placement نیست - باید همراه با تعداد replica/انرژی خوانده شود.
HPA چون کاملاً latency-unaware است (بخش ۳ مقدمه‌ی مقاله‌ی Voila)، کمترین
انرژی و کمترین churn زیرساخت را دارد ولی بدترین کیفیت QoS را.

## تصمیمات مهندسی/فرضیات صریح (برای گزارش نهایی حتماً ذکر کنید)

- **`L0_MS = 20ms`**: در بخش ۹ سند مقدار عددی `l0` (که در بخش ۴/۵ استفاده
  می‌شود) فراموش شده بود؛ طبق مقدار پیش‌فرض خود مقاله‌ی Voila قرار داده شد.
- **`bts_id` سرورها**: نزدیک‌ترین BTSID واقعی دیتاست به هر مختصات داده‌شده
  (همه زیر ۱ کیلومتر فاصله - محاسبه‌شده یک‌بار روی هر ۴ روز، مقادیر داخل
  `common/config.py`).
- **جایگذاری اولیه با ظرفیت کافی**: پوشش حریصانه‌ی صرفِ جغرافیایی (بخش ۴)
  کافی نبود چون مجموع `cpu_demand` هر ۱۵ سرویس (=۲۴۸) از ظرفیت هر سرور
  به‌تنهایی (حداکثر ۲۰۰) بیشتر است؛ انتخاب اولیه را تا کافی‌شدن ظرفیت کل هم
  گسترش دادم (`algorithms/base.py:initial_placement`).
- **محافظ ایمنی drain**: اگر migration نتواند مقصدی برای همه‌ی سرویس‌های
  تک‌رپلیکای یک سرور در حال drain پیدا کند، drain آن چرخه لغو می‌شود (به‌جای
  قطع کامل آن سرویس‌ها). بخش ۶.۲ سند در این حالت لبه می‌گوید یک سرور جدید
  Boot اضطراری شود - این حالت را فعلاً به‌صورت «عقب‌انداختن drain تا چرخه‌ی
  بعد» ساده‌سازی کردم؛ اگر پیاده‌سازی دقیق‌تر emergency-boot را می‌خواهید بگویید.
- **utilization برای provisioning**: `used_cpu/capacity` بر اساس رپلیکاهای
  *در حال پردازش* در لحظه (نه صرفاً deploy‌شده)، دقیقاً طبق فرمول بخش ۲.۴.
- **مرکز ثقل تقاضا برای Voila**: چون AlgorithmBase استاندارد به snapshot خام
  موقعیت هر درخواست دسترسی نمی‌دهد، `simulator/engine.py` یک میانگین متحرک
  نمایی (EMA, α=۰.۳) از موقعیت جغرافیایی درخواست‌های اخیر هر سرویس را در
  `metrics_snapshot["services"][sid]["demand_centroid"]` نگه می‌دارد؛ Voila
  از این برای انتخاب مکان replica/migration نزدیک به تقاضای واقعی استفاده
  می‌کند (برخلاف Greedy/HPA که فقط بر اساس موقعیت خودِ سرورها تصمیم می‌گیرند).
- **آستانه‌های heuristic هر الگوریتم** (`OCC_UP_THRESHOLD`, `TARGET_UTILIZATION`
  و مشابه) در همان فایل الگوریتم نگه داشته شدند، نه `config.py`، چون این‌ها
  سیاست تصمیم‌گیری مختص هر الگوریتم‌اند، نه قید فیزیکی سیستم.
- **ثابت‌های نرمال‌سازی reward PPO** (`_NORM_RESPONSE_TIME_SEC=300`,
  `_NORM_ENERGY_JOULE=50000` در `state_builder.py`/`env.py`): چون سند «نرمال‌سازی
  با warm-up» را پیشنهاد داده ولی مکانیزم دقیقش را مشخص نکرده، مقادیر ثابت و
  مستند به‌جایش گذاشتم؛ در صورت نیاز به warm-up واقعی بگویید تا اضافه کنم.

## ساختار پروژه

مطابق دقیق بخش ۱۰ سند معماری:

```
eotch/
  common/{config,models,metrics,geo,logger,state_builder}.py
  data/loader.py
  simulator/{engine,events}.py
  algorithms/
    base.py
    greedy/greedy_algorithm.py
    voila/voila_algorithm.py      (فاز ۲)
    hpa/hpa_algorithm.py          (فاز ۲)
    ppo/{env,policy_network,train,infer,ppo_algorithm}.py
  evaluation/compare_runs.py
  run.py
```
