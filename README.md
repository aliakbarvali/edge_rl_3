# edge-resource-management (eotch)

شبیه‌سازی مدیریت پویای منابع لبه (Edge Resource Management) با ۴ الگوریتم
قابل‌سوییچ (Greedy / VOILA / Kubernetes-HPA / PPO-DRL)، طبق «سند معماری کامل
پروژه نسخه ۱.۰».

## وضعیت فعلی (فاز ۱ و ۲ از نقشه‌راه بخش ۰ سند)

| بخش | وضعیت |
|---|---|
| `common/` (config, models, geo, metrics, logger, state_builder) | ✅ کامل |
| `data/loader.py` (۴ فایل + آفست روزانه) | ✅ کامل |
| `simulator/engine.py` (موتور discrete-event) | ✅ کامل — شامل emergency-boot بخش ۶.۲ و gate ظرفیت-محور جدید (زیر را بخوانید) |
| `algorithms/base.py` (AlgorithmBase مشترک) | ✅ کامل |
| `algorithms/greedy/` | ✅ کامل و تست‌شده — تنها الگوریتمی که فیکس capacity-starved را در تصمیم خودش هم دارد |
| `algorithms/voila/` | ✅ کامل و تست‌شده — هنوز فیکس capacity-starved را در تصمیم خودش ندارد (زیر را بخوانید) |
| `algorithms/hpa/` | ✅ کامل و تست‌شده — هنوز فیکس capacity-starved را در تصمیم خودش ندارد |
| `algorithms/ppo/` (env, train, infer, ppo_algorithm) | ✅ کامل — آموزش با BC warm-start + fine-tune واقعی (۳ میلیون timestep، ۸ دور train، نتایج زیر) |
| `analyze_decision_quality.py` | ✅ ابزار تحلیل کیفیت تصمیم با پنجره‌ی lookahead + تحلیل flapping سرور (جدید) |
| `k8s_adapter/` (فاز ۳) | ⏳ هنوز شروع نشده |

## نحوه‌ی اجرا

```bash
pip install -r requirements.txt

# یک الگوریتم را روی داده‌ی تست (Data4.csv) اجرا کن
python run.py --algorithm greedy --data test

# آموزش PPO (روی Data1-3.csv، با BC warm-start از Greedy)
python -m algorithms.ppo.train

# ارزیابی PPO آموزش‌دیده (روی Data4.csv، inference-only)
python -m algorithms.ppo.infer

# مقایسه‌ی هر ۴ الگوریتم روی همان داده و تولید جدول مقایسه
python -m evaluation.compare_runs --data test

# تحلیل کیفیت تصمیمات یک اجرا (SCALE_UP/DOWN با lookahead + flapping سرور)
python3 analyze_decision_quality.py outputs/greedy_events.jsonl
```

> **نکته‌ی reproducibility:** `algorithms/ppo/ppo_model.zip` در ریپو commit
> نشده (`.gitignore` قانون `*.zip` دارد). برای بازتولید نتایج PPO زیر، اول
> `python -m algorithms.ppo.train` را اجرا کنید.

> **متغیر محیطی داده:** مسیر داده‌ها با `EOTCH_DATA_DIR` تنظیم می‌شود
> (پیش‌فرض: `<ریشه‌ی پروژه>/data/raw`).

## نتیجه‌ی واقعی هر ۴ الگوریتم روی Data4.csv (بعد از فیکس capacity-starved + بازآموزی PPO)

اجرای کامل `evaluation/compare_runs.py --data test` (۳۴٬۱۳۷ درخواست کل):

| متریک | Greedy | Voila | HPA | **PPO** |
|---|---|---|---|---|
| avg_response_time_sec | **48.88** | 52.54 | 51.10 | 52.90 |
| p95 / p99 response_time_sec | 211.0 / 413.0 | 241.0 / 440.0 | 233.0 / 429.0 | 246.0 / 449.2 |
| deadline_violations | **1305** | 1794 | 1596 | 1825 |
| deadline_violation_rate_pct | **3.82** | 5.26 | 4.68 | 5.35 |
| cumulative_energy_joule | 42.1M | 43.0M | 40.0M | **30.6M** |
| avg_distance_km | 17.55 | **17.09** | 17.39 | 20.49 |
| avg_load_balance_cv | 0.882 | 0.858 | 0.742 | **0.245** |
| avg_network_delay_ms | 2.351 | 2.342 | 2.348 | **2.410** (بدترین، تفاوت ناچیز) |
| num_requests_rejected_no_replica | 14 | 14 | 14 | 14 |
| num_requests_rejected_queue_full | **861** | 1139 | 1043 | 1141 |
| avg_active_servers | 1.44 | 1.17 | 1.57 | **2.01** |
| num_server_boots / shutdowns | **319 / 317** | 85 / 81 | 194 / 190 | **10 / 8** |
| num_pod_creates / deletes | 623 / 608 | 180 / 165 | 474 / 459 | 72 / 55 |
| completed_requests | **33262** | 32984 | 33080 | 32982 |

### کیفیت تصمیمات (بخش ۸ سند: correctness مستقل از منطق داخلی هر الگوریتم)

| الگوریتم | SCALE_UP correct/incorrect (rate) | SCALE_DOWN correct/incorrect (rate) | TURN_ON correct (missed) | TURN_OFF correct (missed) |
|---|---|---|---|---|
| Greedy | 569 / 0 (100%) | 59 / 0 (100%) | 306 (missed 388) | 318 (missed 96) |
| Voila | 134 / 0 (100%) | 36 / 0 (100%) | 63 (missed 1365) | 84 (missed 54) |
| HPA | 388 / 0 (100%) | 60 / 201 (**23%**) | 21 (missed 914) | 191 (missed 726) |
| **PPO** | 22 / 34 (**39%**) | 8 / 7 (**53%**) | 7 (missed 1567) | 8 (missed 750) |

## تفسیر صادقانه

### ۱) باگ «capacity-starved» رفع شد — و این مهم‌ترین تغییر این نسخه است

معیار قدیمی `turn_on_necessary` فقط utilization لحظه‌ای (busy-fraction رپلیکاهای
*در حال پردازش*) هر سرور ACTIVE را می‌سنجید، نه اینکه اصلاً `free_capacity`ی
برای رپلیکای جدید مانده یا نه. یک سرور می‌توانست کاملاً پر باشد
(`free_capacity=0`) ولی چون هم‌زمان همه‌ی رپلیکاهایش مشغول نبودند
`utilization<0.95` بماند — یعنی TURN_ON هرگز trigger نمی‌شد، حتی وقتی سیستم
واقعاً جای رشد نداشت (شواهد واقعی: ۳۵۰۰ از ۳۵۰۸ تلاش SCALE_UP با
`no_target_server` شکست خورده بودند). فیکس (`_any_service_capacity_starved`
در `simulator/engine.py` + `_capacity_starved_services` در
`algorithms/base.py`) این را برطرف کرد. نتیجه: کیفیت QoS به‌طور محسوس بهتر
شد — Greedy از deadline_violation_rate=5.11% به **3.82%** رسید، completed_requests
از 33009 به **33262** رسید.

### ۲) اما این فیکس یک الگوی flapping آشکار در provisioning ایجاد کرده

مقایسه‌ی قبل/بعدِ فیکس برای Greedy:

| | قبل | بعد |
|---|---|---|
| num_server_boots / shutdowns | 51 / 49 | **319 / 317** |
| cumulative_energy_joule | 32.2M | **42.1M** (⬆ با وجود کاهش avg_active_servers) |
| avg_active_servers | 2.10 | **1.44** (⬇) |

انرژی بالا رفته **با وجود** کاهش میانگین سرور فعال — نشانه‌ی کلاسیک flapping:
سرورها زمان زیادی را در حالت‌های گذار (`BOOTING`/`DRAINING`) می‌گذرانند که
برق مصرف می‌کنند (طبق فرمول بخش ۲.۴ سند) ولی در `avg_active_servers` شمرده
نمی‌شوند. با تقریب `avg_active_servers × ۸۶۴۰۰ثانیه ÷ num_boots` (میانگین
مدت واقعی ACTIVE‌ماندن هر چرخه‌ی boot):

| الگوریتم | میانگین dwell تقریبی هر چرخه |
|---|---|
| Greedy | **~۳۸۹ ثانیه (~۶.۵ دقیقه)** |
| HPA | ~۶۹۸ ثانیه (~۱۱.۶ دقیقه) |
| Voila | ~۱۱۸۶ ثانیه (~۱۹.۸ دقیقه) |
| PPO | ~۱۷٬۳۸۸ ثانیه (~۴.۸ ساعت) — بدون flapping |

حداقل زمان تئوریک یک چرخه‌ی کامل boot→cooldown(60s)→drain تقریباً ۹۰-۱۱۰
ثانیه است (`COOLDOWN_SEC=60` + یک `DECISION_INTERVAL_SEC=30` + drain grace)؛
یعنی Greedy با میانگین ۳۸۹ ثانیه فقط ~۳.۵-۴ برابر این حداقل فاصله دارد —
عملاً نزدیک به سریع‌ترین چرخه‌ی ممکن سیستم. این را با نسخه‌ی گسترش‌یافته‌ی
`analyze_decision_quality.py` (بخش جدید TURN_ON/TURN_OFF flapping) روی
یک اجرای آزمایشی synthetic هم تأیید کردیم: **۸۸٪ از چرخه‌های on→off زیر
آستانه‌ی ۳۰۰ ثانیه بودند**، با چند چرخه دقیقاً روی کف تئوریک ۹۰ ثانیه.
(برای عدد دقیق روی Data4.csv واقعی، این ابزار را روی
`outputs/greedy_events.jsonl` خودتان اجرا کنید — فایل‌های jsonl واقعی در
این ریپو commit نشده‌اند.)

**چرا PPO این مشکل را ندارد:** gate سطح موتور (`_any_service_capacity_starved`)
برای هر ۴ الگوریتم یکسان اعمال می‌شود، ولی چون مدل PPO خودش به‌ندرت TURN_ON
پیشنهاد می‌دهد (فقط ۱۰ بار در کل روز)، این gate جدید عملاً کمتر رویش اثر
می‌گذارد — نتیجه: کمترین انرژی (30.6M، حدود ۲۷٪ کمتر از میانگین سه‌تای
دیگر) و بهترین `avg_load_balance_cv` (0.245) در کل مقایسه، اما با بدترین
`deadline_violation_rate` (5.35%) و بیشترین `rejected_queue_full` (1141) —
یعنی محافظه‌کاری در provisioning را با QoS ضعیف‌تر معامله کرده.

**پیشنهاد فنی برای رفع flapping:** سیگنال `_any_service_capacity_starved`
برخلاف `_any_active_server_sustained_overloaded` هیچ الزام تداوم (چند-تیکی)
ندارد — یک نمونه‌ی لحظه‌ای کافی است تا TURN_ON مجاز شود. اضافه‌کردن یک
sustain مشابه (مثلاً چند تیک متوالی capacity-starved، هم‌تراز
`SUSTAIN_HIGH_SEC`) یا افزایش `COOLDOWN_SEC` مختص بعد از boot (قبل از
مجازبودن هر TURN_OFF روی همان سرور) می‌تواند این نوسان را کاهش دهد بدون از
دست‌دادن بهبود QoS بخش ۱.

### ۳) ناهماهنگی بین الگوریتم‌ها در استفاده از فیکس capacity-starved

فیکس در سطح *تصمیم الگوریتم* (نه فقط gate سطح موتور) فقط به Greedy اضافه
شده (`algorithms/greedy/greedy_algorithm.py` از `_capacity_starved_services`
مشترک در `base.py` استفاده می‌کند)؛ HPA و Voila دست‌نخورده ماندند و هنوز
فقط بر مبنای `avg_util` خودشان TURN_ON پیشنهاد می‌دهند. همین چیز در جدول
missed-opportunity هم دیده می‌شود: TURN_ON missed برای Voila=1365 و
HPA=914، در برابر Greedy=388 (که چون خودش این سیگنال را می‌بیند، کمتر آن
را از دست می‌دهد). برای مقایسه‌ی کاملاً منصفانه‌ی ۴ الگوریتم، پیشنهاد
می‌شود همان یک خط (`self._capacity_starved_services(...)`) به
`provision_decision` در `hpa_algorithm.py` و `voila_algorithm.py` هم اضافه
شود.

### ۴) رد به‌دلیل نبود replica (`=14`) در هر ۴ الگوریتم برابر است

این تصادفی نیست: منطق `_initial_placement` بین هر ۴ الگوریتم مشترک است
(`AlgorithmBase.initial_placement`/`select_replica`) و این ۱۴ رد همگی در
پنجره‌ی اول (پیش از هر تصمیم الگوریتم‌محور) رخ می‌دهند.

## تصمیمات مهندسی/فرضیات صریح (برای گزارش نهایی حتماً ذکر کنید)

- **`L0_MS = 20ms`**: در بخش ۹ سند مقدار عددی `l0` فراموش شده بود؛ طبق
  مقدار پیش‌فرض خود مقاله‌ی Voila قرار داده شد.
- **`bts_id` سرورها**: نزدیک‌ترین BTSID واقعی دیتاست به هر مختصات داده‌شده.
- **جایگذاری اولیه با ظرفیت کافی**: پوشش حریصانه‌ی صرفِ جغرافیایی (بخش ۴)
  کافی نبود؛ انتخاب اولیه تا کافی‌شدن ظرفیت کل هم گسترش داده شد
  (`algorithms/base.py:initial_placement`).
- **Emergency-boot بخش ۶.۲**: وقتی migration یک سرویس تک‌رپلیکای در حال
  drain نتواند مقصد ACTIVE مناسبی پیدا کند، یک سرور OFF جدید بلافاصله Boot
  می‌شود («Provisioning اضطراری» طبق متن دقیق سند) و migration واقعی
  به‌محض ACTIVE شدن آن سرور کامل می‌شود. برای جلوگیری از trigger مکرر روی
  همان سرویس (که در نسخه‌ی اول این فیکس باعث ۵۱۰ بار boot پیاپی و بی‌مورد
  شده بود)، یک دیکشنری ردیابی (`_emergency_boot_for_service`) اضافه شد که
  تا وقتی سرور مقصد قبلی ACTIVE نشده، دوباره برای همان سرویس boot جدید
  trigger نمی‌کند (`simulator/engine.py:_trigger_emergency_boot`).
- **فیکس «capacity-starved» بخش ۶.۱**: نگاه کنید بخش «تفسیر صادقانه» بالا —
  این مهم‌ترین تغییر رفتاری این نسخه است و trade-off واقعی (QoS بهتر در
  برابر churn/انرژی بیشتر برای الگوریتم‌های قاعده‌محور) ایجاد کرده که هنوز
  به‌طور کامل رفع نشده (پیشنهاد: افزودن الزام تداوم مشابه `SUSTAIN_HIGH_SEC`).
- **⚠️ رگرسیون شناخته‌شده - لاگ کامل بخش ۱۲**: نسخه‌ی قبلی این فایل رویدادهای
  `request_routed` و `request_queued` را هم لاگ می‌کرد (برای audit trail
  کامل بخش ۳ گام ۳/۴). این نسخه که emergency-boot را از نو با شواهد واقعی
  بازنویسی کرده، این دو رویداد را ندارد. اگر audit trail کامل بخش ۱۲ لازم
  است، باید این دو دوباره به `_handle_arrival` اضافه شوند.
- **رفع باگ متریک فاصله/تاخیر شبکه**: `avg_distance_km`/`avg_network_delay_ms`
  فقط برای درخواست‌های `COMPLETED` محاسبه می‌شوند (`common/metrics.py`).
- **وزن‌های reward PPO**: `w1_response_time=0.12, w2_deadline=0.20,
  w3_energy=0.30, w4_load_balance=0.23, w5_rejected=0.15` (جمع=۱.۰،
  `common/config.py`)؛ `num_rejected_recent` نرمال و با وزن صریح ترکیب
  می‌شود (نه جریمه‌ی جدا و نرمال‌نشده مثل قبل).
- **آموزش PPO نهایی**: `total_timesteps=3,000,000` (از ۵۰۰هزار افزایش
  یافت)، `bc_epochs=25` (از ۱۰) — ۸ دور train کامل انجام شد (نگاه کنید
  `logs/tensorboard/ppo_run_1..8`).
- **مرکز ثقل تقاضا برای Voila**: EMA (α=۰.۳) از موقعیت جغرافیایی درخواست‌های
  اخیر هر سرویس (`simulator/engine.py`، `demand_centroid`).
- **ثابت نرمال‌سازی انرژی**: `_NORM_ENERGY_JOULE=12000`؛ فعلاً در دو جا
  (`state_builder.py` و `env.py`) مستقل هاردکد شده.
- **پورتابیلیتی `DATA_DIR`**: با `EOTCH_DATA_DIR` (fallback به
  `<ریشه‌ی پروژه>/data/raw`).

## موارد باز/شناخته‌شده (برای ادامه‌ی کار)

1. **flapping بعد از فیکس capacity-starved** (بالا) — بیشترین اولویت؛ نیاز
   به sustain چند-تیکی مشابه overload.
2. **ناهماهنگی بین الگوریتم‌ها**: `_capacity_starved_services` فقط در
   Greedy استفاده می‌شود، نه HPA/Voila.
3. **رگرسیون لاگ**: `request_routed`/`request_queued` باید دوباره اضافه شوند.
4. Placement اولیه‌ی هر سرویس بر اساس centroid سرورهای انتخاب‌شده است، نه
   BTSهای واقعی آن سرویس.
5. کیفیت تصمیمات PPO (correctness ~۴۰-۵۳٪ برای SCALE_UP/DOWN) هنوز به
   heuristics نرسیده؛ `analyze_decision_quality.py` نشان می‌دهد بخشی از این
   ممکن است رفتار پیش‌بینانه‌ی موجّه باشد نه نویز خالص — قبل از قضاوت نهایی
   حتماً این ابزار را روی `outputs/ppo_events.jsonl` واقعی اجرا کنید.
6. `k8s_adapter/` (فاز ۳) هنوز شروع نشده.

## ساختار پروژه

```
eotch/
  common/{config,models,metrics,geo,logger,state_builder}.py
  data/loader.py
  simulator/{engine,events}.py
  algorithms/
    base.py
    greedy/greedy_algorithm.py
    voila/voila_algorithm.py
    hpa/hpa_algorithm.py
    ppo/{env,policy_network,train,infer,ppo_algorithm}.py
  analyze_decision_quality.py
  evaluation/compare_runs.py
  run.py
```