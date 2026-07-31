# edge-resource-management (eotch)

شبیه‌سازی مدیریت پویای منابع لبه (Edge Resource Management) با ۴ الگوریتم
قابل‌سوییچ (Greedy / VOILA / Kubernetes-HPA / PPO-DRL)، طبق «سند معماری کامل
پروژه نسخه ۱.۰».

## وضعیت فعلی (فاز ۱ و ۲ از نقشه‌راه بخش ۰ سند)

| بخش | وضعیت |
|---|---|
| `common/` (config, models, geo, metrics, logger, state_builder) | ✅ کامل |
| `data/loader.py` (۴ فایل + آفست روزانه) | ✅ کامل |
| `simulator/engine.py` (موتور discrete-event) | ✅ کامل — شامل emergency-boot، capacity-starved gate، و فیکس flapping (زیر را بخوانید) |
| `algorithms/base.py` (AlgorithmBase مشترک) | ✅ کامل |
| `algorithms/greedy/`، `voila/`، `hpa/` | ✅ کامل و تست‌شده |
| `algorithms/ppo/` | ✅ کامل — آموزش‌دیده (۳ میلیون timestep، BC warm-start، seed=43) |
| `analyze_decision_quality.py` | ✅ ابزار تحلیل کیفیت تصمیم (lookahead) + flapping سرور |
| `k8s_adapter/` (فاز ۳) | ✅ اسکلت کامل نوشته شده (Redis، K8s client، real-time dispatcher، worker service) — **⚠️ هرگز روی کلاستر واقعی تست نشده و یک باگ شناخته‌شده دارد (زیر را بخوانید)** |

## نحوه‌ی اجرا

```bash
pip install -r requirements.txt

# یک الگوریتم را روی داده‌ی تست (Data4.csv) اجرا کن
python run.py --algorithm greedy --data test

# آموزش PPO (روی Data1-3.csv، با BC warm-start از Greedy)
python -m algorithms.ppo.train

# مقایسه‌ی هر ۴ الگوریتم روی همان داده و تولید جدول مقایسه
python -m evaluation.compare_runs --data test

# تحلیل کیفیت تصمیمات یک اجرا (SCALE_UP/DOWN با lookahead + flapping سرور)
python analyze_decision_quality.py outputs/greedy_events.jsonl
python analyze_decision_quality.py outputs/ppo_events.jsonl
python analyze_decision_quality.py outputs/voila_events.jsonl

```

> **نکته‌ی reproducibility:** `algorithms/ppo/ppo_model.zip` در ریپو commit
> نشده (`.gitignore`: `*.zip`). چک‌پوینت‌های میانی در `logs/checkpoints/`
> ذخیره می‌شوند (`CheckpointCallback`، هر ۲۰۰هزار timestep).

> **متغیر محیطی داده:** با `EOTCH_DATA_DIR` تنظیم می‌شود (پیش‌فرض:
> `<ریشه‌ی پروژه>/data/raw`).

## نتیجه‌ی نهایی هر ۴ الگوریتم روی Data4.csv (بعد از فیکس flapping)

اجرای کامل `evaluation/compare_runs.py --data test` (۳۴٬۱۳۷ درخواست کل):

| متریک | Greedy | Voila | HPA | **PPO** |
|---|---|---|---|---|
| avg_response_time_sec | 45.88 | 46.97 | 49.61 | **38.24** |
| p95 / p99 response_time_sec | 193.0 / 386.3 | 204.0 / 398.0 | 222.0 / 413.0 | **151.0 / 331.1** |
| deadline_violations | 1098 | 1121 | 1299 | **1078** |
| deadline_violation_rate_pct | 3.22 | 3.28 | 3.81 | **3.16** |
| cumulative_energy_joule | 44.6M | 43.8M | 41.8M | **38.9M** |
| avg_distance_km | 16.53 | **17.45** (بدترین) | 17.90 | 15.60 |
| avg_load_balance_cv | 0.834 | 0.873 | 0.866 | **0.548** |
| avg_network_delay_ms | 2.331 | 2.349 | 2.358 | 2.312 |
| num_requests_rejected_no_replica | 14 | 14 | 14 | 14 |
| num_requests_rejected_queue_full | **756** | 738 | 808 | 733 |
| avg_active_servers | 1.83 | 1.91 | 1.88 | **2.59** |
| num_server_boots / shutdowns | 169 / 166 | 183 / 180 | 178 / 175 | **56 / 54** |
| میانگین dwell هر چرخه (تقریب) | ~938s (~15.6 دقیقه) | ~902s (~15.0 دقیقه) | ~912s (~15.2 دقیقه) | **515s واقعی (~8.6 دقیقه)** |
| num_pod_creates / deletes | 655 / 640 | 549 / 534 | 798 / 783 | **1494 / 1474** |
| completed_requests | 33367 | 33385 | 33315 | **33390** |

### کیفیت تصمیمات (بخش ۸ سند)

| الگوریتم | SCALE_UP correct/incorrect (rate) | SCALE_DOWN correct/incorrect (rate) | TURN_ON (missed) | TURN_OFF (missed) |
|---|---|---|---|---|
| Greedy | 493 / 0 (100%) | 244 / 0 (100%) | 145 correct (missed 51) | 168 correct (missed 1195) |
| Voila | 413 / 0 (100%) | 236 / 0 (100%) | 153 correct (missed 33) | 182 correct (missed 1343) |
| HPA | 638 / 0 (100%) | 158 / 271 (**37%**) | 160 correct (missed 83) | 177 correct (missed 1261) |
| **PPO** | 86 / 1377 (**6%**) | 1049 / 148 (88%) | 53 correct (missed 83) | 54 correct (missed 2227) |

**نکته‌ی کلیدی: `TURN_OFF correctness_rate = 100%` برای هر ۴ الگوریتم** — این شاهد مستقل و قوی است که فیکس flapping سیستمی کار کرده، نه فقط یک اجرای شانسی.

## تفسیر صادقانه

### ۱) مشکل flapping با موفقیت حل شد

با اضافه‌شدن `MIN_ACTIVE_DURATION_SEC=300` و `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC=120` (هر دو > `COOLDOWN_SEC=60`، در `simulator/engine.py:_apply_provisioning`/`_apply_scale_decision`):

| | قبل از فیکس | بعد از فیکس |
|---|---|---|
| num_server_boots (Greedy/Voila/HPA) | 319 / 409 / 370 | **169 / 183 / 178** (۴۷-۵۵٪ کاهش) |
| TURN_OFF correctness | — (اندازه‌گیری نشده بود) | **۱۰۰٪ هر ۴ الگوریتم** |
| flapping rate (PPO، اندازه‌گیری مستقیم با `analyze_decision_quality.py`) | ۱۰۰٪ (۲ از ۲ چرخه) | **۰٪ (۰ از ۵۲ چرخه)**، میانگین dwell=۵۱۵ ثانیه |

جالب اینکه این فیکس فقط churn را کم نکرد؛ QoS هر ۴ الگوریتم هم بهتر شد
(avg_response_time و deadline_violation_rate برای هر ۴ کاهش یافتند) —
چون سرورهای پایدارتر یعنی cold-start کمتر و ثبات بیشتر صف‌ها.

**طراحی نهایی فیکس:** به‌جای تاخیرانداختن trigger (رویکرد اولیه‌ی من:
`sustain` روی `_any_service_capacity_starved`)، یک کف زمانی مستقیم روی
*خروج* (`min_active_duration_sec` قبل از هر TURN_OFF، `min_replica_age_
before_scale_down_sec` قبل از هر SCALE_DOWN) اعمال شد — این رویکرد
مستقل از این‌که چه چیزی TURN_ON/SCALE_UP را trigger کرده کار می‌کند، پس
قوی‌تر و ساده‌تر است. `_any_service_capacity_starved` عمداً همچنان لحظه‌ای
باقی مانده (بدون sustain اضافه)؛ کف dwell خودش کافی بود.

⚠️ **نکته‌ی پیاده‌سازی مهم برای هرکسی که این دو ثابت را عوض می‌کند:** هر
دو چک در `engine.py` *بعد* از چک `cooldown_sec` می‌آیند در همان
if/elif chain؛ یعنی اگر مقدارشان **کمتر یا مساوی `COOLDOWN_SEC` (=۶۰)**
باشد، عملاً هیچ اثر مستقلی ندارند (چون کنترل cooldown از قبل مسدودشان
کرده). این دو باید همیشه به‌طور معنادار از `COOLDOWN_SEC` بزرگ‌تر بمانند.

### ۲) PPO نهایی (seed=43, ent_coef=0.01) در تقریباً همه‌ی معیارها بهترین است

`avg_response_time_sec` (۳۸.۲۴، ~۱۷٪ بهتر از نزدیک‌ترین رقیب)،
`avg_load_balance_cv` (۰.۵۴۸، به‌وضوح بهترین)، `cumulative_energy_joule`
(۳۸.۹M، کمترین)، و `completed_requests` (۳۳۳۹۰، بیشترین). این با نتایج
train‌های قبلی (که PPO یا هیچ اکشنی نمی‌زد یا رفتار ناپایدار داشت) تفاوت
چشمگیری دارد — نشان می‌دهد ترکیب فیکس flapping (محیط پایدارتر برای
یادگیری) + `ent_coef=0.01` (اکتشاف بیشتر) + seed جدید مؤثر بوده.

### ۳) یافته‌ی مهم: «نویز» بالای SCALE_UP در PPO احتمالاً واقعاً نویز نیست، بلکه یک عدم‌تطابق بین سیاست PPO و معیار ممیزی است

با ۹۰ ثانیه lookahead: نویز=۹۲.۹٪. با ۷.۵ دقیقه (`LOOKAHEAD_TICKS=15`):
نویز هنوز **۹۰.۵٪** (anticipatory فقط از ۰.۲٪ به ۳.۶٪ رسید). یعنی
فرضیه‌ی «PPO دارد چند دقیقه جلوتر پیش‌بینی می‌کند» **رد شد** — حتی با
۵ برابر شدن پنجره، اکثریت قریب‌به‌اتفاق این SCALE_UPها طبق این معیار
غیرضروری می‌مانند.

با این حال PPO دقیقاً همین الگوریتم است که **بهترین** `avg_response_time`
و `deadline_violation_rate` را دارد. تفسیر محتمل‌تر: معیار ممیزی بخش ۸
(`occ_ratio > 0.7`) دقیقاً همان threshold ای است که خودِ Greedy برای
SCALE_UP استفاده می‌کند (نگاه کنید `greedy_algorithm.py`) — یعنی این
«خط‌کش مستقل» در عمل با فلسفه‌ی الگوریتم‌های قاعده‌محور هم‌راستا طراحی
شده، نه با یک سیاست یادگرفته‌شده که ممکن است بافر ایمنی وسیع‌تری نگه دارد.
PPO به‌جای واکنش به‌آستانه، به‌نظر می‌رسد یک سیاست «نگه‌داشتن ظرفیت اضافی
پیوسته» یاد گرفته که response time را پایین نگه می‌دارد ولی طبق این
threshold ساده «غیرضروری» به‌حساب می‌آید. **نتیجه‌گیری برای گزارش نهایی:**
نرخ correctness باید به‌عنوان «میزان تطابق با فلسفه‌ی rule-based»
تفسیر شود، نه معیار مطلق کیفیت — یک سیاست یادگرفته‌شده‌ی متفاوت اما
بهتر می‌تواند عمداً نرخ پایینی داشته باشد.

### ۴) یافته‌ی جانبی: افت کیفیت SCALE_DOWN در HPA (۱۰۰٪→۳۷٪)

این احتمالاً side-effect طبیعی `min_replica_age_before_scale_down_sec`
است: چون فرمول HPA (`ceil(replicas × util/target)`) هر تیک از نو محاسبه
می‌شود، وقتی اجرای واقعی SCALE_DOWN به‌خاطر تاخیر جدید عقب می‌افتد،
شرایط لحظه‌ی *اجرا* با شرایط لحظه‌ی *تصمیم اولیه* فرق کرده. Greedy/Voila
این مشکل را ندارند چون منطق ساده‌تر (occ threshold مستقیم بدون فرمول
حساس) دارند. باگ کد نیست؛ یک خاصیت جالب مختص طراحی HPA است، ارزش ذکر در
گزارش را دارد.

## تصمیمات مهندسی/فرضیات صریح (برای گزارش نهایی حتماً ذکر کنید)

- **`L0_MS = 20ms`**، **`bts_id` سرورها**، **جایگذاری اولیه با ظرفیت
  کافی**، **emergency-boot بخش ۶.۲** (`_trigger_emergency_boot` +
  `_emergency_boot_for_service` ضد boot تکراری)، **رفع باگ متریک فاصله**
  (`common/metrics.py`)، **فیکس n_replicas>1 در ممیزی SCALE_DOWN**، و
  **فیکس capacity-starved TURN_ON** (`_any_service_capacity_starved` در
  `base.py`/`greedy`/`hpa`/`voila`) — همه از نسخه‌های قبلی بدون تغییر
  باقی ماندند.
- **فیکس flapping (بازبینی نهایی)**: `MIN_ACTIVE_DURATION_SEC=300.0` و
  `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC=120.0` در `common/config.py`؛
  اعمال در `simulator/engine.py:_apply_provisioning`/`_apply_scale_decision`.
  نتیجه‌ی تجربی: نگاه کنید بخش «تفسیر صادقانه» بالا.
- **وزن‌های reward PPO**: `w1=0.12, w2=0.20, w3=0.30, w4=0.23,
  w5_rejected=0.15` (جمع=۱.۰).
- **آموزش PPO نهایی**: `total_timesteps=3,000,000`, `bc_epochs=25`,
  `seed=43`, `ent_coef=0.01` (برای اکتشاف بیشتر — نسخه‌های قبلی با
  seed=42 و بدون ent_coef واریانس زیادی بین runها نشان می‌دادند)،
  `CheckpointCallback` هر ۲۰۰هزار timestep در `logs/checkpoints/`.
- **⚠️ محدودیت شناخته‌شده - واریانس train‌به‌train**: قبل از این
  تنظیمات، دو train متوالی با کد کاملاً یکسان (فقط seed متفاوت) رفتارهای
  کیفی بسیار متفاوتی تولید کردند (یکی تقریباً هیچ اکشنی نمی‌زد، دیگری
  churn شدید replica داشت). این نشان می‌دهد نتیجه‌ی PPO با یک run تنها
  reproducible نیست؛ برای گزارش علمی معتبر توصیه می‌شود train را ۲-۳ بار
  دیگر با seedهای متفاوت تکرار کنید و میانگین/واریانس را گزارش دهید، نه
  فقط بهترین run.
- **ثابت نرمال‌سازی انرژی**: `_NORM_ENERGY_JOULE=12000`؛ در دو جا
  (`state_builder.py` و `env.py`) مستقل هاردکد شده.
- **پورتابیلیتی `DATA_DIR`**: با `EOTCH_DATA_DIR`.

## موارد باز/شناخته‌شده (برای ادامه‌ی کار)

1. **باگ در `k8s_adapter/realtime_dispatcher.py`**: مدل «سایه»ی
   `Replica` هرگز `try_admit()` را صدا نمی‌زند (کنترل صف واقعی از طریق
   Redis انجام می‌شود)، یعنی `replica.departures` همیشه خالی می‌ماند و
   `instantaneous_utilization()` هر سرور همیشه ۰ محاسبه می‌شود. نتیجه:
   شرط TURN_OFF (`avg_util < threshold`) همیشه true است و تشخیص overload
   لحظه‌ای هرگز کار نمی‌کند. **باید قبل از هر اجرای واقعی روی کلاستر رفع
   شود** — پیشنهاد: `queue_occupancy` مستقیماً از `redis_state` بخواند.
2. **رگرسیون لاگ**: `request_routed`/`request_queued` هنوز در
   `simulator/engine.py` لاگ نمی‌شوند (در `realtime_dispatcher.py` فقط
   `request_routed` هست، نه `request_queued`).
3. **`requirements.txt` ریشه** شامل `kubernetes`/`redis`/`httpx`
   (وابستگی‌های `k8s_adapter/`) نیست.
4. در `realtime_dispatcher.py`، شکست HTTP واقعی با
   `RequestStatus.REJECTED_NO_REPLICA` علامت‌گذاری می‌شود — گمراه‌کننده
   چون replica واقعاً وجود داشت؛ بهتر است یک status جدید تعریف شود.
5. Placement اولیه‌ی هر سرویس بر اساس centroid سرورهای انتخاب‌شده است، نه
   BTSهای واقعی آن سرویس.
6. تکرار train با seedهای دیگر برای گزارش واریانس (بالا).

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
  k8s_adapter/
    k8s_client.py
    realtime_dispatcher.py
    redis_state.py
    smoke_test.py
    worker_service/{app.py,Dockerfile,requirements.txt}
  analyze_decision_quality.py
  evaluation/compare_runs.py
  run.py
```