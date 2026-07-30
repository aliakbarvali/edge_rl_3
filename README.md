# edge-resource-management (eotch)

شبیه‌سازی مدیریت پویای منابع لبه (Edge Resource Management) با ۴ الگوریتم
قابل‌سوییچ (Greedy / VOILA / Kubernetes-HPA / PPO-DRL)، طبق «سند معماری کامل
پروژه نسخه ۱.۰».

## وضعیت فعلی (فاز ۱ و ۲ از نقشه‌راه بخش ۰ سند)

| بخش | وضعیت |
|---|---|
| `common/` (config, models, geo, metrics, logger, state_builder) | ✅ کامل |
| `data/loader.py` (۴ فایل + آفست روزانه) | ✅ کامل |
| `simulator/engine.py` (موتور discrete-event) | ✅ کامل (شامل emergency-boot بخش ۶.۲ و لاگ کامل بخش ۱۲) |
| `algorithms/base.py` (AlgorithmBase مشترک) | ✅ کامل |
| `algorithms/greedy/` | ✅ کامل و تست‌شده |
| `algorithms/voila/` | ✅ کامل و تست‌شده |
| `algorithms/hpa/` | ✅ کامل و تست‌شده |
| `algorithms/ppo/` (env, train, infer, ppo_algorithm) | ✅ کامل — **آموزش با BC warm-start + fine-tune واقعاً روی داده اجرا شد** (نتایج زیر) |
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
```

> **نکته‌ی reproducibility:** `algorithms/ppo/ppo_model.zip` (چک‌پوینت آموزش‌دیده)
> در ریپو commit نشده، چون `.gitignore` قانون `*.zip` دارد. برای بازتولید نتایج
> PPO زیر، اول `python -m algorithms.ppo.train` را اجرا کنید (یا اگر می‌خواهید
> مدل را هم نسخه‌بندی کنید، این فایل را از قانون gitignore استثنا بگذارید یا
> از Git LFS/GitHub Release استفاده کنید).

> **متغیر محیطی داده:** مسیر داده‌ها دیگر هاردکد نیست؛ با `EOTCH_DATA_DIR`
> تنظیم می‌شود (پیش‌فرض: `<ریشه‌ی پروژه>/data/raw`):
> ```bash
> export EOTCH_DATA_DIR=/path/to/data      # لینوکس/مک
> set EOTCH_DATA_DIR=D:\path\to\data       # ویندوز (cmd)
> ```

## نتیجه‌ی واقعی هر ۴ الگوریتم روی Data4.csv

اجرای کامل `evaluation/compare_runs.py --data test` (۳۴٬۱۳۷ درخواست کل):

| متریک | Greedy | Voila | HPA | **PPO** |
|---|---|---|---|---|
| avg_response_time_sec | 51.95 | 51.91 | 52.70 | **50.40** |
| p95_response_time_sec | 240.00 | 239.01 | 242.25 | 242.00 |
| deadline_violation_rate_pct | 5.11 | 5.20 | 5.54 | 5.35 |
| deadline_violations | 1743 | 1776 | 1890 | 1828 |
| cumulative_energy_joule | 32.2M | **32.1M** | 35.7M | 34.0M |
| avg_distance_km | 17.43 | 21.33 | **17.34** | 18.49 |
| avg_load_balance_cv | 0.307 | **0.285** | 0.584 | 0.369 |
| avg_network_delay_ms | 2.349 | 2.427 | 2.347 | 2.370 |
| num_requests_rejected_no_replica | 14 | 14 | 14 | 14 |
| num_requests_rejected_queue_full | 1114 | 1125 | 1227 | 1145 |
| avg_active_servers | 2.10 | 1.99 | **1.15** | 2.08 |
| num_server_boots / shutdowns | 51 / 49 | 39 / 36 | **22 / 20** | 13 / 9 |
| num_pod_creates / deletes | 268 / 253 | 230 / 215 | 141 / 126 | 128 / 102 |
| completed_requests | 33009 | 32998 | 32896 | 32978 |

### کیفیت تصمیمات (بخش ۸ سند: correctness مستقل از منطق داخلی هر الگوریتم)

| الگوریتم | SCALE_UP correct/incorrect (rate) | SCALE_DOWN correct/incorrect (rate) | TURN_ON correct | TURN_OFF correct (missed) |
|---|---|---|---|---|
| Greedy | 240 / 0 (100%) | 167 / 0 (100%) | 48 / 0 (100%) | 50 (missed 927) |
| Voila | 199 / 0 (100%) | 146 / 0 (100%) | 36 / 0 (100%) | 37 (missed 857) |
| HPA | 113 / 0 (100%) | 25 / 56 (31%) | 19 / 0 (100%) | 21 (missed 314) |
| **PPO** | 41 / 62 (**40%**) | 22 / 29 (**43%**) | 10 / 0 (100%) | 10 (missed **1395**) |

## تفسیر صادقانه (به‌روزرسانی‌شده پس از رفع باگ متریک فاصله)

- **باگ متریک رفع شد:** نسخه‌ی قبلی این README ادعا می‌کرد `avg_distance_km`
  Greedy (۱۸.۱۵) از Voila (۲۰.۶۴) کمتر است چون Greedy replica بیشتری ساخته.
  این ادعا تا حدی نتیجه‌ی یک باگ بود: `common/metrics.py` قبلاً برای
  درخواست‌های **رد‌شده** هم مقدار `distance_km=0.0`/`network_delay_ms=0.0`
  را در میانگین لحاظ می‌کرد (چون این دو فیلد فقط برای درخواست‌های پذیرفته‌شده
  واقعاً محاسبه می‌شوند)، که میانگین را به نسبت تعداد رد هر الگوریتم به‌طور
  مصنوعی پایین می‌آورد. بعد از رفع باگ (append فقط برای `COMPLETED`، مثل
  `response_times`)، عدد Greedy=17.43 و Voila=21.33 است — یعنی Greedy واقعاً
  و به‌طور واقعی هم replica نزدیک‌تری به BTSها می‌سازد (چون replica بیشتری
  دارد)، نه فقط به‌خاطر باگ. HPA با کمترین تعداد سرور فعال (۱.۱۵ میانگین) و
  کمترین replica (141) در عمل نزدیک‌ترین میانگین فاصله را دارد (۱۷.۳۴) که
  نشان می‌دهد `avg_distance_km` به‌تنهایی هنوز معیار کاملی برای کیفیت
  placement نیست؛ چون HPA کاملاً latency-unaware است و این عدد پایین صرفاً
  از تراکم جغرافیایی سرورهای باقی‌مانده می‌آید، نه هوشمندی مکان‌یابی.
- **Voila** کمترین `avg_load_balance_cv` (بار متوازن‌تر بین سرورها) و کمترین
  انرژی مصرفی را دارد — دقیقاً مطابق فلسفه‌ی location-aware مقاله.
- **HPA** چون کاملاً latency-unaware است (بخش ۳ مقدمه‌ی مقاله‌ی Voila)،
  کمترین churn زیرساخت (کمترین boot/shutdown) را دارد، ولی بدترین
  `avg_load_balance_cv` (۰.۵۸۴) و بیشترین انرژی و نقض deadline را — یعنی با
  کمترین سرور فعال، بار را نامتوازن روی همان چند سرور متمرکز می‌کند.
- **PPO** بهترین `avg_response_time_sec` خام (۵۰.۴۰) را دارد، ولی جدول
  «کیفیت تصمیمات» نشان می‌دهد این عدد رایگان نیست: نرخ درستی SCALE_UP/DOWN
  آن (~۴۰-۴۳٪) به‌مراتب پایین‌تر از سه الگوریتم قاعده‌محور (که همه ۱۰۰٪
  یا نزدیک آن‌اند، چون خودِ threshold آن‌ها با معیار ممیزی این پروژه هم‌راستا
  طراحی شده) است، و ۱۳۹۵ فرصت TURN_OFF را از دست داده (در مقابل ۹۲۷ برای
  Greedy و ۸۵۷ برای Voila) — یعنی با وجود gate سراسری sustain، عامل هنوز
  گرایش به نگه‌داشتن سرور اضافی دارد. رفع باگ نرمال‌سازی reward (نگاه کنید
  `algorithms/ppo/env.py` CHANGELOG بازبینی ۳) این را بهبود داد (قبل از آن
  عامل تقریباً هیچ اکشنی نمی‌زد)، ولی هنوز فاصله‌ی محسوسی تا سیاست heuristic
  دارد؛ برای گزارش نهایی این را به‌عنوان یک محدودیت شناخته‌شده ذکر کنید، نه
  یک برتری کامل PPO.
- **رد به‌دلیل نبود replica (`num_requests_rejected_no_replica=14`) در هر ۴
  الگوریتم دقیقاً برابر است** — این تصادفی نیست: این ۱۴ رد همگی در پنجره‌ی
  مشترک `_initial_placement` (پیش از هر تصمیم الگوریتم‌محور، وقتی سرورهای
  اولیه هنوز BOOTING‌اند) رخ می‌دهند و منطق initial placement بین هر ۴
  الگوریتم مشترک است (`AlgorithmBase.initial_placement`/`select_replica`)؛
  تفاوت واقعی بین الگوریتم‌ها در `num_requests_rejected_queue_full` دیده
  می‌شود (Greedy=1114 تا HPA=1227).

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
- **Emergency-boot بخش ۶.۲**: وقتی migration یک سرویس تک‌رپلیکای در حال
  drain نتواند مقصد ACTIVE مناسبی پیدا کند، به‌جای لغو کامل drain (رفتار
  نسخه‌ی قبلی)، الان یک سرور OFF جدید بلافاصله Boot می‌شود («Provisioning
  اضطراری» طبق متن دقیق سند) و migration واقعی به‌محض ACTIVE شدن آن سرور
  کامل می‌شود (`simulator/engine.py:_start_server_drain` /
  `_handle_boot_done`، رویداد لاگ `emergency_boot_triggered`). فقط اگر
  اصلاً هیچ سرور OFFی موجود نباشد (همه‌ی ۱۰ سرور از قبل ACTIVE/BOOTING/
  DRAINING‌اند)، drain همچنان طبق محافظ قبلی این چرخه لغو می‌شود.
- **لاگ ساخت‌یافته‌ی کامل بخش ۱۲**: علاوه بر رویدادهای قبلی، `request_routed`
  (لحظه‌ی انتخاب replica توسط Router، پیش از admit به صف) و `request_queued`
  (لحظه‌ی ورود واقعی به صف FIFO) هم اضافه شدند تا هر ۹ رویداد اجباری بخش ۱۲
  پوشش داده شوند.
- **رفع باگ متریک فاصله/تاخیر شبکه**: `avg_distance_km` و
  `avg_network_delay_ms` قبلاً برای درخواست‌های رد‌شده هم مقدار صفر را در
  میانگین append می‌کردند (چون این فیلدها فقط برای درخواست‌های پذیرفته‌شده
  واقعاً محاسبه می‌شوند)؛ الان مثل `response_times` فقط برای `COMPLETED`
  append می‌شوند (`common/metrics.py`).
- **پورتابیلیتی `DATA_DIR`**: به‌جای مسیر مطلق هاردکد، از `EOTCH_DATA_DIR`
  (با fallback به `<ریشه‌ی پروژه>/data/raw`) استفاده می‌شود.
- **utilization برای provisioning**: `used_cpu/capacity` بر اساس رپلیکاهای
  *در حال پردازش* در لحظه (نه صرفاً deploy‌شده)، دقیقاً طبق فرمول بخش ۲.۴.
  (نکته‌ی باز: این «لحظه‌ای» است، نه میانگین متحرک روی `MONITOR_WINDOW_SEC`
  که متن دقیق بخش ۶.۱ سند پیشنهاد می‌دهد؛ تداوم چند-تیکی `SUSTAIN_HIGH/LOW_SEC`
  تا حدی همین اثر را می‌دهد ولی معادل ریاضی EMA/میانگین متحرک نیست.)
- **مرکز ثقل تقاضا برای Voila**: چون AlgorithmBase استاندارد به snapshot خام
  موقعیت هر درخواست دسترسی نمی‌دهد، `simulator/engine.py` یک میانگین متحرک
  نمایی (EMA, α=۰.۳) از موقعیت جغرافیایی درخواست‌های اخیر هر سرویس را در
  `metrics_snapshot["services"][sid]["demand_centroid"]` نگه می‌دارد؛ Voila
  از این برای انتخاب مکان replica/migration نزدیک به تقاضای واقعی استفاده
  می‌کند (برخلاف Greedy/HPA که فقط بر اساس موقعیت خودِ سرورها تصمیم می‌گیرند).
- **آستانه‌های heuristic هر الگوریتم** (`OCC_UP_THRESHOLD`, `TARGET_UTILIZATION`
  و مشابه) در همان فایل الگوریتم نگه داشته شدند، نه `config.py`، چون این‌ها
  سیاست تصمیم‌گیری مختص هر الگوریتم‌اند، نه قید فیزیکی سیستم.
- **وزن‌های reward PPO (بازبینی ۳)**: `w1_response_time=0.12, w2_deadline=0.20,
  w3_energy=0.30, w4_load_balance=0.23, w5_rejected=0.15` (جمع=۱.۰،
  `common/config.py`). قبلاً جریمه‌ی «درخواست ردشده» جدا و نرمال‌نشده اعمال
  می‌شد و می‌توانست ۵-۱۵ برابر بقیه‌ی اجزای reward بزرگ‌تر شود و کل سیگنال را
  تحت‌الشعاع قرار دهد (عامل یاد گرفته بود عملاً هیچ اکشنی نزند)؛ الان
  `num_rejected_recent` هم مثل بقیه نرمال (کلمپ در `_NORM_REJECTED_PER_TICK=5`)
  و با وزن صریح `w5_rejected` ترکیب می‌شود (`algorithms/ppo/env.py`).
- **ثابت نرمال‌سازی انرژی**: `_NORM_ENERGY_JOULE=12000` (کالیبره‌شده روی
  میانگین/صدک۹۰ واقعی انرژی هر تیک با Greedy روی Data4.csv؛ مقدار اولیه‌ی
  ۵۰۰۰۰ خیلی بزرگ بود و جریمه‌ی انرژی را عملاً بی‌اثر می‌کرد). این عدد فعلاً
  در دو جا (`common/state_builder.py` و `algorithms/ppo/env.py`) به‌طور
  مستقل هاردکد شده — برای گزارش نهایی بهتر است یکی import کند تا یک منبع
  حقیقت باشد.

## موارد باز/شناخته‌شده (برای ادامه‌ی کار)

- Placement اولیه‌ی هر سرویس (`_nearest_capable_server`) بر اساس نزدیکی به
  **centroid سرورهای انتخاب‌شده** است، نه نزدیکی به BTSهای واقعی آن سرویس
  در پنجره‌ی اول؛ ممکن است تفسیر دقیق‌تری از سند («نزدیک‌ترین سرور فعال»)
  باشد ولی اگر هدف نزدیکی به تقاضای واقعی است، باید فیلتر شود.
- کیفیت تصمیمات PPO (نرخ درستی SCALE_UP/DOWN ~۴۰٪، missed TURN_OFF بالا)
  هنوز به سیاست heuristic نرسیده؛ کاندید بهبود بعدی برای fine-tuning.
- `k8s_adapter/` (فاز ۳) هنوز شروع نشده.

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