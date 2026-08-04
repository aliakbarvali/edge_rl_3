# edge_rl_3 — مدیریت پویای منابع محاسباتی لبه (Edge Resource Management)

شبیه‌سازی و مقایسه‌ی چهار الگوریتم مدیریت منابع در محیط edge computing —
**Greedy**، **VOILA**، **Kubernetes-HPA** و **PPO-DRL** — روی داده‌ی واقعی
ترافیک شبکه‌ی موبایل (BTS)، با معماری قابل‌سوییچ که امکان اجرای همان منطق
تصمیم‌گیری هم در یک موتور شبیه‌سازی discrete-event و هم روی یک کلاستر
Kubernetes واقعی را فراهم می‌کند.

---

## فهرست

- [این پروژه چه کاری انجام می‌دهد](#این-پروژه-چه-کاری-انجام-میدهد)
- [ایده‌ی کلی سیستم](#ایدهی-کلی-سیستم)
- [چهار الگوریتم](#چهار-الگوریتم)
- [ساختار پروژه](#ساختار-پروژه)
- [نصب](#نصب)
- [آماده‌سازی داده](#آمادهسازی-داده)
- [اجرای شبیه‌سازی](#اجرای-شبیهسازی)
- [آموزش عامل PPO](#آموزش-عامل-ppo)
- [مقایسه‌ی هر چهار الگوریتم](#مقایسهی-هر-چهار-الگوریتم)
- [معیارهای خروجی](#معیارهای-خروجی)
- [ابزار تحلیل کیفیت تصمیم](#ابزار-تحلیل-کیفیت-تصمیم)
- [اجرای واقعی روی Kubernetes (فاز ۳)](#اجرای-واقعی-روی-kubernetes-فاز-۳)
- [پیکربندی](#پیکربندی)
- [افزودن یک الگوریتم جدید](#افزودن-یک-الگوریتم-جدید)

---

## این پروژه چه کاری انجام می‌دهد

فرض کنید ۱۰ سرور لبه (edge server) در نقاط مختلف یک شهر قرار دارند و باید
به درخواست‌های ۱۵ نوع سرویس مختلف که از ایستگاه‌های پایه‌ی موبایل (BTS)
می‌رسند پاسخ دهند. سؤال اصلی این است: **چه زمانی سرور روشن/خاموش شود، چند
نمونه (replica) از هر سرویس اجرا شود، و درخواست‌ها به کدام سرور هدایت
شوند** — طوری که هم‌زمان تأخیر پاسخ‌دهی پایین بماند، سررسیدها (deadline)
نقض نشوند، و مصرف انرژی هم کنترل‌شده باشد؟

این پروژه چهار استراتژی متفاوت برای پاسخ به این سؤال پیاده‌سازی می‌کند و
آن‌ها را روی داده‌ی واقعی ترافیک BTS شانگهای، با معیارهای یکسان، مقایسه
می‌کند.

## ایده‌ی کلی سیستم

- **۱۰ سرور ناهمگن** با سه پروفایل ظرفیتی (`edge_small`, `medium`, `large`)
  و موقعیت جغرافیایی ثابت.
- **۱۵ سرویس** با نیاز CPU، زمان اجرا، طول صف و سررسید (deadline) مشخص
  برای هرکدام.
- هر سرور یک ماشین‌حالت دارد: `OFF → BOOTING → ACTIVE → DRAINING → OFF`.
- هر نمونه‌ی سرویس (replica) هم ماشین‌حالت خودش را دارد:
  `STARTING → READY → DRAINING → TERMINATED`.
- هر replica یک **صف FIFO واقعی** با ظرفیت محدود دارد (نه تخمین آماری) و
  هم‌زمان فقط یک درخواست را پردازش می‌کند.
- درخواست‌ها بر اساس **نزدیکی جغرافیایی** (فاصله‌ی هاورسین بین BTS و سرور)
  و در دسترس‌بودن صف مسیریابی می‌شوند.
- مدل تأخیر شبکه: `network_delay_ms = BASE_LATENCY_MS + K_MS_PER_KM × distance_km`
- زمان پاسخ کامل هر درخواست شامل تأخیر شبکه (رفت و برگشت)، زمان انتظار در
  صف، زمان اجرا و جریمه‌ی احتمالی cold-start است.
- هر ۳۰ ثانیه‌ی شبیه‌سازی، یک «تیک تصمیم» رخ می‌دهد که در آن الگوریتم فعال
  سه نوع تصمیم می‌گیرد:
  1. **Auto-scaling** — افزایش/کاهش تعداد replica هر سرویس
  2. **Provisioning** — روشن/خاموش‌کردن سرور
  3. **Placement** — انتخاب سرور مقصد برای replica جدید

همه‌ی این منطق مشترک (مدل‌های داده، محاسبه‌ی فاصله/تأخیر، موتور رویداد‌محور،
متریک‌ها) در `common/` و `simulator/` قرار دارد و هیچ‌کدام از چهار الگوریتم
لازم نیست آن را بازنویسی کند — فقط تصمیم‌گیری‌ها فرق می‌کنند.

## چهار الگوریتم

هر الگوریتم پیاده‌سازی‌ای از یک اینترفیس مشترک (`AlgorithmBase`) است:

| الگوریتم | فلسفه |
|---|---|
| **Greedy** | آستانه‌ساده و مکان‌آگاه: بر اساس اشغال صف و نرخ رد شدن تصمیم می‌گیرد؛ placement/migration را بر مبنای نزدیک‌ترین سرور به مرکز سرورهای فعال انتخاب می‌کند. baseline پروژه است. |
| **VOILA** | مبتنی بر مقاله‌ی VOILA؛ placement و migration را بر اساس **مرکز ثقل تقاضای واقعی** هر سرویس (میانگین متحرک موقعیت درخواست‌های اخیر) انتخاب می‌کند، نه صرفاً نزدیک‌ترین به مرکز سرورهای فعال. علاوه بر نقض ظرفیت (queue occupancy)، نقض نزدیکی جغرافیایی (proximity violation) را هم به‌عنوان سیگنال scale-up در نظر می‌گیرد و برای scale-down به یک «streak» چند-تیکی از وضعیت سالم نیاز دارد (ضد نوسان). |
| **HPA** | معادل الگوریتم Kubernetes Horizontal Pod Autoscaler: کاملاً location-unaware، فقط بر اساس نسبت اشغال صف نسبت به یک هدف ثابت (۷۰٪) تعداد replica مطلوب را محاسبه می‌کند. |
| **PPO-DRL** | یک عامل یادگیری تقویتی (Proximal Policy Optimization، با `MaskablePPO` از `sb3-contrib`) که هر سه نوع تصمیم را هم‌زمان از یک بردار حالت ۱۲۲بعدی یاد می‌گیرد. آموزش با **warm-start از دموی Greedy** (Behavior Cloning) شروع می‌شود و سپس با RL روی پاداشی وزن‌دار از زمان پاسخ، نقض سررسید، انرژی، توازن بار و نرخ رد شدن fine-tune می‌شود. |

هر چهار الگوریتم از همان منطق مشترک برای **جای‌گذاری اولیه** (پوشش حریصانه‌ی
BTSهای فعال، مشابه Set-Cover) و **مسیریابی درخواست** (نزدیک‌ترین replica
با صف خالی) استفاده می‌کنند تا مقایسه منصفانه بماند.

## ساختار پروژه

```
edge_rl_3/
├── common/
│   ├── models.py         # Server, Replica, Request و ماشین‌حالت‌ها
│   ├── config.py         # همه‌ی پارامترها و ثابت‌های قابل‌کالیبراسیون
│   ├── geo.py             # فاصله‌ی هاورسین و مدل تأخیر شبکه
│   ├── metrics.py         # جمع‌آوری و محاسبه‌ی معیارهای نهایی
│   ├── logger.py          # لاگ ساخت‌یافته‌ی JSON برای هر رویداد
│   └── state_builder.py   # ساخت بردار state مشترک برای PPO
│
├── data/
│   └── loader.py           # خواندن CSV، فیلتر جغرافیایی، آفست روزانه
│
├── simulator/
│   ├── engine.py            # موتور discrete-event (مبتنی بر heapq)
│   └── events.py            # تعریف انواع رویداد
│
├── algorithms/
│   ├── base.py               # اینترفیس مشترک AlgorithmBase
│   ├── greedy/greedy_algorithm.py
│   ├── voila/voila_algorithm.py
│   ├── hpa/hpa_algorithm.py
│   └── ppo/
│       ├── env.py             # محیط Gymnasium برای آموزش
│       ├── policy_network.py  # معماری شبکه‌ی سیاست/ارزش
│       ├── train.py           # آموزش (BC warm-start + PPO fine-tune)
│       ├── infer.py           # اجرای inference-only مدل آموزش‌دیده
│       └── ppo_algorithm.py   # پیاده‌سازی AlgorithmBase با مدل آموزش‌دیده
│
├── k8s_adapter/                # فاز ۳: اجرای واقعی روی کلاستر Kubernetes
│   ├── k8s_client.py            # ساخت/حذف Deployment، cordon/uncordon نود
│   ├── redis_state.py           # هماهنگی وضعیت لحظه‌ای روی Redis
│   ├── realtime_dispatcher.py   # معادل real-time موتور شبیه‌سازی
│   ├── smoke_test.py            # تست اتصال Redis/K8s قبل از اجرای کامل
│   └── worker_service/          # سرویس FastAPI مینیمال داخل هر pod
│
├── evaluation/
│   ├── compare_runs.py    # اجرای هر ۴ الگوریتم روی داده‌ی یکسان
│   └── aggregate_seeds.py # میانگین/انحراف‌معیار نتایج PPO روی چند seed
│
├── analyze_decision_quality.py  # تحلیل flapping و کیفیت تصمیمات SCALE_UP/DOWN
├── run.py                        # نقطه‌ی ورود اصلی CLI
└── requirements.txt
```

## نصب

```bash
python3 -m venv venv
source venv/bin/activate        # ویندوز: venv\Scripts\activate
pip install -r requirements.txt
```

برای اجرای فاز ۳ (Kubernetes واقعی) به‌طور جداگانه نیاز است:
```bash
pip install kubernetes redis httpx paramiko
```

## آماده‌سازی داده

پروژه روی داده‌ی ترافیک BTS شانگهای کار می‌کند (ستون‌های `id, BTSID, Lat,
Long, ServiceID, startSec`). چهار فایل CSV لازم است:

- `Data1.csv`, `Data2.csv`, `Data3.csv` → داده‌ی آموزش (سه شنبه‌ی متوالی)
- `Data4.csv` → داده‌ی تست/ارزیابی (شنبه‌ی چهارم، مستقل از train)

مسیر پوشه‌ی داده با متغیر محیطی قابل تنظیم است (پیش‌فرض: `<ریشه‌ی
پروژه>/data/raw`):

```bash
export EOTCH_DATA_DIR=/path/to/data      # لینوکس/مک
set EOTCH_DATA_DIR=D:\path\to\data       # ویندوز (cmd)
```

`data/loader.py` هر فایل را به بازه‌ی جغرافیایی/سرویس‌های فعال فیلتر
می‌کند و با `global_start_sec = day_index * 86400 + startSec` یک تایم‌لاین
پیوسته‌ی چندروزه می‌سازد.

## اجرای شبیه‌سازی

اجرای یک الگوریتم روی داده‌ی تست:

```bash
python3 run.py --algorithm greedy --data test
python3 run.py --algorithm voila  --data test
python3 run.py --algorithm hpa    --data test
python3 run.py --algorithm ppo    --data test   # نیاز به مدل آموزش‌دیده دارد
```

آرگومان‌ها:

| آرگومان | مقادیر | توضیح |
|---|---|---|
| `--algorithm` | `greedy` / `voila` / `hpa` / `ppo` | الگوریتم مورد استفاده |
| `--mode` | `sim` (پیش‌فرض) / `k8s` | شبیه‌سازی یا اجرای واقعی روی کلاستر |
| `--data` | `test` (پیش‌فرض) / `train` | مجموعه‌ی داده |
| `--output-dir` | مسیر دلخواه (پیش‌فرض `outputs/`) | محل ذخیره‌ی لاگ‌ها و نتایج |

خروجی هر اجرا:
- `outputs/<algorithm>_events.jsonl` — لاگ ساخت‌یافته‌ی هر رویداد (ورود
  درخواست، مسیریابی، صف، تکمیل/رد، boot/drain سرور، pod create/delete،
  تصمیمات scale/provision)
- `outputs/<algorithm>_result.json` — خلاصه‌ی معیارهای نهایی

## آموزش عامل PPO

```bash
python3 -m algorithms.ppo.train
```

مراحل داخلی:
1. اجرای Greedy روی داده‌ی آموزش و ثبت (state, action) هر تیک به‌عنوان دمو
2. **Behavior Cloning warm-start**: یادگیری تحت‌نظارت سیاست روی دموی Greedy
3. **Fine-tune با RL**: آموزش MaskablePPO با ۸ محیط موازی روی پنجره‌های
   زمانی تصادفی از تایم‌لاین سه‌روزه‌ی train

پارامترهای پیش‌فرض قابل override با آرگومان تابع `main()` هستند (از جمله
`total_timesteps`, `bc_epochs`, `n_envs`, `window_hours`).

seed آموزش با متغیر محیطی قابل تنظیم است تا بتوان چند مدل با seedهای مختلف
آموزش داد:

```bash
EOTCH_SEED=43 python3 -m algorithms.ppo.train
```

خروجی: `algorithms/ppo/ppo_model_seed<N>.zip` + آمار نرمال‌سازی reward +
لاگ‌های `Monitor` (reward-per-episode) و TensorBoard در `logs/`.

برای تماشای منحنی یادگیری:
```bash
tensorboard --logdir logs/tensorboard
```

پس از آموزش، ارزیابی inference-only روی داده‌ی تست:
```bash
python3 -m algorithms.ppo.infer
```

اگر چند seed آموزش داده و هرکدام را با `evaluation.compare_runs` ارزیابی
کرده باشید، می‌توانید میانگین/انحراف‌معیار معیارها را جمع‌بندی کنید:
```bash
python3 -m evaluation.aggregate_seeds --seeds 42 43 44 45 --base-dir outputs
```

## مقایسه‌ی هر چهار الگوریتم

```bash
python3 -m evaluation.compare_runs --data test
```

این اسکریپت هر چهار الگوریتم را (هرکدام که آماده باشد؛ اگر مدل PPO هنوز
آموزش داده نشده باشد، آن یکی با هشدار رد می‌شود نه با خطا) روی داده‌ی یکسان
اجرا می‌کند و جدول مقایسه‌ای در `outputs/comparison_summary.csv` تولید
می‌کند.

## معیارهای خروجی

هر اجرا این معیارها را گزارش می‌دهد:

**کیفیت سرویس**
- `avg/p95/p99_response_time_sec` — زمان پاسخ
- `deadline_violations`, `deadline_violation_rate_pct` — نقض سررسید
- `avg_distance_km` — فاصله‌ی جغرافیایی میانگین درخواست تا سرور پاسخ‌دهنده
- `avg/p95/p99_network_delay_ms` — تأخیر شبکه (یک‌طرفه)
- `num_requests_rejected_queue_full`, `num_requests_rejected_no_replica`

**منابع و انرژی**
- `cumulative_energy_joule` — مجموع انرژی مصرفی (حالت‌های سرور + گذارها)
- `avg_active_servers`
- `avg_load_balance_cv` — ضریب تغییرات (std/mean) بار بین سرورهای فعال

**عملیاتی**
- `num_server_boots`, `num_server_shutdowns`
- `num_pod_creates`, `num_pod_deletes`
- `num_scale_up`, `num_scale_down`, `num_turn_on`, `num_turn_off`

**کیفیت تصمیم (بخش مهم برای مقایسه‌ی منصفانه)**

هر اکشن اعمال‌شده در برابر یک **معیار ممیزی مستقل و یکسان برای هر ۴
الگوریتم** سنجیده می‌شود (نه threshold داخلی خودِ الگوریتم)، تا مشخص شود
چند درصد از تصمیمات واقعاً لازم بودند و چند فرصت واقعی از دست رفته است:

```json
"decision_correctness": {
  "SCALE_UP":  {"correct": ..., "incorrect": ..., "missed_opportunities": ..., "correctness_rate_pct": ...},
  "SCALE_DOWN": {...},
  "TURN_ON":    {...},
  "TURN_OFF":   {...}
}
```

> نکته: این معیار «همسویی با یک قاعده‌ی ثابت» را می‌سنجد نه لزوماً
> «کیفیت مطلق». الگوریتمی که فلسفه‌ی متفاوتی دارد (مثلاً حاشیه‌ی امنیت
> بزرگ‌تر برای QoS بهتر) ممکن است طبق این معیار نمره‌ی پایین‌تری بگیرد
> بدون آنکه واقعاً تصمیمات بدتری گرفته باشد؛ برای تفسیر دقیق‌تر رفتار
> عامل PPO از `analyze_decision_quality.py` استفاده کنید.

## ابزار تحلیل کیفیت تصمیم

```bash
python3 analyze_decision_quality.py outputs/ppo_events.jsonl
```

این اسکریپت روی فایل لاگ JSONL هر الگوریتم دو نوع تحلیل انجام می‌دهد:

1. **طبقه‌بندی SCALE_UP/SCALE_DOWN «غیرضروری»**: با نگاه‌کردن به چند تیک
   بعدی، مشخص می‌کند که آیا یک SCALE_UP زودهنگام واقعاً **پیش‌بینانه**
   بوده (کمی بعد نیاز واقعی پیش آمد) یا صرفاً **نویز**؛ و آیا یک
   SCALE_DOWN زودهنگام **بی‌ضرر** بوده یا واقعاً **مخاطره‌آمیز** (بلافاصله
   کمبود ظرفیت ایجاد کرد).
2. **تحلیل نوسان (flapping) سرور**: چرخه‌های `TURN_ON → TURN_OFF` هر سرور
   را می‌سنجد و آن‌هایی که مدت فعالیت‌شان (`dwell`) کمتر از یک آستانه
   (پیش‌فرض ۳۰۰ ثانیه) بوده را به‌عنوان flapping علامت می‌زند.

## اجرای واقعی روی Kubernetes (فاز ۳)

معماری این پروژه طوری طراحی شده که همان چهار پیاده‌سازی `AlgorithmBase`
(بدون هیچ تغییری در منطق تصمیم‌گیری) روی یک کلاستر Kubernetes واقعی هم قابل
اجرا باشند — فقط لایه‌ی اجرا (`k8s_adapter/`) به‌جای دستکاری آبجکت در
حافظه، واقعاً Deployment می‌سازد/حذف می‌کند و نود cordon/uncordon می‌کند.

پیش‌نیازها:
1. لیبل‌گذاری هر ۱۰ نود worker: `kubectl label node <نام‌نود> edge-server-id=<id>`
2. ساخت namespace: `kubectl create namespace edge-rl`
3. build و push ایمیج worker (`k8s_adapter/worker_service/`):
   ```bash
   python3 build_push_pull_worker.py
   ```
4. دسترسی معتبر `~/.kube/config` به کلاستر، و یک نمونه‌ی Redis در دسترس

پیش از اجرای کامل، حتماً تست اتصال را اجرا کنید:
```bash
python3 -m k8s_adapter.smoke_test
```

سپس اجرای واقعی:
```bash
python3 run.py --algorithm greedy --mode k8s --data test
```

در این حالت، `decision_loop` هر ۳۰ ثانیه‌ی واقعی تصمیمات scale/provision
را اعمال می‌کند و `dispatch_loop` به‌طور موازی، رویدادهای CSV را با
زمان‌بندی واقعی (نه فشرده) به سرویس‌های واقعی مستقر روی worker nodeها
ارسال می‌کند؛ هماهنگی وضعیت لحظه‌ای بین این دو حلقه از طریق Redis انجام
می‌شود.

## پیکربندی

همه‌ی پارامترهای عددی (تأخیرها، آستانه‌ها، وزن‌های reward، ثابت‌های انرژی و
غیره) در یک فایل مرکزی `common/config.py` قرار دارند و هیچ‌کجای دیگر کد
hardcode نشده‌اند. مهم‌ترین‌ها:

| پارامتر | پیش‌فرض | معنا |
|---|---|---|
| `DECISION_INTERVAL_SEC` | ۳۰ | فاصله‌ی هر تیک تصمیم |
| `BOOT_DELAY_SEC` / `POD_STARTUP_DELAY_SEC` | ۳۰ / ۵ | زمان روشن‌شدن سرور / آماده‌شدن replica |
| `UTIL_SCALE_UP_THRESHOLD` / `UTIL_SCALE_DOWN_THRESHOLD` | ۰.۹۵ / ۰.۴۵ | آستانه‌ی provisioning سرور |
| `MIN_ACTIVE_DURATION_SEC` | ۳۰۰ | حداقل مدت فعال بودن سرور قبل از خاموش‌شدن (ضد نوسان) |
| `COOLDOWN_SEC` | ۶۰ | حداقل فاصله بین دو اکشن معکوس روی همان منبع |
| `PPO_REWARD_WEIGHTS` | جمع=۱.۰ | وزن اجزای پاداش PPO (زمان پاسخ، نقض سررسید، انرژی، توازن بار، نرخ رد) |
| `DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD` | ۰.۷ | آستانه‌ی ممیزی مستقل درستی تصمیم |

برای هر اجرا با تنظیمات متفاوت، مقدار مورد نظر را در همین فایل تغییر دهید
یا برای موارد قابل override با متغیر محیطی (`EOTCH_DATA_DIR`,
`EOTCH_SEED`) استفاده کنید.

## افزودن یک الگوریتم جدید

برای افزودن یک استراتژی جدید کافیست کلاسی از `AlgorithmBase`
(`algorithms/base.py`) بسازید و این متدها را پیاده‌سازی کنید:

```python
class MyAlgorithm(AlgorithmBase):
    name = "my_algo"

    def scale_decision(self, service_id, metrics_snapshot) -> ScaleAction: ...
    def provision_decision(self, servers, metrics_snapshot, now) -> ProvisionAction: ...
    def select_placement_server(self, service_id, servers) -> Optional[int]: ...
    def migration_decision(self, draining_server, servers) -> List[MigrationStep]: ...
```

متدهای `initial_placement` و `select_replica` به‌طور پیش‌فرض در
`AlgorithmBase` پیاده‌سازی شده‌اند و بین همه‌ی الگوریتم‌ها مشترک‌اند (قابل
override در صورت نیاز). سپس فقط کافیست الگوریتم را در `run.py` و
`evaluation/compare_runs.py` به تابع سازنده اضافه کنید — موتور شبیه‌سازی و
adapter کلاستر هیچ تغییری نیاز ندارند.
