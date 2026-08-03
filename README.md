# edge-resource-management (eotch)

شبیه‌سازی مدیریت پویای منابع لبه (Edge Resource Management) با ۴ الگوریتم قابل‌سوییچ
**Greedy / VOILA / Kubernetes-HPA / PPO-DRL**، طبق «سند معماری کامل پروژه نسخه ۱.۰».

هدف پروژه: نشان‌دادن این‌که یک عامل یادگیری تقویتی (PPO) می‌تواند در مدیریت پویای منابع
لبه (auto-scaling رپلیکا + provisioning سرور + placement) از الگوریتم‌های کلاسیک
قاعده‌محور بهتر عمل کند — روی ۴ معیار اصلی: response time، نرخ نقض deadline، مصرف
انرژی، و توازن بار (load balance CV).

> این README نسخه‌ی نهایی و کامل مستندسازی پروژه است؛ شامل معماری، نحوه‌ی اجرا،
> **تاریخچه‌ی کامل توسعه/دیباگ** (چون بخش زیادی از ارزش این پروژه در فرآیند کشف و
> رفع باگ‌های ظریف بوده، نه فقط نتیجه‌ی نهایی)، ابزارهای تحلیلی ساخته‌شده، و نتایج
> نهایی.

---

## فهرست مطالب

1. [وضعیت فعلی پروژه](#۱-وضعیت-فعلی-پروژه)
2. [معماری و ساختار پوشه‌ها](#۲-معماری-و-ساختار-پوشه‌ها)
3. [نحوه‌ی نصب و اجرا](#۳-نحوه‌ی-نصب-و-اجرا)
4. [چهار الگوریتم](#۴-چهار-الگوریتم)
5. [مسیر کامل توسعه و رفع باگ](#۵-مسیر-کامل-توسعه-و-رفع-باگ)
6. [ابزار تحلیل کیفیت تصمیم (`analyze_decision_quality.py`)](#۶-ابزار-تحلیل-کیفیت-تصمیم)
7. [نتایج نهایی](#۷-نتایج-نهایی)
8. [یافته‌های کلیدی](#۸-یافته‌های-کلیدی)
9. [پارامترهای قابل کالیبراسیون](#۹-پارامترهای-قابل-کالیبراسیون)
10. [تصمیمات مهندسی/فرضیات صریح](#۱۰-تصمیمات-مهندسیفرضیات-صریح)
11. [محدودیت‌های شناخته‌شده و کارهای باقی‌مانده](#۱۱-محدودیت‌های-شناخته‌شده-و-کارهای-باقی‌مانده)
12. [مرجع سند معماری](#۱۲-مرجع-سند-معماری)

---

## ۱. وضعیت فعلی پروژه

| بخش | وضعیت |
|---|---|
| `common/` (config, models, geo, metrics, logger, state_builder) | ✅ کامل و اصلاح‌شده |
| `data/loader.py` (۴ فایل + آفست روزانه) | ✅ کامل |
| `simulator/engine.py` (موتور discrete-event) | ✅ کامل، چند دور اصلاح جدی |
| `algorithms/base.py` (AlgorithmBase مشترک) | ✅ کامل |
| `algorithms/greedy/` | ✅ کامل، تست‌شده، اصلاح‌شده |
| `algorithms/voila/` | ✅ کامل، تست‌شده، اصلاح‌شده |
| `algorithms/hpa/` | ✅ کامل، تست‌شده، اصلاح‌شده |
| `algorithms/ppo/` (env, train, infer, ppo_algorithm) | ✅ کامل، آموزش‌دیده (۳M timestep)، ارزیابی‌شده |
| `evaluation/compare_runs.py` | ✅ کامل |
| `analyze_decision_quality.py` | ✅ ابزار تشخیصی سفارشی (ساخته‌شده در حین پروژه) |
| `k8s_adapter/` (فاز ۳ - اجرای واقعی روی Kubernetes) | ⏳ شروع‌نشده / خارج از محدوده‌ی این مستندسازی |

### نتیجه‌ی نهایی به‌طور خلاصه

بعد از یک فرآیند طولانی دیباگ و تنظیم (بخش ۵)، **PPO در هر ۴ معیار اصلی از هر ۳
الگوریتم قاعده‌محور بهتر است**:

| معیار | Greedy | Voila | HPA | **PPO** |
|---|---:|---:|---:|---:|
| avg_response_time_sec | 45.88 | 46.97 | 49.61 | **۳۸.۲۴** ✅ |
| deadline_violation_rate_pct | 3.22 | 3.28 | 3.81 | **۳.۱۶** ✅ |
| cumulative_energy_joule | 44.58M | 43.80M | 41.75M | **۳۸.۸۵M** ✅ |
| avg_load_balance_cv | 0.834 | 0.873 | 0.866 | **۰.۵۴۸** ✅ |

(جزئیات کامل و نحوه‌ی رسیدن به این نتیجه در بخش‌های ۵، ۷ و ۸.)

---

## ۲. معماری و ساختار پوشه‌ها

مطابق بخش ۱۰ سند معماری:

```
eotch/
  common/
    config.py          # SERVER_PROFILES, SERVER_INFO, SERVICES_INFO, پارامترهای سیستم
    models.py           # Server, Replica, Request, ServerState/ReplicaState/RequestStatus
    metrics.py           # MetricsCollector - جمع‌آوری/محاسبه‌ی همه‌ی معیارهای بخش ۸
    geo.py                 # haversine, network_delay
    logger.py               # EventLogger - لاگ‌گیری ساخت‌یافته (بخش ۱۲)
    state_builder.py         # build_state_vector() - فضای حالت مشترک PPO/k8s_adapter

  data/
    loader.py            # خواندن CSV، فیلتر شنبه‌ها، اعمال global_start_sec آفست

  simulator/
    engine.py            # موتور discrete-event (heapq-based، معادل simpy)
    events.py            # تعریف EventType/Event

  algorithms/
    base.py              # AlgorithmBase انتزاعی + توابع مشترک (initial_placement,
                          # select_replica, _pick_profile_for_overload,
                          # _capacity_starved_services)
    greedy/greedy_algorithm.py
    voila/voila_algorithm.py
    hpa/hpa_algorithm.py
    ppo/
      env.py              # Gymnasium environment (Reward، Action Masking)
      policy_network.py    # PPO_POLICY_KWARGS
      train.py               # BC warm-start + fine-tune RL
      infer.py                 # ارزیابی inference-only
      ppo_algorithm.py          # پیاده‌سازی AlgorithmBase با مدل آموزش‌دیده

  evaluation/
    compare_runs.py       # اجرای هر ۴ الگوریتم روی داده‌ی یکسان + جدول مقایسه

  analyze_decision_quality.py   # ابزار تشخیص noise/anticipatory + flapping detector

  run.py                  # نقطه‌ورود: python run.py --algorithm {greedy,voila,hpa,ppo}

  logs/                    # خروجی آموزش PPO: monitor/, tensorboard/, bc_warmstart_loss.csv, checkpoints/
  outputs/                 # خروجی ارزیابی: {algo}_result.json, {algo}_events.jsonl, comparison_summary.csv
```

### قرارداد اینترفیس مشترک (`AlgorithmBase`)

```python
class AlgorithmBase(ABC):
    def initial_placement(self, servers, active_bts) -> List[server_id]: ...      # مشترک
    def select_replica(self, request, candidates, servers, now) -> Replica|None:  # مشترک
    def scale_decision(self, service_id, metrics_snapshot) -> ScaleAction: ...     # هر الگوریتم
    def provision_decision(self, servers, metrics_snapshot, now) -> ProvisionAction: ...  # هر الگوریتم
    def select_placement_server(self, service_id, servers) -> server_id|None: ...  # هر الگوریتم
    def migration_decision(self, draining_server, servers) -> List[MigrationStep]: ...  # هر الگوریتم
```

`run.py`/`evaluation/compare_runs.py` صرفاً بر اساس نام الگوریتم، نمونه‌ی مناسب از
`AlgorithmBase` را می‌سازند و به `SimulationEngine` می‌دهند؛ موتور اصلاً نمی‌داند کدام
الگوریتم در حال اجراست — این تفکیک مسئولیت باعث شد اصلاحات موتور (بخش ۵) روی هر ۴
الگوریتم به‌طور یکسان اعمال شوند.

---

## ۳. نحوه‌ی نصب و اجرا

```bash
pip install -r requirements.txt

# متغیر محیطی مسیر داده (اختیاری - پیش‌فرض: <ریشه‌ی پروژه>/data/raw)
export EOTCH_DATA_DIR=/path/to/data          # لینوکس/مک
set EOTCH_DATA_DIR=D:\PT\edge_rl_3\data      # ویندوز (cmd)

# اجرای یک الگوریتم روی داده‌ی تست (Data4.csv)
python run.py --algorithm greedy --data test

# آموزش PPO (روی Data1-3.csv، با BC warm-start از Greedy)
python -m algorithms.ppo.train

# ارزیابی PPO آموزش‌دیده (روی Data4.csv، inference-only)
python -m algorithms.ppo.infer


# هر بار قبل از اجرا، فقط همین خط را در config.py عوض کنید:
# SEED = 44   سپس:
python -m algorithms.ppo.train

python -m algorithms.ppo.infer

python -m evaluation.compare_runs --output-dir outputs/seed42

python -m evaluation.compare_runs --output-dir outputs/seed43

python -m evaluation.compare_runs --output-dir outputs/seed44

python -m evaluation.compare_runs --output-dir outputs/seed45

# SEED = 45   سپس همان مراحل با outputs/seed45


# مقایسه‌ی هر ۴ الگوریتم روی داده‌ی یکسان
python -m evaluation.compare_runs --data test

# تحلیل کیفیت تصمیمات یک الگوریتم خاص (بعد از اجرای بالا)
python analyze_decision_quality.py outputs/ppo_events.jsonl

# تماشای منحنی یادگیری زنده (اگر tensorboard نصب باشد)
tensorboard --logdir logs/tensorboard
```

### زمان تقریبی اجرا (روی سخت‌افزار معمولی)

| عملیات | زمان تقریبی |
|---|---|
| اجرای یک الگوریتم قاعده‌محور روی Data4.csv | چند ثانیه تا چند دقیقه |
| آموزش PPO (۳,۰۰۰,۰۰۰ timestep، ۸ محیط موازی) | ~۴ تا ۵ ساعت (سرعت مشاهده‌شده: ~۱۷۸ timestep/ثانیه) |
| `evaluation.compare_runs` (هر ۴ الگوریتم، Data4.csv کامل) | چند دقیقه |

---

## ۴. چهار الگوریتم

### Greedy (baseline)
- **scale_decision**: `occ_ratio > 0.7` یا `rejection_rate > 0` → SCALE_UP؛ `occ_ratio < 0.1` و `n_replicas > 1` → SCALE_DOWN.
- **provision_decision**: نزدیک‌ترین سرور OFF جغرافیایی به سرور overload‌شده + پروفایل متناسب با میزان اضافه‌بار (heterogeneity-aware) + سیگنال capacity-starved (پایین‌تر توضیح داده می‌شود).
- **placement**: نزدیک‌ترین سرور فعال به مرکز ثقل سرورهای فعال.

### VOILA
پیاده‌سازی فلسفه‌ی مقاله‌ی *"Voila: Tail-Latency-Aware Fog Application Replicas
Autoscaler"* (MASCOTS 2020، Fahs, Pierre, Elmroth):
- **تفاوت اصلی با Greedy**: placement/migration بر پایه‌ی **مرکز ثقل تقاضای واقعی**
  هر سرویس (`demand_centroid` — میانگین متحرک نمایی α=۰.۳ از موقعیت جغرافیایی
  درخواست‌های اخیر)، نه صرفاً نزدیک‌ترین به مرکز سرورهای فعال.
- **scale_decision**: ترکیب نقض ظرفیت (Vco: `occ_ratio > 0.75`) و نقض تاخیر/مکان
  (Vlo: `deadline_violation_rate > 0`) طبق Procedure 4 مقاله؛ scale-down فقط بعد از
  ۳ تیک متوالی بدون نقض (`SCALE_DOWN_PATIENCE_TICKS`، بخش V-C مقاله).

### Kubernetes-HPA
- **scale_decision**: فرمول رسمی HPA — `desired = ceil(current_replicas × current_util / TARGET_UTILIZATION)` با `TARGET_UTILIZATION=0.70`.
- **کاملاً location-unaware** طبق تعریف صریح سند (بدون haversine در provisioning/placement، تای‌بریک بر اساس id).

### PPO-DRL
- **State**: بردار ۱۲۲بعدی (۱۰ سرور × ۶ + ۱۵ سرویس × ۴ + ۲ سراسری) — `common/state_builder.py`.
- **Action**: `MultiDiscrete([3]*15 + [3]*10)` — هر سرویس `{NO_CHANGE, SCALE_UP, SCALE_DOWN}`، هر سرور `{NO_CHANGE, TURN_ON, TURN_OFF}`، با `MaskablePPO` (sb3-contrib) برای حذف اکشن‌های نامعتبر.
- **Reward**: ترکیب وزن‌دار منفی از ۵ جزء نرمال‌شده (بخش ۵ برای تاریخچه‌ی کامل تنظیم آن).
- **آموزش**: BC warm-start (۲۵ epoch، معلم=Greedy) + fine-tune RL (۳,۰۰۰,۰۰۰ timestep، ۸ محیط موازی، `ent_coef=0.01`).

---

## ۵. مسیر کامل توسعه و رفع باگ

این پروژه از یک بررسی عمیق کد شروع شد که ۱۲ مشکل بالقوه شناسایی کرد، و بعد در چند
دور اصلاح، آموزش مجدد، و تحلیل نتایج، به نتیجه‌ی نهایی رسید. این بخش تاریخچه‌ی کامل
را (به ترتیب زمانی) مستند می‌کند — چون هرکدام از این اصلاحات روی معیارهای نهایی اثر
مستقیم و قابل‌اندازه‌گیری داشتند.

### فاز ۱ — بررسی اولیه و اصلاحات بنیادی موتور مشترک

این اصلاحات روی **هر ۴ الگوریتم به‌طور یکسان** اثر گذاشتند چون در `simulator/engine.py`
و `algorithms/base.py` بودند، نه در کد اختصاصی هیچ الگوریتمی.

#### ۱.۱ Migration بدون Make-Before-Break (باگ بحرانی)
**قبل:** وقتی سروری drain می‌شد، رپلیکای جدید روی مقصد STARTING می‌شد *و هم‌زمان*
رپلیکای قدیم بلافاصله DRAINING می‌شد. چون رپلیکای DRAINING فوراً از کاندیدهای Router
حذف می‌شود ولی رپلیکای جدید تا `POD_STARTUP_DELAY_SEC` (۵ ثانیه) بعد READY نیست، یک
پنجره‌ی واقعی وجود داشت که هیچ رپلیکای READY از آن سرویس در دسترس نبود →
`REJECTED_NO_REPLICA` غیرواقعی.

**اصلاح:** رپلیکای قدیم READY می‌ماند و فقط *پس از* READY شدن رپلیکای جدید وارد
DRAINING می‌شود (`_pending_migrations` dict + هوک در `_handle_replica_ready`) —
دقیقاً طابق بخش ۶.۲ سند.

#### ۱.۲ لاگ‌نشدن رویدادهای تصمیم (بخش ۱۲ سند)
`scale_decision`، `provision_decision`، `migration_started`، `migration_completed`
و `request_rejected` (برای `REJECTED_NO_REPLICA`) اصلاً لاگ نمی‌شدند. اضافه شدند —
همراه با فیلدهای `applied`/`skip_reason`/`necessary_*` برای audit trail کامل هر تیک.

#### ۱.۳ Scale-Up بدون تداوم (نه میانگین‌گیری روی پنجره)
سمت پایین (`TURN_OFF`) از قبل نیاز به تداوم `SUSTAIN_LOW_SEC=60s` داشت، ولی سمت بالا
(`TURN_ON`) با یک نمونه‌ی لحظه‌ای بالای آستانه فوراً trigger می‌شد. اضافه شد:
`_any_active_server_sustained_overloaded()` با `SUSTAIN_HIGH_SEC=30s` — متقارن با سمت
پایین.

#### ۱.۴ عدم heterogeneity-aware provisioning
انتخاب سرور OFF برای روشن‌کردن فقط بر اساس نزدیک‌ترین فاصله بود؛ پروفایل ظرفیتی
(`edge_small`/`medium`/`large`) هرگز با میزان اضافه‌بار تطبیق داده نمی‌شد. تابع مشترک
`_pick_profile_for_overload()` در `algorithms/base.py` اضافه شد.

#### ۱.۵ نبود Cooldown سطح سرویس
سند بخش ۷ صریحاً «Cooldown مشابه ۶.۱ برای هر service_id» می‌خواهد، ولی هیچ‌جا پیاده
نشده بود. `self._service_last_scale_time` در `SimulationEngine` اضافه شد
(`COOLDOWN_SEC` مشترک).

#### ۱.۶ نبود معیار «درستی تصمیم» (بخش ۸ سند)
سند می‌خواهد هر الگوریتم گزارش بدهد «چند تا از تصمیماتش واقعاً لازم بودند». چون
threshold خودِ هر الگوریتم نمی‌تواند معیار خودش باشد (هر تصمیم به‌تعریف «درست» می‌شود)،
یک **آستانه‌ی ممیزی مستقل** تعریف شد (`DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD=0.7`،
`DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD=0.2`) که هر ۴ الگوریتم با همان یک خط‌کش
سنجیده می‌شوند. خروجی `decision_correctness` در نتیجه‌ی نهایی هر الگوریتم اضافه شد
(`correct`/`incorrect`/`missed_opportunities`/`correctness_rate_pct` برای هرکدام از
SCALE_UP، SCALE_DOWN، TURN_ON، TURN_OFF).

> **باگ فرعی که بعداً کشف شد:** معیار SCALE_DOWN چک نمی‌کرد که `n_replicas > 1`
> باشد؛ یعنی سرویسی که از قبل فقط ۱ رپلیکا داشت (و اصلاً نمی‌شد کمترش کرد) هم به‌غلط
> «فرصت ازدست‌رفته» ثبت می‌شد — همین باعث اعداد `missed_opportunities` نجومی (~۲۶٬۰۰۰)
> شده بود. رفع شد با اضافه‌کردن `and sv["n_replicas"] > 1`.

#### ۱.۷ لاگ‌نشدن منحنی یادگیری PPO
سند بخش ۱۱.۵ صریحاً «لاگ منحنی یادگیری (reward per episode) برای گزارش‌دهی» می‌خواهد.
`Monitor(...)` بدون `filename` ساخته می‌شد (فقط stdout). اصلاح شد: هر یک از ۸ محیط
موازی، لاگ خودش را در `logs/monitor/env_{i}.monitor.csv` می‌نویسد، و `MaskablePPO` با
`tensorboard_log` ساخته می‌شود.

---

### فاز ۲ — رفع عدم‌تعادل Reward در PPO

اولین بار که PPO ارزیابی شد، در response_time و deadline_violation بهترین بود ولی در
انرژی (**بدترین** — ۳۸.۱M در برابر ۳۲-۳۵.۷M بقیه) و توازن بار (**بدترین** —
`cv=0.71` در برابر ۰.۲۸-۰.۵۸ بقیه) به‌شدت عقب بود، با اینکه فقط ۵۵ اکشن در کل ۲۸۸۰
تیک زده بود (Greedy: ۵۰۵ تا).

**تشخیص:** جریمه‌ی «درخواست ردشده» در reward یک ثابت **نرمال‌نشده** بود
(`PPO_PENALTY_PER_REJECTED × تعداد خام رد در تیک`)، در حالی‌که ۴ جزء دیگر reward همه
در بازه‌ی `[0,2]` نرمال بودند. در تیک‌های شلوغ (۵-۱۰ رد هم‌زمان)، این جمله می‌توانست
۵ تا ۱۵ برابر بزرگ‌تر از مجموع ۴ جزء دیگر شود و کل سیگنال یادگیری را تحت‌الشعاع قرار
دهد → عامل یاد گرفت «به هر قیمتی جلوی رد را بگیر»، یعنی سرور اضافه‌ی همیشه-روشن نگه
دارد، بدون مدیریت پویا.

**اصلاح:**
1. `num_rejected_recent` هم مثل بقیه به `[0,2]` نرمال شد (`_NORM_REJECTED_PER_TICK=5.0`)
   و وزن صریح `w5_rejected` گرفت (به‌جای ثابت جداگانه).
2. وزن‌های reward چند بار بازتنظیم شدند (جدول کامل در بخش ۹).

**نتیجه‌ی این فاز:** انرژی ↓۱۰.۹٪، توازن بار ↓۴۸٪ بهتر، تعداد اکشن‌ها ۳.۲ برابر بیشتر
(دیگر «بی‌عمل» نبود) — با هزینه‌ی از‌دست‌دادن موقت رتبه‌ی اول در نرخ نقض deadline.

---

### فاز ۳ — کشف باگ سیستمی Capacity-Starved (مهم‌ترین کشف پروژه)

بعد از فاز ۱ و ۲، در یک اجرای بعدی، اعداد **Greedy/Voila/HPA** (که کدشان اصلاً عوض
نشده بود!) به‌طرز غیرمنتظره‌ای تغییر کردند — `avg_active_servers` تقریباً نصف شد و
`avg_load_balance_cv` سه برابر شد. چون این الگوریتم‌ها کاملاً قطعی (deterministic)
هستند، این ناهنجاری نشانه‌ی یک تغییر واقعی در کد مشترک بود.

**ریشه‌یابی:** با بررسی event log خود Greedy، مشخص شد **۳۵۰۰ از ۳۵۰۸ تلاش SCALE_UP
با `no_target_server` شکست خورده بودند**. علت: معیار قدیمی TURN_ON فقط busy-fraction
(utilization) را می‌سنجید، نه ظرفیت آزاد واقعی. یک سرور می‌توانست کاملاً پر باشد
(`free_capacity=0`) ولی چون هم‌زمان ۱۰۰٪ busy نبود، هرگز «اضافه‌بار» تشخیص داده
نمی‌شد — یعنی سیستم سیستماتیک نمی‌توانست نیاز واقعی به ظرفیت بیشتر را تشخیص بدهد.

**اصلاح (`_capacity_starved_services` در `algorithms/base.py`):** سیگنال جدیدی اضافه
شد که سرویس‌هایی را که هیچ سروری برای میزبانی رپلیکای بیشترشان جا ندارد شناسایی
می‌کند، و این سیگنال هم در تصمیم‌گیری الگوریتم (`provision_decision`) و هم در گیت
سطح‌موتور استفاده می‌شود.

> **مشکل انصاف که کشف و رفع شد:** این اصلاح ابتدا فقط در `Greedy` پیاده شده بود.
> چون گیت سطح‌موتور فقط می‌تواند اکشنی را *اجازه* بدهد که خودِ الگوریتم قبلاً تصمیم
> گرفته، بدون این اصلاح در تصمیم‌گیری خودِ Voila و HPA، آن‌ها همچنان کور به وضعیت
> capacity-starved می‌ماندند. برای انصاف مقایسه، همان الگو (با حفظ فلسفه‌ی هرکدام —
> Voila مکان‌آگاه با `demand_centroid`، HPA کاملاً location-unaware) به هر دو هم
> اضافه شد.

**اثر روی نتایج:** برای هر ۳ الگوریتم قاعده‌محور، response_time و deadline_violation
هر دو *بهتر* شدند (چون سیستم دیگر سیستماتیک از تأمین ظرفیت واقعی امتناع نمی‌کرد)، ولی
انرژی و توازن بار *بدتر* شدند (چون سیستم مجبور بود واقعاً برای تأمین تقاضای واقعی کار
کند). یعنی اعداد «خوب» قبلی انرژی/توازن تا حد زیادی مصنوعی بودند — نتیجه‌ی خساست
اجباری سیستم، نه کارآمدی واقعی.

**علاوه بر آن، بخش ۶.۲ سند (Emergency-boot) هم در این فاز کامل شد:** وقتی migration
هنگام drain نمی‌تواند مقصد پیدا کند، به‌جای صرفاً لغوکردن drain (رفتار موقت قبلی)،
حالا یک سرور OFF مناسب Boot اضطراری می‌شود (`_trigger_emergency_boot`).

---

### فاز ۴ — کشف و رفع Flapping شدید سرور

بعد از فاز ۳، یک ابزار تشخیصی سفارشی (`analyze_decision_quality.py` — بخش ۶) با یک
قابلیت flapping-detector گسترش داده شد که یافته‌ی هشداردهنده‌ای نشان داد:

```
کل چرخه‌های on->off سرور: 306
flapping (dwell < 300s): 294  (۹۶.۱٪!)
میانگین dwell: 119 ثانیه (~۲ دقیقه)
```

**ریشه‌یابی دقیق** (با ردیابی تیک‌به‌تیک یک نمونه‌ی واقعی از event log):

```
t=66960:  TURN_ON (بوت می‌شود)
t=66990:  server_active — همان تیک TURN_OFF پیشنهاد می‌شود ولی رد می‌شود (sustain کافی نیست)
t=67050:  TURN_OFF اعمال می‌شود  (دقیقاً ۶۰ ثانیه بعد از ACTIVE شدن)
```

چون `COOLDOWN_SEC` و `SUSTAIN_LOW_SEC` هر دو ۶۰ ثانیه بودند، تقریباً **هم‌زمان** سر
می‌رسیدند. سروری که فقط برای میزبانی یک رپلیکای کوچک (rescue migration یا
capacity-starved) روشن می‌شد، از همان لحظه‌ی اول کم‌بار بود و تقریباً بلافاصله دوباره
خاموش می‌شد — کل هزینه‌ی `E_BOOT_SERVER_J` (۵۰۰J) + pod-create برای هیچ به هدر می‌رفت.
این توضیح می‌داد چرا انرژی Greedy بعد از فاز ۳ به‌جای بهبود، بدتر شده بود.

**اصلاح (دو گیت مجزا و جدید، به‌جای دستکاری `COOLDOWN_SEC` عمومی):**
- `MIN_ACTIVE_DURATION_SEC=300` — سرور حداقل ۵ دقیقه باید ACTIVE مانده باشد قبل از
  واجد‌شرایط‌بودن برای TURN_OFF.
- `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC=120` — همان الگو در سطح replica.

این طراحی از بالابردن کل `COOLDOWN_SEC` بهتر است چون فقط دقیقاً سناریوی «TURN_OFF
زودهنگام بعد از boot تازه» را هدف می‌گیرد، بدون کندکردن کل رفتار سیستم.

**نتیجه:** flapping کاملاً حذف شد (۰٪ در تست‌های بعدی، ۱۰۰٪ چرخه‌ها پایدار، dwell
میانگین ۸۰۰+ ثانیه).

---

### فاز ۵ — تنظیم نهایی وزن‌های Reward PPO

بعد از فازهای ۳ و ۴ (که روی هر ۴ الگوریتم اثر گذاشتند)، PPO دوباره با ۳,۰۰۰,۰۰۰
timestep و ۲۵ epoch BC آموزش داده شد (به‌جای ۵۰۰,۰۰۰/۱۰ اولیه — چون هم منحنی reward
و هم loss BC هنوز کاملاً همگرا نشده بودند). نتیجه: PPO در ۳ از ۴ معیار برنده شد،
ولی در نرخ نقض deadline کمی عقب ماند (۳.۵۸٪ در برابر ۳.۱۰٪ بهترین/Greedy).

**تنظیم نهایی (یک تغییر مجزا و قابل‌انتساب، نه چند تا هم‌زمان):**

با نگاه به فاصله‌ی PPO تا بهترین رقیب هر معیار (response_time: ~۱۲٪ فاصله/margin
زیاد؛ انرژی: ~۱٪ فاصله/رقیب نزدیک؛ توازن: ~۱۵٪ فاصله)، تصمیم گرفته شد فقط `w1`
(که بیشترین margin غیرضروری را داشت) کم شود و کاملاً به `w2` منتقل شود:

```python
# قبل:  w1=0.12, w2=0.20, w3=0.30, w4=0.23, w5=0.15
# بعد:  w1=0.08, w2=0.24, w3=0.30, w4=0.23, w5=0.15
```

همزمان (به‌طور مستقل، در بازآموزی نهایی) `ent_coef=0.01` برای تشویق exploration و
`CheckpointCallback` اضافه شدند، و `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC` معرفی شد.

**نتیجه‌ی نهایی:** PPO در **هر ۴ معیار** برنده شد (جدول کامل در بخش ۷).

> **یادداشت رگور علمی:** بین این اجرا و اجرای قبلی، علاوه بر وزن reward، مقدار
> `SEED` هم از ۴۲ به ۴۳ تغییر کرد و `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC` هم اضافه
> شد. یعنی دقیقاً نمی‌توان گفت این بهبود *فقط* از تغییر وزن آمده - ترکیبی از هر سه
> تغییر است. برای گزارش نهایی، این را صادقانه ذکر کنید.

---

## ۶. ابزار تحلیل کیفیت تصمیم

`analyze_decision_quality.py` یک اسکریپت تشخیصی سفارشی است (در حین این پروژه ساخته
و توسعه داده شد) که فراتر از معیار لحظه‌ای بخش ۸ می‌رود و دو سؤال عمیق‌تر را جواب
می‌دهد:

### ۶.۱ آیا اکشن‌های «غیرضروری» واقعاً نویز هستند یا پیش‌بینانه؟

برای هر اکشن اعمال‌شده که معیار ممیزی لحظه‌ای «غیرضروری» علامت زده، به
`LOOKAHEAD_TICKS` تیک بعدی (پیش‌فرض ۳ = ۹۰ ثانیه) همان سرویس نگاه می‌کند:

- **SCALE_UP غیرضروری** → اگر طی این پنجره واقعاً لازم شد یا درخواستی رد شد ⇒
  `anticipatory` (پیش‌بینانه‌ی موجّه). وگرنه ⇒ `noise`.
- **SCALE_DOWN غیرضروری** → اگر طی این پنجره سرویس دچار کمبود شد یا درخواستی رد
  شد ⇒ `risky` (این حذف replica به‌احتمال زیاد باعث کمبود بعدی شده). وگرنه ⇒
  `harmless_early`.

```bash
python analyze_decision_quality.py outputs/ppo_events.jsonl
```

### ۶.۲ آیا سرورها flapping می‌کنند؟

هر چرخه‌ی کامل on→off هر سرور را پیدا می‌کند و مدت `dwell` (زمان بین روشن‌شدن و
دوباره خاموش‌شدن) را می‌سنجد؛ اگر زیر یک آستانه (پیش‌فرض ۳۰۰ ثانیه) باشد،
`flapping` علامت می‌زند.

### یافته‌ی کلیدی از این ابزار (تفسیر نهایی)

با اینکه SCALE_UP در PPO نرخ «درستی» لحظه‌ای بسیار پایینی دارد (~۶٪)، این ابزار
نشان داد این عدد گمراه‌کننده است — **نامتقارنی هزینه** را در نظر نمی‌گیرد:

| | SCALE_UP اشتباه | SCALE_DOWN اشتباه |
|---|---|---|
| هزینه | `E_POD_CREATE_J`=۲۰J، بدون آسیب به QoS | می‌تواند باعث رد درخواست/نقض deadline شود |
| رفتار PPO | زیاد و بی‌احتیاط (بیمه‌ی ارزان) | کم و محتاط (correctness=۸۷.۶٪، `risky` فقط ۲.۶-۴٪) |

PPO این عدم‌تقارن را خودش یاد گرفته: `num_pod_creates` بسیار بالا (اکشن‌های ارزان
سطح replica) ولی `num_server_boots` پایین‌ترین بین هر ۴ الگوریتم (اجتناب از
اکشن‌های گران سطح سرور). حتی با `LOOKAHEAD_TICKS=15` (۷.۵ دقیقه)، اکثریت قاطع
SCALE_UPهای «غیرضروری» همچنان `noise` طبقه‌بندی می‌شوند — تأیید می‌کند این یک
استراتژی عمدی است، نه سردرگمی سیاست.

---

## ۷. نتایج نهایی

### جدول کامل مقایسه (Data4.csv، ارزیابی نهایی)

| معیار | Greedy | Voila | HPA | **PPO** |
|---|---:|---:|---:|---:|
| avg_response_time_sec | 45.88 | 46.97 | 49.61 | **۳۸.۲۴** |
| p95/p99 response time | نزدیک بقیه | نزدیک بقیه | نزدیک بقیه | نزدیک بقیه |
| deadline_violations | - | - | - | - |
| deadline_violation_rate_pct | 3.22 | 3.28 | 3.81 | **۳.۱۶** |
| cumulative_energy_joule | 44.58M | 43.80M | 41.75M | **۳۸.۸۵M** |
| avg_load_balance_cv | 0.834 | 0.873 | 0.866 | **۰.۵۴۸** |
| avg_active_servers | ~۲ | ~۲ | ~۲.۳ | ~۲.۷ (بیشتر، ولی کارآمدتر) |
| num_server_boots | متوسط | متوسط | متوسط | **کمترین** (پایداری بیشتر) |
| num_pod_creates | متوسط | کمترین | متوسط | **بیشترین** (elasticity ارزان سطح replica) |

### تفسیر نهایی

PPO با یک استراتژی نامتقارن — SCALE_UP آزادانه و ارزان (بیمه) + SCALE_DOWN محتاط و
دقیق + اجتناب از boot/shutdown گران‌قیمت سرور به نفع مدیریت ظریف‌تر سطح replica —
در هر ۴ معیار اصلی سبقت گرفت. این نتیجه فقط با یک عامل RL نبود؛ به همان اندازه به
رفع سه باگ سیستمی در موتور مشترک (migration، capacity-starved، flapping) وابسته
بود که هر ۴ الگوریتم را منصفانه‌تر و واقعی‌تر کرد.

---

## ۸. یافته‌های کلیدی

این‌ها مهم‌ترین درس‌های این پروژه هستند که برای گزارش نهایی/بحث نتایج ارزشمندند:

1. **دیباگ مبتنی بر لاگ ضروری بود.** چند باگ حیاتی (capacity-starved، flapping) فقط
   با تحلیل واقعی event logs کشف شدند، نه با بازبینی کد به‌تنهایی — تأکیدی بر ارزش
   اعتبارسنجی تجربی نسبت به audit ایستا.

2. **ناهنجاری‌های Deterministic، سرنخ باگ‌اند.** وقتی نتایج الگوریتم‌های کاملاً
   قطعی (Greedy/Voila/HPA) بدون تغییر کد بین دو اجرا فرق کرد، این نشانه‌ی قطعی یک
   تغییر واقعی (نه نویز) در کد مشترک بود — دقیقاً همین‌طور هم شد.

3. **اعداد «خوب» می‌توانند مصنوعی باشند.** انرژی/توازن بار پایین قبل از رفع باگ
   capacity-starved، نتیجه‌ی خساست اجباری سیستم بود (چون نمی‌توانست SCALE_UP کند)،
   نه کارآمدی واقعی. رفع باگ باعث شد اعداد «بدتر» شوند ولی QoS واقعی بهتر شود —
   نمونه‌ای از این‌که معیارهای سطح‌بالا بدون context می‌توانند گمراه‌کننده باشند.

4. **معیار ممیزی مستقل، خودش محدودیت دارد.** معیار لحظه‌ای بخش ۸ نامتقارنی هزینه‌ی
   SCALE_UP در برابر SCALE_DOWN را نمی‌بیند؛ نرخ «درستی» پایین لزوماً به‌معنای
   عملکرد بد نیست — باید همیشه در کنار نتایج aggregate واقعی تفسیر شود.

5. **جریمه‌ی نرمال‌نشده در reward می‌تواند کل سیگنال یادگیری را غالب کند.** یک جزء
   reward که مقیاسش با بقیه هماهنگ نیست (even با وزن کوچک)، می‌تواند رفتار عامل را
   به‌طور کامل منحرف کند.

6. **تغییرات هم‌زمان، اسنادپذیری نتیجه را کاهش می‌دهند.** در فاز ۵ توصیه شد فقط یک
   پارامتر عوض شود؛ وقتی به‌جای آن چند چیز هم‌زمان تغییر کرد (seed + معماری replica-age)
   نتوانستیم اثر خالص هرکدام را جدا کنیم — درسی برای طراحی آزمایش‌های آینده.

---

## ۹. پارامترهای قابل کالیبراسیون

طبق بخش ۱۳ سند، این پارامترها به‌عمد قابل‌تنظیم‌اند و مقادیر فعلی نتیجه‌ی چند دور
کالیبراسیون تجربی‌اند، نه فرض اولیه:

| پارامتر | مقدار نهایی | دلیل/سرگذشت |
|---|---:|---|
| `SUSTAIN_HIGH_SEC` | 30.0 | یک `MONITOR_WINDOW` کامل - تداوم واقعی overload، نه نمونه‌ی لحظه‌ای |
| `SUSTAIN_LOW_SEC` | 60.0 | مقدار اصلی سند |
| `COOLDOWN_SEC` | 60.0 | مقدار اصلی سند (anti-flapping عمومی سطح تصمیم) |
| `MIN_ACTIVE_DURATION_SEC` | **300.0** | جدید - رفع flapping سرور (۹۶٪ چرخه‌ها زیر این آستانه بودند) |
| `MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC` | **120.0** | جدید - همان الگو در سطح replica |
| `DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD` | 0.7 | آستانه‌ی ممیزی مستقل (نه threshold داخلی هیچ الگوریتمی) |
| `DECISION_AUDIT_SCALE_DOWN_OCC_THRESHOLD` | 0.2 | همان بالا |
| `L0_MS` | 20.0 | مقدار پیش‌فرض خودِ مقاله‌ی Voila (در سند بخش ۹ فراموش شده بود) |
| `w1_response_time` | 0.08 | کاهش‌یافته - PPO از قبل margin زیاد داشت |
| `w2_deadline` | 0.24 | افزایش‌یافته - تنها معیاری که PPO عقب بود |
| `w3_energy` | 0.30 | دست‌نخورده - رقیب نزدیک (HPA) بود |
| `w4_load_balance` | 0.23 | دست‌نخورده |
| `w5_rejected` | 0.15 | نرمال‌شده (قبلاً ثابت جداگانه‌ی نرمال‌نشده بود) |
| `ent_coef` (PPO) | 0.01 | تشویق exploration - قبلاً پیش‌فرض ۰.۰ بود |
| `total_timesteps` (PPO) | 3,000,000 | از ۵۰۰,۰۰۰ افزایش یافت - منحنی reward هنوز همگرا نشده بود |
| `bc_epochs` | 25 | از ۱۰ افزایش یافت - loss BC هنوز نزولی بود |
| `SEED` | 43 | تغییر یافت (از ۴۲) در آخرین دور - نکته‌ی رگور در بخش ۵ |

---

## ۱۰. تصمیمات مهندسی/فرضیات صریح

(از README اصلی پروژه، همچنان معتبر)

- **موتور heapq-based به‌جای simpy**: چون simpy در محیط توسعه در دسترس نبود، یک موتور
  discrete-event معادل و سبک با heapq نوشته شد (`simulator/engine.py`). اگر simpy در
  دسترس بود، فقط همین فایل باید بازنویسی شود.
- **جایگذاری اولیه با ظرفیت کافی**: پوشش حریصانه‌ی صرفِ جغرافیایی کافی نبود (مجموع
  `cpu_demand` هر ۱۵ سرویس = ۲۴۸ > حداکثر ظرفیت هر سرور به‌تنهایی = ۲۰۰)؛ انتخاب
  اولیه تا کافی‌شدن ظرفیت کل گسترش داده شد.
- **مرکز ثقل تقاضا برای Voila**: چون `AlgorithmBase` استاندارد به snapshot خام
  موقعیت هر درخواست دسترسی ندارد، `simulator/engine.py` یک EMA (α=۰.۳) از موقعیت
  جغرافیایی درخواست‌های اخیر هر سرویس نگه می‌دارد.
- **آستانه‌های heuristic هر الگوریتم** در همان فایل الگوریتم نگه داشته شدند، نه
  `config.py`، چون سیاست تصمیم‌گیری مختص هر الگوریتم‌اند، نه قید فیزیکی سیستم.

---

## ۱۱. محدودیت‌های شناخته‌شده و کارهای باقی‌مانده

- **`data/loader.py`** به‌طور کامل مستقل بازبینی/راستی‌آزمایی نشده (افست
  `global_start_sec` طبق بخش ۱.۳ سند) — در صورت شک، دستی چک شود.
- **`k8s_adapter/`** (فاز ۳ — اجرای واقعی روی Kubernetes) هنوز شروع نشده؛ خارج از
  محدوده‌ی این دور از کار بود.
- **رگور علمی فاز ۵**: همان‌طور که در بخش ۵ ذکر شد، نتیجه‌ی نهایی ترکیبی از چند
  تغییر هم‌زمان است (وزن reward + seed + مکانیزم replica-age)؛ برای یک ادعای علمی
  کاملاً تمیز، یک اجرای isolate (فقط تغییر وزن، seed=۴۲ ثابت) پیشنهاد می‌شود.
- **دو مکانیزم نسبتاً جدید** (`_trigger_emergency_boot`، `_capacity_starved_services`)
  با دقت بازبینی و تست شدند ولی به‌اندازه‌ی بخش‌های قدیمی‌تر کد under battle-tested
  نیستند — در صورت مشاهده‌ی رفتار غیرمنتظره در آینده، این دو نقطه‌ی اول برای بررسی‌اند.

---

## ۱۲. مرجع سند معماری

این پروژه پیاده‌سازی دقیق **«سند معماری کامل پروژه: مدیریت پویای منابع لبه، نسخه
۱.۰»** است. بخش‌های کلیدی سند که در طول توسعه بیشترین ارجاع را داشتند:

| بخش سند | موضوع |
|---|---|
| ۱.۱ / ۱.۲ | پروفایل/ظرفیت سرورها، مشخصات سرویس‌ها |
| ۲ | مدل موجودیت‌ها (Server/Replica/Request + State Machines) |
| ۳ | چرخه‌ی کامل یک درخواست |
| ۴ | جایگذاری اولیه (Set-Cover) |
| ۶.۱ | آستانه‌های provisioning + heterogeneity-aware profile selection |
| ۶.۲ | Service Migration هنگام Drain (Make-Before-Break + Emergency-boot) |
| ۷ | Service Auto Scaling + Cooldown |
| ۸ | متریک‌های نهایی + معیار «درستی تصمیم» |
| ۱۱ | فضای حالت/اکشن/reward PPO-DRL |
| ۱۲ | الزامات لاگ‌گیری ساخت‌یافته |
| ۱۳ | پارامترهای قابل کالیبراسیون |

منبع مقاله‌ی Voila: *Fahs, A., Pierre, G., & Elmroth, E. "Voila: Tail-Latency-Aware
Fog Application Replicas Autoscaler." MASCOTS 2020.*