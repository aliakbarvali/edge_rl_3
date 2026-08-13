import numpy as np
from data.loader import load_train
from simulator.engine import SimulationEngine
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm

events = load_train()
engine = SimulationEngine(events, GreedyAlgorithm(), "greedy")
engine.prime()

# جمع‌آوری خام از هر تیک تصمیم
arrivals_per_service_tick = []      # recent_arrivals هر سرویس در هر تیک
response_time_per_tick = []         # avg_response_time_recent هر تیک (فقط غیرصفر)
energy_per_tick = []                # energy_recent_joule هر تیک
rejected_per_tick = []              # num_rejected_recent هر تیک (کل سیستم)
rejected_tick_times = []            # sim_time_sec متناظر با هر عنصر rejected_per_tick
                                     # (برای ریشه‌یابی تیک‌های پرت مثل max=84)

n_ticks = 0
while True:
    snapshot, done = engine.step()
    if done:
        break
    n_ticks += 1

    for svc_id, sv in snapshot["services"].items():
        arrivals_per_service_tick.append(sv["recent_arrivals"])

    g = snapshot["global"]
    if g["avg_response_time_recent"] > 0:
        response_time_per_tick.append(g["avg_response_time_recent"])
    energy_per_tick.append(g["energy_recent_joule"])

    n_rejected = g["num_rejected_recent"]
    rejected_per_tick.append(n_rejected)
    rejected_tick_times.append(engine.now)


def report(name, arr):
    arr = np.array(arr, dtype=float)
    print(f"\n=== {name} (n={len(arr)}) ===")
    if len(arr) == 0:
        print("  (هیچ نمونه‌ای موجود نیست)")
        return
    print(f"  mean   = {arr.mean():.3f}")
    print(f"  median = {np.median(arr):.3f}")
    print(f"  p90    = {np.percentile(arr, 90):.3f}")
    print(f"  p95    = {np.percentile(arr, 95):.3f}")
    print(f"  p99    = {np.percentile(arr, 99):.3f}")
    print(f"  max    = {arr.max():.3f}")


print(f"تعداد کل تیک‌های تصمیم پردازش‌شده: {n_ticks}")
report("recent_arrivals (هر سرویس، هر تیک)  -> برای _NORM_ARRIVAL_RATE", arrivals_per_service_tick)
report("avg_response_time_recent (هر تیک، فقط غیرصفر)  -> برای _NORM_RESPONSE_TIME_SEC", response_time_per_tick)
report("energy_recent_joule (هر تیک)  -> برای _NORM_ENERGY_JOULE (تأیید مجدد)", energy_per_tick)
report("num_rejected_recent (هر تیک، کل سیستم - همه‌ی تیک‌ها شامل صفر)  -> فقط برای مرجع", rejected_per_tick)

# *** رفع مشکل: p90/p95 خام num_rejected_recent صفر است چون >۹۵٪ تیک‌ها
# اصلاً رد شدنی ندارند (توزیع کاملاً skewed/دوقطبی). برای کالیبراسیون
# _NORM_REJECTED_PER_TICK باید فقط روی تیک‌هایی که واقعاً رد شدن داشته‌اند
# percentile گرفت - دقیقاً همان الگویی که avg_response_time_recent با
# فیلتر "فقط غیرصفر" از قبل استفاده می‌کند.
nonzero_rejected = [r for r in rejected_per_tick if r > 0]
report(f"num_rejected_recent (فقط تیک‌های غیرصفر، {len(nonzero_rejected)}/{len(rejected_per_tick)} تیک)"
       f"  -> برای کالیبراسیون واقعی _NORM_REJECTED_PER_TICK", nonzero_rejected)

# ریشه‌یابی مستقیم outlierهای بزرگ: ۱۰ تیک با بیشترین رد شدن، همراه با
# sim_time_sec تا بشود همان بازه را در greedy_events.jsonl (event_type در
# {"scale_decision", "provision_decision", "server_boot_started",
# "server_drain_started", ...}) پیدا و بررسی کرد.
print("\n" + "=" * 60)
print("۱۰ تیک با بیشترین num_rejected_recent (برای ریشه‌یابی outlierها):")
paired = sorted(zip(rejected_per_tick, rejected_tick_times), key=lambda x: -x[0])[:10]
for rejected, t in paired:
    print(f"  sim_time_sec={t:>12.1f}   num_rejected_recent={rejected:.0f}")

print("\n" + "=" * 60)
print("راهنمای انتخاب عدد:")
print("عدد p90 یا p95 را به‌عنوان 'حداکثر معمول' (norm=1.0) انتخاب کن.")
print("برای num_rejected_recent از بخش 'فقط تیک‌های غیرصفر' بالا استفاده کن،")
print("نه از p90/p95 خام (که چون اکثر تیک‌ها صفرند، همیشه صفر می‌شوند).")
print("از max خام استفاده نکن، چون یک نقطه‌ی پرت (outlier) کل مقیاس را خراب می‌کند.")
print("=" * 60)