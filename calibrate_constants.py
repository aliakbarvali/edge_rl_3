# calibrate_constants.py
"""
اندازه‌گیری واقعی ثابت‌های نرمال‌سازی (common/state_builder.py و
algorithms/ppo/env.py) روی Data4.csv با اجرای Greedy - دقیقاً همان روش
مستندشده برای _NORM_ENERGY_JOULE.

اجرا: python3 calibrate_constants.py
"""
# calibrate_rejected.py
import numpy as np
from data.loader import load_test
from simulator.engine import SimulationEngine
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm

events = load_test()
engine = SimulationEngine(events, GreedyAlgorithm(), "greedy")
engine.prime()

rejected_per_tick = []
while True:
    snapshot, done = engine.step()
    if done:
        break
    rejected_per_tick.append(snapshot["global"]["num_rejected_recent"])

arr = np.array(rejected_per_tick, dtype=float)
nonzero = arr[arr > 0]

print(f"کل تیک‌ها: {len(arr)}")
print(f"تیک‌های با رد>0: {len(nonzero)}  ({100*len(nonzero)/len(arr):.2f}%)")
print(f"\n--- روی کل تیک‌ها (شامل صفرها) ---")
for p in [90, 95, 99, 99.9]:
    print(f"  p{p} = {np.percentile(arr, p):.2f}")
print(f"  max = {arr.max():.2f}")

print(f"\n--- فقط روی تیک‌های با رد>0 (شرطی) ---")
if len(nonzero) > 0:
    for p in [50, 75, 90, 95]:
        print(f"  p{p} = {np.percentile(nonzero, p):.2f}")
    print(f"  max = {nonzero.max():.2f}")
    print(f"  mean = {nonzero.mean():.2f}")
import numpy as np
from data.loader import load_test
from simulator.engine import SimulationEngine
from algorithms.greedy.greedy_algorithm import GreedyAlgorithm

events = load_test()
engine = SimulationEngine(events, GreedyAlgorithm(), "greedy")
engine.prime()

# جمع‌آوری خام از هر تیک تصمیم
arrivals_per_service_tick = []      # recent_arrivals هر سرویس در هر تیک
response_time_per_tick = []         # avg_response_time_recent هر تیک (فقط غیرصفر)
energy_per_tick = []                # energy_recent_joule هر تیک
rejected_per_tick = []              # num_rejected_recent هر تیک (کل سیستم)

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
    rejected_per_tick.append(g["num_rejected_recent"])

def report(name, arr):
    arr = np.array(arr, dtype=float)
    print(f"\n=== {name} (n={len(arr)}) ===")
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
report("num_rejected_recent (هر تیک، کل سیستم)  -> برای _NORM_REJECTED_PER_TICK (تأیید مجدد)", rejected_per_tick)

print("\n" + "="*60)
print("راهنمای انتخاب عدد:")
print("عدد p90 یا p95 را به‌عنوان 'حداکثر معمول' (norm=1.0) انتخاب کن.")
print("از max خام استفاده نکن، چون یک نقطه‌ی پرت (outlier) کل مقیاس را خراب می‌کند.")
print("="*60)