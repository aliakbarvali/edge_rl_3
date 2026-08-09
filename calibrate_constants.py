
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