"""
evaluation/aggregate_seeds.py
جمع‌بندی نتایج چند اجرای PPO با seedهای مختلف: میانگین ± انحراف‌معیار
برای هر معیار، تا در گزارش نهایی به‌جای یک عدد تک، بازه‌ی قابل‌اعتماد
گزارش شود.

اجرا (برای هر seed، اول مدل مخصوص همان seed را آموزش/ارزیابی کنید - نگاه
کنید common/config.py:EOTCH_SEED و algorithms/ppo/train.py:MODEL_PATH):
    EOTCH_SEED=42 python3 -m algorithms.ppo.train
    EOTCH_SEED=42 python3 -m evaluation.compare_runs --output-dir outputs/seed42
    EOTCH_SEED=43 python3 -m algorithms.ppo.train
    EOTCH_SEED=43 python3 -m evaluation.compare_runs --output-dir outputs/seed43
    ... (تکرار برای هر seed)
    python3 -m evaluation.aggregate_seeds --seeds 42 43 44 45 --base-dir outputs
"""
from __future__ import annotations
import argparse
import json
import os
import statistics


METRICS = [
    "avg_response_time_sec", "p95_response_time_sec", "p99_response_time_sec",
    "deadline_violation_rate_pct", "cumulative_energy_joule", "avg_distance_km",
    "avg_load_balance_cv", "num_requests_rejected_queue_full",
    "avg_active_servers", "completed_requests",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--base-dir", default="outputs")
    parser.add_argument("--algorithm", default="ppo")
    args = parser.parse_args()

    per_seed = {}
    for seed in args.seeds:
        path = os.path.join(args.base_dir, f"seed{seed}", f"{args.algorithm}_result.json")
        if not os.path.exists(path):
            print(f"[رد شد] {path} پیدا نشد.")
            continue
        with open(path, encoding="utf-8") as f:
            per_seed[seed] = json.load(f)

    if not per_seed:
        print("هیچ نتیجه‌ای پیدا نشد.")
        return

    print(f"تعداد seed یافت‌شده: {len(per_seed)}  ({sorted(per_seed.keys())})\n")
    print(f"{'metric':35s} {'mean':>15s} {'std':>15s} {'min':>15s} {'max':>15s}")
    for metric in METRICS:
        values = [r[metric] for r in per_seed.values() if metric in r]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{metric:35s} {mean:15.3f} {std:15.3f} {min(values):15.3f} {max(values):15.3f}")
 
    print("\n--- جدول خام هر seed ---")
    header = ["seed"] + METRICS
    print(",".join(header))
    for seed, r in sorted(per_seed.items()):
        row = [str(seed)] + [str(r.get(m, "")) for m in METRICS]
        print(",".join(row))


if __name__ == "__main__":
    main()
    
    
#python -m evaluation.aggregate_seeds --seeds 42 43 44 45 --base-dir outputs