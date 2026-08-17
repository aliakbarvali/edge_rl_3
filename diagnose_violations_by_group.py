"""
diagnose_violations_by_group.py

مکمل diagnose_violations_by_service.py: به‌جای گزارش هر سرویس جدا برای یک
الگوریتم، نرخ نقض deadline رو به تفکیک گروه سرویس (SERVICE_CHAIN_GROUPS در
common/config.py: G1={1,2,3}..G5={13,14,15}) برای چند الگوریتم *کنار هم*
گزارش می‌دهد.

هدف: تشخیص اینکه آیا افتراق واقعی بین الگوریتم‌ها در سرویس‌های با deadline
بازتر (G2-G5) وجود دارد ولی چون سرویس‌های G1 (deadline بسیار سفت، سقف
فیزیکی: queue_len=1-2) نرخ نقض بسیار بالا و تقریباً یکسانی بین هر ۴ الگوریتم
دارند، در معیار تجمیعی deadline_violation_rate_pct محو می‌شود.

اجرا:
    python diagnose_violations_by_group.py \
        outputs/greedy_events.jsonl \
        outputs/voila_events.jsonl \
        outputs/hpa_events.jsonl \
        outputs/ppo_events.jsonl
"""
from __future__ import annotations
import json
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import CFG, SERVICE_CHAIN_GROUPS

GROUP_LABELS = {grp: f"G{i+1}={list(grp)}" for i, grp in enumerate(SERVICE_CHAIN_GROUPS)}
SVC_TO_GROUP = {sid: grp for grp in SERVICE_CHAIN_GROUPS for sid in grp}


def load_algo_stats(path: str) -> dict[int, dict]:
    """برای یک فایل events.jsonl، به‌ازای هر service_id: arrivals/violated
    را می‌شمارد. دقیقاً همان منطق diagnose_violations_by_service.py
    (منبع اول deadline_violated روی خود رکورد، وگرنه مقایسه‌ی مستقیم
    response_time_sec با deadline؛ رد شدن هم نقض حساب می‌شود)."""
    deadline_sec_by_svc = {sid: info["deadline"] for sid, info in CFG.services_info.items()}
    arrivals = defaultdict(int)
    violated = defaultdict(int)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            et = rec.get("event_type")
            sid = rec.get("service_id")
            if sid is None:
                continue

            if et == "request_arrived":
                arrivals[sid] += 1
            elif et == "request_completed":
                if "deadline_violated" in rec:
                    is_violated = bool(rec.get("deadline_violated"))
                else:
                    dl = deadline_sec_by_svc.get(sid)
                    rt = rec.get("response_time_sec", 0.0) or 0.0
                    is_violated = dl is not None and rt > dl
                if is_violated:
                    violated[sid] += 1
            elif et == "request_rejected":
                violated[sid] += 1

    return {"arrivals": arrivals, "violated": violated}


def main():
    if len(sys.argv) < 2:
        print("استفاده: python diagnose_violations_by_group.py <events1.jsonl> [events2.jsonl ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    algo_names = [os.path.basename(p).replace("_events.jsonl", "") for p in paths]
    per_algo = {name: load_algo_stats(p) for name, p in zip(algo_names, paths)}

    # ---------------- جدول ۱: نرخ نقض به تفکیک گروه × الگوریتم ----------------
    print("=== نرخ نقض deadline به تفکیک گروه سرویس (violation_rate = violated/arrivals در همان گروه) ===\n")
    header = f"{'group':22s}" + "".join(f"{name:>14s}" for name in algo_names)
    print(header)
    print("-" * len(header))

    group_rates = defaultdict(dict)
    for grp in SERVICE_CHAIN_GROUPS:
        label = GROUP_LABELS[grp]
        row = f"{label:22s}"
        for name in algo_names:
            stats = per_algo[name]
            grp_arrivals = sum(stats["arrivals"].get(sid, 0) for sid in grp)
            grp_violated = sum(stats["violated"].get(sid, 0) for sid in grp)
            rate = 100.0 * grp_violated / grp_arrivals if grp_arrivals else 0.0
            group_rates[grp][name] = rate
            row += f"{rate:13.2f}%"
        print(row)

    # ---------------- جدول ۲: سهم هر گروه از کل نقض‌های هر الگوریتم ----------------
    print("\n=== سهم هر گروه از کل نقض‌های همان الگوریتم (violation_share) ===\n")
    print(header)
    print("-" * len(header))
    total_violated_by_algo = {
        name: sum(per_algo[name]["violated"].values()) or 1 for name in algo_names
    }
    for grp in SERVICE_CHAIN_GROUPS:
        label = GROUP_LABELS[grp]
        row = f"{label:22s}"
        for name in algo_names:
            stats = per_algo[name]
            grp_violated = sum(stats["violated"].get(sid, 0) for sid in grp)
            share = 100.0 * grp_violated / total_violated_by_algo[name]
            row += f"{share:13.2f}%"
        print(row)

    # ---------------- فاصله‌ی بین الگوریتم‌ها در هر گروه ----------------
    print("\n=== فاصله‌ی max-min نرخ نقض بین الگوریتم‌ها، به‌ازای هر گروه ===\n")
    for grp in SERVICE_CHAIN_GROUPS:
        rates = list(group_rates[grp].values())
        spread = max(rates) - min(rates)
        print(f"{GROUP_LABELS[grp]:22s} فاصله: {spread:6.2f}pp   "
              f"(min={min(rates):.2f}%  max={max(rates):.2f}%)")

    print("\n--- تفسیر ---")
    print("اگر فاصله‌ی max-min در G1 (deadline سفت، queue_len=1-2) کوچک باشد ولی")
    print("در G2-G5 (deadline بازتر) بزرگ‌تر باشد، یعنی الگوریتم‌ها واقعاً از هم")
    print("افتراق دارند ولی چون G1 سهم بزرگی از violation_share کل را می‌گیرد و")
    print("تقریباً برای همه یکسان است (سقف فیزیکی، نه محدودیت سیاست)، این افتراق")
    print("در معیار تجمیعی deadline_violation_rate_pct محو می‌شود.")
    print("اگر برعکس، فاصله در همه‌ی گروه‌ها کوچک بود، مشکل واقعاً همگرایی")
    print("ساختاری الگوریتم‌هاست، نه صرفاً یک معیار گزارش‌دهی گمراه‌کننده.")


if __name__ == "__main__":
    main()