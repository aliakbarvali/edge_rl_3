#!/usr/bin/env python3
"""
analyze_scaleup_by_service.py

مکمل analyze_decision_quality.py: به‌جای فقط گزارش کلی correct_now/anticipatory/noise
برای SCALE_UP، این آمار را به‌ازای هر service_id تفکیک می‌کند و میانگین فاصله‌ی زمانی
بین دو SCALE_UP متوالیِ روی همان سرویس را هم گزارش می‌دهد (برای سنجش شدت flapping).

هدف: تست این فرضیه که بیشترین سهم SCALE_UPهای «نویز» مربوط به سرویس‌هایی با
deadline خیلی سفت (سرویس‌های ۱ و ۲) است -- یعنی عامل روی این سرویس‌ها محافظه‌کارانه
و بدون‌نیاز واقعی، بیش‌ازحد hedge می‌کند.

استفاده:
    python analyze_scaleup_by_service.py outputs/seed43/ppo_events.jsonl

فرض بر این است که رکوردهای scale_decision در JSONL این شکل را دارند (مطابق common/logger.py):
    {"event_type": "scale_decision", "service_id": <int>, "sim_time_sec": <float>,
     "action": "SCALE_UP"|"SCALE_DOWN"|"NO_CHANGE", "applied": true|false, ...}
اگر نام فیلدها در پروژه‌ی شما فرق دارد (مثلاً "decision" به‌جای "action"،
یا "time" به‌جای "sim_time_sec")، بخش CANDIDATE_KEYS پایین را ویرایش کنید.
"""

import sys
import json
from collections import defaultdict

CANDIDATE_ACTION_KEYS = ["action", "decision", "scale_action"]
CANDIDATE_TIME_KEYS = ["sim_time_sec", "time", "sim_time"]
CANDIDATE_SERVICE_KEYS = ["service_id", "svc", "svc_id"]
CANDIDATE_APPLIED_KEYS = ["applied", "was_applied", "executed"]
LOOKAHEAD_TICKS = 15
DECISION_INTERVAL_SEC = 30.0


def first_present(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main():
    if len(sys.argv) < 2:
        print("استفاده: python analyze_scaleup_by_service.py <path/to/events.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    events = load_events(path)

    scale_events = [
        e for e in events
        if e.get("event_type") == "scale_decision"
        and first_present(e, CANDIDATE_ACTION_KEYS) in ("SCALE_UP", "SCALE_DOWN")
    ]

    # آخرین snapshot متریک هر تیک تصمیم را برای سنجش necessary_up لازم داریم؛
    # اگر رکوردها همان فیلدهایی که analyze_decision_quality.py استفاده می‌کند دارند
    # (مثلاً occ_ratio یا rejection_rate)، این‌جا هم می‌شود از همان معیار استفاده کرد.
    # برای سادگی و مستقل بودن این اسکریپت از جزئیات دقیق ممیزی، این نسخه فقط
    # شمارش خام SCALE_UP اعمال‌شده به تفکیک سرویس و بازه‌ی زمانی بین رخدادهای
    # متوالی را حساب می‌کند -- برای طبقه‌بندی دقیق correct/anticipatory/noise
    # آن را کنار خروجی analyze_decision_quality.py بخوانید.

    per_service_times = defaultdict(list)
    for e in scale_events:
        action = first_present(e, CANDIDATE_ACTION_KEYS)
        applied = first_present(e, CANDIDATE_APPLIED_KEYS, default=True)
        if action != "SCALE_UP" or not applied:
            continue
        svc = first_present(e, CANDIDATE_SERVICE_KEYS)
        t = first_present(e, CANDIDATE_TIME_KEYS)
        if svc is None or t is None:
            continue
        per_service_times[svc].append(float(t))

    total_up = sum(len(v) for v in per_service_times.values())
    if total_up == 0:
        print("هیچ رکورد SCALE_UP اعمال‌شده‌ای با نام‌فیلدهای شناخته‌شده پیدا نشد.")
        print("نام فیلدهای موجود در اولین رکورد scale_decision:")
        if scale_events:
            print(sorted(scale_events[0].keys()))
        sys.exit(0)

    print(f"فایل: {path}")
    print(f"مجموع SCALE_UP اعمال‌شده: {total_up}\n")
    print(f"{'svc':>4} {'count':>7} {'share':>8} {'avg_gap_sec':>12} {'min_gap_sec':>12}")
    for svc in sorted(per_service_times.keys()):
        times = sorted(per_service_times[svc])
        gaps = [t2 - t1 for t1, t2 in zip(times, times[1:])]
        avg_gap = sum(gaps) / len(gaps) if gaps else float("nan")
        min_gap = min(gaps) if gaps else float("nan")
        share = len(times) / total_up * 100
        print(f"{svc:>4} {len(times):>7} {share:>7.2f}% {avg_gap:>12.1f} {min_gap:>12.1f}")

    print(
        "\nتفسیر: اگر count/share سرویس‌های ۱ و ۲ (سفت‌ترین deadlineها) به‌طور نامتناسب\n"
        "بزرگ‌تر از سهمشان از کل ترافیک باشد (سهم ترافیک را از diagnose_violations_by_service.py\n"
        "بگیرید)، و avg_gap_sec آن‌ها نزدیک به COOLDOWN_SEC (پیش‌فرض ۶۰ ثانیه) باشد،\n"
        "فرضیه‌ی 'hedging دفاعی مکرر روی سرویس‌های تنگ‌deadline بلافاصله بعد از پایان\n"
        "cooldown' تأیید می‌شود."
    )


if __name__ == "__main__":
    main()
