"""
diagnose_violations_by_service.py

تشخیص می‌دهد که نرخ نقض deadline (که در هر ۴ الگوریتم تقریباً یکسان است)
از کدام سرویس‌ها می‌آید: آیا فقط چند سرویس با deadline خیلی سفت مقصرند
(که یعنی مشکل ساختاری/تنظیمات است، نه رفتار الگوریتم)، یا نقض به‌طور
یکنواخت بین همه‌ی ۱۵ سرویس پخش شده (که یعنی مشکل در صف/ظرفیت است).

اجرا:
    python diagnose_violations_by_service.py outputs/seed45/greedy_events.jsonl
    python diagnose_violations_by_service.py outputs/seed45/ppo_events.jsonl
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("استفاده: python3 diagnose_violations_by_service.py <path-to-events.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    total_by_svc = defaultdict(int)
    completed_by_svc = defaultdict(int)
    violated_by_svc = defaultdict(int)
    rejected_by_svc = defaultdict(int)
    rt_sum_by_svc = defaultdict(float)
    rt_max_by_svc = defaultdict(float)

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
                total_by_svc[sid] += 1
            elif et == "request_completed":
                completed_by_svc[sid] += 1
                rt = rec.get("response_time_sec", 0.0) or 0.0
                rt_sum_by_svc[sid] += rt
                rt_max_by_svc[sid] = max(rt_max_by_svc[sid], rt)
            elif et == "request_rejected":
                rejected_by_svc[sid] += 1
                violated_by_svc[sid] += 1  # رد شدن هم نقض حساب می‌شود

    from common.config import CFG
    print(f"فایل: {path}\n")
    print(f"{'svc':>3} {'deadline_ms':>11} {'arrivals':>9} {'completed':>10} "
          f"{'rejected':>9} {'avg_rt_ms':>10} {'max_rt_ms':>10} {'violation_share':>16}")

    grand_total_arrivals = sum(total_by_svc.values()) or 1
    rows = []
    for sid in sorted(CFG.services_info.keys()):
        arrivals = total_by_svc.get(sid, 0)
        completed = completed_by_svc.get(sid, 0)
        rejected = rejected_by_svc.get(sid, 0)
        dl_ms = CFG.services_info[sid]["deadline"] * 1000
        avg_rt = (rt_sum_by_svc.get(sid, 0.0) / completed * 1000) if completed else 0.0
        max_rt = rt_max_by_svc.get(sid, 0.0) * 1000
        # از آنجا که deadline_violated داخل events لاگ نشده به‌صورت مستقیم در همه نسخه‌ها،
        # این اسکریپت را می‌توانید با اضافه‌کردن deadline_violated=req.deadline_violated
        # به لاگ request_completed در engine.py دقیق‌تر کنید. فعلاً از میانگین/بیشینه‌ی
        # response_time در مقابل deadline برای برآورد استفاده می‌کنیم.
        share = arrivals / grand_total_arrivals * 100
        rows.append((sid, dl_ms, arrivals, completed, rejected, avg_rt, max_rt, share))

    for sid, dl_ms, arrivals, completed, rejected, avg_rt, max_rt, share in rows:
        flag = "  <<< میانگین از deadline بیشتر است!" if avg_rt > dl_ms else ""
        print(f"{sid:>3} {dl_ms:>11.1f} {arrivals:>9} {completed:>10} "
              f"{rejected:>9} {avg_rt:>10.2f} {max_rt:>10.2f} {share:>15.2f}%{flag}")

    print("\nنکته: اگر avg_rt_ms ستون چند سرویس خاص (مثلاً svc1/svc2) به‌طور "
          "سیستماتیک نزدیک یا بالاتر از deadline خودشان باشد در حالی که سهم "
          "زیادی از کل ترافیک (ستون آخر) را هم دارند، نقض ۴۰٪ عمدتاً ساختاری "
          "است (بند ۴ گزارش) نه ضعف الگوریتم.")


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
