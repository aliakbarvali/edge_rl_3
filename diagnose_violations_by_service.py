"""
diagnose_violations_by_service.py  (نسخه‌ی اصلاح‌شده)

تشخیص می‌دهد که نرخ نقض deadline (که در هر ۴ الگوریتم تقریباً یکسان است)
از کدام سرویس‌ها می‌آید: آیا فقط چند سرویس با deadline خیلی سفت مقصرند
(که یعنی مشکل ساختاری/تنظیمات است، نه رفتار الگوریتم)، یا نقض به‌طور
یکنواخت بین همه‌ی ۱۵ سرویس پخش شده (که یعنی مشکل در صف/ظرفیت است).

--- تغییرات نسبت به نسخه‌ی قبلی ---
باگ اصلی نسخه‌ی قبلی: ستونی که «violation_share» نام‌گذاری شده بود، در واقع
فقط arrivals[sid] / sum(arrivals) بود، یعنی «سهم هر سرویس از کل ترافیک»،
نه سهم آن سرویس از نقض‌های deadline. متغیر violated_by_svc قبلاً هم فقط با
رد شدن‌ها (rejected) پر می‌شد و هرگز برای request_completedهایی که دیرتر از
deadline تمام شده بودند به‌روزرسانی نمی‌شد، و اصلاً هم چاپ نمی‌شد.

این نسخه:
  ۱) برای هر request_completed چک می‌کند که آیا response_time_sec از
     deadline سرویس بیشتر شده یا نه (یا اگر رکورد خودش فیلد
     deadline_violated را همراه دارد، همان را معتبرتر می‌داند)، و اگر بله
     violated_by_svc[sid] را افزایش می‌دهد.
  ۲) رد شدن (request_rejected) هم مثل قبل به‌عنوان نقض حساب می‌شود.
  ۳) ستون violation_share حالا واقعاً violated_by_svc[sid] / کل نقض‌ها است.
  ۴) traffic_share (سهم از کل ترافیک) به‌عنوان یک ستون جدا و مجزا نگه
     داشته شده، چون خودش هم اطلاعات مفیدی است، ولی دیگر با violation_share
     اشتباه گرفته نمی‌شود.
  ۵) یک ستون violation_rate هم اضافه شده: violated_by_svc[sid] / arrivals[sid]
     یعنی «چند درصد از درخواست‌های همین سرویس نقض شدند» - این برای فهمیدن
     اینکه کدام سرویس نسبت به حجم خودش بدترین رفتار را دارد لازم است،
     چون violation_share به‌تنهایی سرویس‌های پرترافیک را طبیعتاً بزرگ‌تر
     نشان می‌دهد حتی اگر نرخ نقض‌شان کم باشد.

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

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common.config import CFG

    total_by_svc = defaultdict(int)
    completed_by_svc = defaultdict(int)
    violated_by_svc = defaultdict(int)
    rejected_by_svc = defaultdict(int)
    rt_sum_by_svc = defaultdict(float)
    rt_max_by_svc = defaultdict(float)

    # deadline هر سرویس بر حسب ثانیه (برای مقایسه‌ی مستقیم با response_time_sec)
    deadline_sec_by_svc = {sid: info["deadline"] for sid, info in CFG.services_info.items()}

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

                # منبع اول: اگر خودِ رکورد فیلد deadline_violated را دارد
                # (نسخه‌ی به‌روز engine.py)، همان معتبرتر است چون دقیقاً
                # همان منطقی را منعکس می‌کند که شبیه‌ساز موقع محاسبه‌ی
                # پاداش/متریک استفاده کرده. اگر نبود، از مقایسه‌ی مستقیم
                # response_time_sec با deadline به‌عنوان جایگزین استفاده
                # می‌کنیم.
                if "deadline_violated" in rec:
                    violated = bool(rec.get("deadline_violated"))
                else:
                    dl = deadline_sec_by_svc.get(sid)
                    violated = dl is not None and rt > dl

                if violated:
                    violated_by_svc[sid] += 1

            elif et == "request_rejected":
                rejected_by_svc[sid] += 1
                violated_by_svc[sid] += 1  # رد شدن هم نقض حساب می‌شود

    grand_total_arrivals = sum(total_by_svc.values()) or 1
    grand_total_violations = sum(violated_by_svc.values()) or 1

    print(f"فایل: {path}\n")
    header = (f"{'svc':>3} {'deadline_ms':>11} {'arrivals':>9} {'completed':>10} "
              f"{'rejected':>9} {'avg_rt_ms':>10} {'max_rt_ms':>10} "
              f"{'violated':>9} {'violation_rate':>15} {'violation_share':>16} {'traffic_share':>14}")
    print(header)

    rows = []
    for sid in sorted(CFG.services_info.keys()):
        arrivals = total_by_svc.get(sid, 0)
        completed = completed_by_svc.get(sid, 0)
        rejected = rejected_by_svc.get(sid, 0)
        violated = violated_by_svc.get(sid, 0)
        dl_ms = deadline_sec_by_svc[sid] * 1000
        avg_rt = (rt_sum_by_svc.get(sid, 0.0) / completed * 1000) if completed else 0.0
        max_rt = rt_max_by_svc.get(sid, 0.0) * 1000

        violation_rate = (violated / arrivals * 100) if arrivals else 0.0
        violation_share = violated / grand_total_violations * 100
        traffic_share = arrivals / grand_total_arrivals * 100

        rows.append((sid, dl_ms, arrivals, completed, rejected, avg_rt, max_rt,
                     violated, violation_rate, violation_share, traffic_share))

    for (sid, dl_ms, arrivals, completed, rejected, avg_rt, max_rt,
         violated, violation_rate, violation_share, traffic_share) in rows:
        flag = "  <<< میانگین از deadline بیشتر است!" if avg_rt > dl_ms else ""
        print(f"{sid:>3} {dl_ms:>11.1f} {arrivals:>9} {completed:>10} "
              f"{rejected:>9} {avg_rt:>10.2f} {max_rt:>10.2f} "
              f"{violated:>9} {violation_rate:>14.2f}% {violation_share:>15.2f}% "
              f"{traffic_share:>13.2f}%{flag}")

    print(f"\nمجموع نقض‌ها (رد + دیرکرد): {sum(violated_by_svc.values())}")

    print("\nنکته: violation_share سهم واقعی هر سرویس از کل نقض‌هاست (نه سهم "
          "ترافیک). اگر یک سرویس هم violation_share بالا و هم traffic_share "
          "بالا داشته باشد ولی violation_rate آن کم باشد، یعنی مشکل صرفاً "
          "«حجمی» است (این سرویس فقط چون درخواست زیاد دارد در اعداد مطلق "
          "بزرگ به‌نظر می‌رسد). اما اگر violation_rate یک سرویس به‌طور "
          "برجسته بالاتر از بقیه باشد (صرف‌نظر از traffic_share آن)، آن "
          "سرویس واقعاً از نظر deadline/ظرفیت مشکل ساختاری دارد و بند ۴ "
          "گزارش را توجیه می‌کند.")


if __name__ == "__main__":
    main()