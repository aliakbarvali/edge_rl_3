"""
diagnose_occ_vs_rejection.py

هدف: تشخیص این‌که در تصمیم SCALE_UP هر سرویس، کدام شرط واقعاً "دارد
trigger می‌زند" - occ_ratio (اشغال صف) یا rejection_rate (رد شدن). این
اسکریپت برای پاسخ به این سؤال ساخته شده: چرا تغییر OCC_UP_TRIGGER در
VoilaAlgorithm.scale_decision تقریباً هیچ اثری روی نتایج نهایی نداشت؟

پیش‌نیاز: باید ابتدا simulator/engine.py و k8s_adapter/realtime_dispatcher.py
را طوری پچ کنید که فیلدهای occ_ratio و rejection_rate هم در رویداد
scale_decision لاگ شوند (نگاه کنید پچ پیشنهادی زیر). بدون این پچ، این
اسکریپت فقط با تخمین غیرمستقیم از necessary_scale_up کار می‌کند که چون
خودش OR چند سیگنال است (occ_ratio>0.85 یا rejection>0 یا deadline_violation>0)
نمی‌تواند دقیقاً occ را از rejection تفکیک کند.

--- پچ پیشنهادی simulator/engine.py (در _apply_scale_decision) ---
قبل از خط self._log("scale_decision", ...)، اضافه کنید:

    sv = snapshot["services"][svc_id]
    occ_ratio_dbg = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0

و در خودِ فراخوانی self._log اضافه کنید:
    occ_ratio=occ_ratio_dbg, rejection_rate=sv["rejection_rate"]

همین دو خط را عیناً در k8s_adapter/realtime_dispatcher.py:_apply_scale_decision
هم اضافه کنید (طبق قانون هم‌گام‌سازی پروژه بین دو موتور).

اجرا (بعد از پچ و اجرای مجدد):
    python diagnose_occ_vs_rejection.py outputs/seed42/voila_events.jsonl
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict

OCC_UP_TRIGGER = 0.80  # اگر مقدار دیگری استفاده کردید، اینجا هم عوض کنید


def main():
    if len(sys.argv) < 2:
        print("استفاده: python diagnose_occ_vs_rejection.py <path-to-events.jsonl>")
        sys.exit(1)

    path = sys.argv[1]

    total_scale_up_applied = 0
    via_occ_only = 0
    via_rejection_only = 0
    via_both = 0
    via_neither_but_applied = 0   # یعنی فیلدهای دیباگ در لاگ نیستند یا فقط از مسیر proximity آمده

    occ_values_at_scaleup = []
    has_debug_fields = False

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event_type") != "scale_decision":
                continue
            if rec.get("decision") != "SCALE_UP" or not rec.get("applied"):
                continue

            total_scale_up_applied += 1

            if "occ_ratio" in rec and "rejection_rate" in rec:
                has_debug_fields = True
                occ = rec["occ_ratio"]
                rej = rec["rejection_rate"]
                occ_values_at_scaleup.append(occ)

                occ_fired = occ > OCC_UP_TRIGGER
                rej_fired = rej > 0.0

                if occ_fired and rej_fired:
                    via_both += 1
                elif occ_fired:
                    via_occ_only += 1
                elif rej_fired:
                    via_rejection_only += 1
                else:
                    via_neither_but_applied += 1  # احتمالاً از مسیر proximity_violation آمده (فقط VOILA)

    print(f"فایل: {path}")
    print(f"تعداد کل SCALE_UP اعمال‌شده: {total_scale_up_applied}\n")

    if not has_debug_fields:
        print("!! فیلدهای occ_ratio/rejection_rate در این لاگ موجود نیستند.")
        print("   ابتدا پچ بالای فایل را در simulator/engine.py و")
        print("   k8s_adapter/realtime_dispatcher.py اعمال کنید و دوباره اجرا بگیرید.")
        sys.exit(0)

    print(f"trigger شده فقط با occ_ratio > {OCC_UP_TRIGGER}: {via_occ_only} "
          f"({100*via_occ_only/total_scale_up_applied:.1f}%)")
    print(f"trigger شده فقط با rejection_rate > 0:            {via_rejection_only} "
          f"({100*via_rejection_only/total_scale_up_applied:.1f}%)")
    print(f"trigger شده با هر دو هم‌زمان:                      {via_both} "
          f"({100*via_both/total_scale_up_applied:.1f}%)")
    print(f"نه occ نه rejection (احتمالاً proximity - فقط VOILA): {via_neither_but_applied} "
          f"({100*via_neither_but_applied/total_scale_up_applied:.1f}%)")

    if occ_values_at_scaleup:
        import statistics
        print(f"\nتوزیع occ_ratio در لحظه‌ی SCALE_UP (n={len(occ_values_at_scaleup)}):")
        print(f"  min={min(occ_values_at_scaleup):.3f}  "
              f"median={statistics.median(occ_values_at_scaleup):.3f}  "
              f"mean={statistics.mean(occ_values_at_scaleup):.3f}  "
              f"max={max(occ_values_at_scaleup):.3f}")

    print("\n--- تفسیر ---")
    dominant_pct = 100 * via_rejection_only / total_scale_up_applied
    if dominant_pct > 50:
        print(f"rejection_rate>0 به‌تنهایی مسئول {dominant_pct:.0f}% از SCALE_UPهاست -")
        print("یعنی occ_ratio/OCC_UP_TRIGGER عملاً بی‌اثر است چون رد شدن همیشه زودتر")
        print("رخ می‌دهد. برای این‌که تغییر OCC_UP_TRIGGER واقعاً اثر بگذارد، باید یا")
        print("queue_len سرویس‌ها را بزرگ‌تر کنید (تا رد شدن دیرتر رخ دهد و occ_ratio")
        print("فرصت trigger زدن پیدا کند)، یا rejection را به یک شرط ثانویه/ضعیف‌تر تبدیل")
        print("کنید (مثلاً rejection_rate > 0.1 به‌جای >0)، یا هر دو سیگنال را جدا بسنجید")
        print("و لاگ کنید که کدام الگوریتم واقعاً دارد از سیگنال جغرافیایی/ظرفیتی خودش")
        print("استفاده می‌کند - نه فقط از یک فال‌بک مشترک.")
    else:
        print("occ_ratio سهم قابل‌توجهی دارد - تغییر OCC_UP_TRIGGER باید قابل‌مشاهده باشد.")
        print("اگر با این حال نتایج نهایی تغییر نکرده، مشکل جای دیگری است (مثلاً cooldown")
        print("یا سقف ظرفیت سرورهای OFF/on که SCALE_UP را قبل از رسیدن به scale_decision")
        print("مسدود می‌کند - نگاه کنید به skip_reason در همین رویدادها).")


if __name__ == "__main__":
    main()
