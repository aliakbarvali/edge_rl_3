"""
analyze_decision_quality.py
تشخیص می‌دهد که آیا اکشن‌های SCALE_UP/SCALE_DOWن که معیار ممیزی مستقل بخش ۸
(occ_ratio > 0.7 یا < 0.2) آن‌ها را «غیرضروری» علامت زده، واقعاً نویز/زیرآموزش
بوده‌اند یا رفتار پیش‌بینانه‌ی (anticipatory) هوشمند.

منطق:
  SCALE_UP «غیرضروری» طبق ممیزی لحظه‌ای:
      - اگر طی LOOKAHEAD_TICKS تیک بعدی همان سرویس واقعاً به آستانه‌ی نیاز
        رسید (necessary_scale_up=True شد) یا درخواستی از آن سرویس رد شد ->
        "anticipatory" (پیش‌بینانه‌ی موجّه؛ عامل زودتر از موعد اما درست عمل کرد)
      - وگرنه -> "noise" (به‌احتمال زیاد اکشن واقعاً زائد بوده)

  SCALE_DOWN «غیرضروری» طبق ممیزی لحظه‌ای:
      - اگر طی LOOKAHEAD_TICKS تیک بعدی همان سرویس دچار کمبود واقعی شد
        (necessary_scale_up=True یا رد درخواست) -> "risky" (این SCALE_DOWN
        به‌احتمال زیاد باعث کمبود ظرفیت بعدی شده - این بدترین دسته است)
      - وگرنه -> "harmless_early" (ظرفیت اضافی را کمی زودتر از حد رسمی آزاد
        کرده، ولی مشکلی هم ایجاد نکرده)

ورودی: مسیر یک فایل *_events.jsonl که با EventLogger تولید شده (باید شامل
رویدادهای scale_decision و request_rejected باشد - نسخه‌ی به‌روزشده‌ی
simulator/engine.py که این‌ها را لاگ می‌کند لازم است).

اجرا:
    python3 analyze_decision_quality.py outputs/ppo_events.jsonl
"""

from __future__ import annotations
import json
import sys
from collections import defaultdict

LOOKAHEAD_TICKS = 3          # چند تیک تصمیم بعدی را بررسی کنیم (۳ تیک = ۹۰ ثانیه با DECISION_INTERVAL_SEC=30)
REJECTION_LOOKAHEAD_SEC = 90  # پنجره‌ی زمانی مشابه برای جست‌وجوی رد درخواست واقعی


def load_events(path: str):
    scale_by_service = defaultdict(list)   # service_id -> [رکورد scale_decision به ترتیب زمان]
    rejections_by_service = defaultdict(list)  # service_id -> [sim_time رد شدن]

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            et = rec.get("event_type")
            if et == "scale_decision":
                sid = rec.get("service_id")
                if sid is None:
                    continue
                scale_by_service[sid].append(rec)
            elif et == "request_rejected":
                sid = rec.get("service_id")
                if sid is None:
                    continue
                rejections_by_service[sid].append(rec.get("sim_time_sec"))

    for sid in scale_by_service:
        scale_by_service[sid].sort(key=lambda r: r["sim_time_sec"])
    return scale_by_service, rejections_by_service


def had_rejection_within(rejections: list, start_time: float, end_time: float) -> bool:
    return any(start_time < t <= end_time for t in rejections if t is not None)


def classify(scale_by_service, rejections_by_service, lookahead_ticks=LOOKAHEAD_TICKS):
    result = {
        "SCALE_UP": {"correct_now": 0, "anticipatory": 0, "noise": 0},
        "SCALE_DOWN": {"correct_now": 0, "harmless_early": 0, "risky": 0},
    }
    examples = {"SCALE_UP": {"anticipatory": [], "noise": []},
                "SCALE_DOWN": {"harmless_early": [], "risky": []}}

    for sid, ticks in scale_by_service.items():
        rejections = rejections_by_service.get(sid, [])
        for i, t in enumerate(ticks):
            if not t.get("applied"):
                continue
            decision = t.get("decision")
            window = ticks[i + 1: i + 1 + lookahead_ticks]
            window_end_time = window[-1]["sim_time_sec"] if window else t["sim_time_sec"]

            if decision == "SCALE_UP":
                if t.get("necessary_scale_up"):
                    result["SCALE_UP"]["correct_now"] += 1
                    continue
                became_necessary = any(w.get("necessary_scale_up") for w in window)
                rejected_soon = had_rejection_within(rejections, t["sim_time_sec"], window_end_time)
                if became_necessary or rejected_soon:
                    result["SCALE_UP"]["anticipatory"] += 1
                    if len(examples["SCALE_UP"]["anticipatory"]) < 3:
                        examples["SCALE_UP"]["anticipatory"].append(
                            (sid, t["sim_time_sec"], "occ رسید" if became_necessary else "رد رخ داد"))
                else:
                    result["SCALE_UP"]["noise"] += 1
                    if len(examples["SCALE_UP"]["noise"]) < 3:
                        examples["SCALE_UP"]["noise"].append((sid, t["sim_time_sec"]))

            elif decision == "SCALE_DOWN":
                if t.get("necessary_scale_down"):
                    result["SCALE_DOWN"]["correct_now"] += 1
                    continue
                became_short = any(w.get("necessary_scale_up") for w in window)
                rejected_soon = had_rejection_within(rejections, t["sim_time_sec"], window_end_time)
                if became_short or rejected_soon:
                    result["SCALE_DOWN"]["risky"] += 1
                    if len(examples["SCALE_DOWN"]["risky"]) < 3:
                        examples["SCALE_DOWN"]["risky"].append(
                            (sid, t["sim_time_sec"], "کمبود occ" if became_short else "رد رخ داد"))
                else:
                    result["SCALE_DOWN"]["harmless_early"] += 1
                    if len(examples["SCALE_DOWN"]["harmless_early"]) < 3:
                        examples["SCALE_DOWN"]["harmless_early"].append((sid, t["sim_time_sec"]))

    return result, examples


def main():
    if len(sys.argv) < 2:
        print("استفاده: python3 analyze_decision_quality.py <path-to-events.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    scale_by_service, rejections_by_service = load_events(path)
    result, examples = classify(scale_by_service, rejections_by_service)

    print(f"فایل: {path}")
    print(f"پنجره‌ی lookahead: {LOOKAHEAD_TICKS} تیک تصمیم\n")

    for kind, stats in result.items():
        total = sum(stats.values())
        print(f"=== {kind} (کل اقدامات اعمال‌شده: {total}) ===")
        for label, count in stats.items():
            pct = 100.0 * count / total if total else 0.0
            print(f"  {label:16s}: {count:5d}  ({pct:5.1f}%)")
        print()

    print("--- نمونه‌ها ---")
    for kind, cats in examples.items():
        for label, exs in cats.items():
            if exs:
                print(f"{kind} / {label}:")
                for ex in exs:
                    print(f"    {ex}")

    print("\n--- تفسیر ---")
    su = result["SCALE_UP"]
    sd = result["SCALE_DOWN"]
    su_incorrect = su["anticipatory"] + su["noise"]
    if su_incorrect:
        print(f"از {su_incorrect} SCALE_UP «غیرضروری طبق لحظه»: "
              f"{su['anticipatory']} تا ({100*su['anticipatory']/su_incorrect:.0f}%) واقعاً پیش‌بینانه/موجّه بودند، "
              f"{su['noise']} تا ({100*su['noise']/su_incorrect:.0f}%) به‌نظر نویز واقعی می‌رسند.")
    sd_incorrect = sd["harmless_early"] + sd["risky"]
    if sd_incorrect:
        print(f"از {sd_incorrect} SCALE_DOWN «غیرضروری طبق لحظه»: "
              f"{sd['risky']} تا ({100*sd['risky']/sd_incorrect:.0f}%) به‌نظر واقعاً *مضر* می‌رسند "
              f"(بلافاصله بعدش کمبود/رد رخ داد)، "
              f"{sd['harmless_early']} تا ({100*sd['harmless_early']/sd_incorrect:.0f}%) بی‌ضرر بودند.")


if __name__ == "__main__":
    main()
