"""
analyze_necessity_by_service.py

هدف: تفکیک این‌که «سیگنال occ_ratio ذاتاً برای سرویس‌های batch طولانی
(۱۱-۱۵) بیشتر necessary_scale_up=True می‌شود» از «PPO حتی وقتی سیگنال هم
necessary نیست بازهم SCALE_UP اعمال می‌کند».

برای هر سرویس، از تمام رویدادهای scale_decision (چه applied چه نه) دو عدد
حساب می‌کند:
  - necessity_rate: چند درصد تیک‌های تصمیم، necessary_scale_up=True بوده
    (این خودِ سیگنال ممیزی داخل موتور است - مستقل از الگوریتم)
  - applied_when_not_necessary_rate: از میان تیک‌هایی که necessary_scale_up
    False بوده، چند درصدشان SCALE_UP واقعاً applied شده (این خطای واقعی
    policy است، نه سیگنال)

اجرا:
    python3 analyze_necessity_by_service.py outputs/seed43/ppo_events.jsonl
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("استفاده: python3 analyze_necessity_by_service.py <path-to-events.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    total_ticks = defaultdict(int)
    necessary_ticks = defaultdict(int)
    applied_when_necessary = defaultdict(int)
    applied_when_not_necessary = defaultdict(int)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event_type") != "scale_decision":
                continue
            sid = rec.get("service_id")
            if sid is None:
                continue
            total_ticks[sid] += 1
            necessary = bool(rec.get("necessary_scale_up"))
            applied = bool(rec.get("applied")) and rec.get("decision") == "SCALE_UP"
            if necessary:
                necessary_ticks[sid] += 1
                if applied:
                    applied_when_necessary[sid] += 1
            else:
                if applied:
                    applied_when_not_necessary[sid] += 1

    print(f"فایل: {path}\n")
    print(f"{'svc':>3} {'ticks':>7} {'necessity_rate':>15} {'applied|necessary':>18} "
          f"{'applied|NOT_necessary':>22} {'share_of_all_bad_applies':>24}")

    total_bad = sum(applied_when_not_necessary.values()) or 1
    for sid in sorted(total_ticks.keys()):
        t = total_ticks[sid]
        nec_rate = 100.0 * necessary_ticks[sid] / t if t else 0.0
        good_apply_rate = (100.0 * applied_when_necessary[sid] / necessary_ticks[sid]
                            if necessary_ticks[sid] else 0.0)
        not_nec = t - necessary_ticks[sid]
        bad_apply_rate = (100.0 * applied_when_not_necessary[sid] / not_nec) if not_nec else 0.0
        bad_share = 100.0 * applied_when_not_necessary[sid] / total_bad
        print(f"{sid:>3} {t:>7} {nec_rate:>14.2f}% {good_apply_rate:>17.2f}% "
              f"{bad_apply_rate:>21.2f}% {bad_share:>23.2f}%")

    print("\nتفسیر:")
    print("- necessity_rate بالا برای svc11-15 یعنی خودِ سیگنال occ_ratio برای این")
    print("  سرویس‌ها ذاتاً بیشتر 'لازم' نشان می‌دهد (به‌خاطر exec_time طولانی) -")
    print("  این یک مسئله‌ی طراحی سیگنال است، نه لزوماً خطای PPO.")
    print("- 'applied|NOT_necessary' بالا یعنی PPO حتی وقتی سیگنال هم می‌گوید لازم")
    print("  نیست، بازهم SCALE_UP می‌زند - این خطای واقعی policy/reward-shaping است.")
    print("- ستون آخر (share_of_all_bad_applies) نشان می‌دهد از کل SCALE_UPهای")
    print("  'غیرضروری طبق سیگنال لحظه‌ای'، چند درصد مال کدام سرویس است -")
    print("  اگر بیشتر مال svc11-15 باشد، فرضیه‌ی 'سیگنال نویزی برای batch طولانی'")
    print("  تقویت می‌شود؛ اگر پخش یکنواخت باشد، مسئله در policy/reward است.")


if __name__ == "__main__":
    main()
