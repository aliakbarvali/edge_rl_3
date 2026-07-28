"""
data/generate_sample_trace.py
==============================
چون فایل واقعی دیتاست شانگهای را هنوز در اختیار ندارم، این اسکریپت یک
فایل CSV *ساختگی* با دقیقاً همان ستون‌های موردانتظار می‌سازد
(id, BTSID, StartTime, Lat, Long, ServiceID) تا بتوانیم کل پایپ‌لاین
(بارگذاری -> چیدمان اولیه -> شبیه‌سازی -> متریک) را از همین حالا تست کنیم.

⚠️ وقتی فایل واقعی را در اختیار گذاشتید، به‌جای اجرای این اسکریپت،
مسیر common/config.py -> REQUEST_TRACE_PATH را به فایل واقعی اشاره
دهید و این اسکریپت را دیگر لازم ندارید.

نحوه‌ی اجرا:
    python data/generate_sample_trace.py
خروجی در data/sample_requests.csv نوشته می‌شود.
"""

from __future__ import annotations
import csv
import random
from datetime import datetime, timedelta

# محدوده‌ی جغرافیایی تقریبی شانگهای (کمی گسترده‌تر از محدوده‌ی سرورها،
# تا هم درخواست‌های نزدیک به سرورها هم دورتر تولید شود)
LAT_RANGE = (30.80, 31.45)
LONG_RANGE = (121.10, 121.80)

NUM_REQUESTS = 5000
NUM_DISTINCT_BTS = 200          # تعداد BTSهای مبدأ متفاوت (بیشتر از ۱۰ سرور)
DURATION_HOURS = 24
OUTPUT_PATH = "data/sample_requests.csv"

# سرویس‌های کوچک (مثل ۱ تا ۵) طبیعتاً پرتکرارترند؛ همین وزن‌دهی ساده را
# برای واقعی‌تر بودن ترافیک شبیه‌سازی‌شده اعمال می‌کنیم.
SERVICE_WEIGHTS = {
    1: 20, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 6, 7: 5, 8: 4, 9: 3, 10: 2,
    11: 1.5, 12: 1.2, 13: 1.0, 14: 0.8, 15: 0.5,
}


def main():
    random.seed(42)

    bts_pool = [
        {"bts_id": f"synthetic_bts_{i:04d}", "lat": random.uniform(*LAT_RANGE), "long": random.uniform(*LONG_RANGE)}
        for i in range(NUM_DISTINCT_BTS)
    ]

    service_ids = list(SERVICE_WEIGHTS.keys())
    weights = list(SERVICE_WEIGHTS.values())

    start_dt = datetime(2024, 1, 1, 0, 0, 0)
    total_seconds = DURATION_HOURS * 3600

    rows = []
    for i in range(NUM_REQUESTS):
        bts = random.choice(bts_pool)
        offset_sec = random.uniform(0, total_seconds)
        request_time = start_dt + timedelta(seconds=offset_sec)
        service_id = random.choices(service_ids, weights=weights, k=1)[0]
        rows.append(
            {
                "id": f"req_{i:06d}",
                "BTSID": bts["bts_id"],
                "StartTime": request_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Lat": round(bts["lat"], 6),
                "Long": round(bts["long"], 6),
                "ServiceID": service_id,
            }
        )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "BTSID", "StartTime", "Lat", "Long", "ServiceID"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{NUM_REQUESTS} درخواست ساختگی در {OUTPUT_PATH} نوشته شد.")


if __name__ == "__main__":
    main()
