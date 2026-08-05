"""
k8s_adapter/bts_simulator.py

چون در این محیط توسعه/دمو هیچ BTS واقعی‌ای وجود ندارد که خودش درخواست
بفرستد، این اسکریپت دقیقاً همان نقشی را بازی می‌کند که در معماری واقعی
BTS بازی می‌کرد: دیتاست تاریخی را با زمان‌بندی واقعی (real-clock replay)
پخش می‌کند و برای هر رویداد *دو* تماس HTTP واقعی و جدا می‌زند - دقیقاً
مثل یک کلاینت واقعی، نه یک shortcut داخل پروسه‌ی دیسپچر:

    ۱) POST به دیسپچر مرکزی (dispatcher_api.py:/route) - سبک، فقط مسیریابی
    ۲) POST مستقیم به IP:PORT پاد (که از جواب مرحله‌ی ۱ گرفته) - سنگین،
       شامل payload واقعی پردازش

این دو مرحله عمداً با httpx واقعی (نه فراخوانی مستقیم تابع پایتون) انجام
می‌شود تا جداییِ control-plane/data-plane واقعاً روی شبکه تست شود، نه فقط
در حافظه‌ی یک پروسه شبیه‌سازی شود.

اجرا (بعد از اینکه dispatcher_api.py و worker podها بالا هستند):
    python3 -m k8s_adapter.bts_simulator --dispatcher-host 192.168.1.30 --dispatcher-port 9000
"""

from __future__ import annotations
import argparse
import asyncio
import time

import httpx
import pandas as pd


async def _send_one_request(row, http_client: httpx.AsyncClient, dispatcher_url: str,
                             request_seq: int):
    # مرحله ۱: مسیریابی - تماس سبک با دیسپچر مرکزی
    try:
        route_resp = await http_client.post(f"{dispatcher_url}/route", json={
            "request_id": request_seq, "service_id": int(row.ServiceID),
            "bts_lat": float(row.Lat), "bts_long": float(row.Long),
        }, timeout=10)
        route_data = route_resp.json()
    except Exception:
        return  # دیسپچر در دسترس نبود؛ در سیستم واقعی BTS این را retry/لاگ می‌کند

    if route_data.get("status") != "ROUTED":
        return  # REJECTED_QUEUE_FULL / REJECTED_NO_REPLICA - چیزی برای فاز ۲ نیست

    # مرحله ۲: تماس سنگین - مستقیم به خودِ پاد، بدون واسطه‌ی دیسپچر
    ip, port = route_data["ip"], route_data["port"]
    deadline = route_data["deadline_sec"]
    try:
        await http_client.post(f"http://{ip}:{port}/process", json={
            "request_id": request_seq, "sent_at_epoch": time.time(),
        }, timeout=deadline + 10)
    except Exception:
        pass  # پاد جواب نداد/کرش کرد - پاد خودش مسئول گزارش عدم‌موفقیت نیست؛
              # این محدودیت شناخته‌شده‌ای است که در ادامه (retry/health-check
              # سمت BTS) باید اضافه شود.


async def replay(events_df: pd.DataFrame, dispatcher_host: str, dispatcher_port: int):
    dispatcher_url = f"http://{dispatcher_host}:{dispatcher_port}"
    base_time = float(events_df.global_start_sec.min())
    wall_start = time.monotonic()
    seq = 0

    async with httpx.AsyncClient() as http_client:
        tasks = []
        for row in events_df.itertuples(index=False):
            target_offset = float(row.global_start_sec) - base_time
            now_offset = time.monotonic() - wall_start
            if target_offset > now_offset:
                await asyncio.sleep(target_offset - now_offset)
            seq += 1
            tasks.append(asyncio.create_task(_send_one_request(row, http_client, dispatcher_url, seq)))
        await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher-host", default="192.168.1.30")
    parser.add_argument("--dispatcher-port", type=int, default=9000)
    parser.add_argument("--data", default="test", choices=["train", "test"])
    args = parser.parse_args()

    from data.loader import load_train, load_test
    events = load_train() if args.data == "train" else load_test()
    asyncio.run(replay(events, args.dispatcher_host, args.dispatcher_port))


if __name__ == "__main__":
    main()