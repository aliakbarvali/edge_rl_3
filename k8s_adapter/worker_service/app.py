"""
k8s_adapter/worker_service/app.py

*** پچ (رفع باگ ۱ - cold-start penalty گم‌شده در مسیر k8s): simulator/engine.py
هنگام سرو یک درخواست روی رپلیکای «تازه READY‌شده» (within cold_start_window)
یک جریمه‌ی اضافه (compute_cold_start_penalty_sec) به زمان اجرا اضافه می‌کند
(_handle_routed -> try_admit(cold_start_extra=...)). این پاد واقعی این منطق
را اصلاً نداشت - همیشه فقط EXEC_TIME_SEC می‌خوابید، صرف‌نظر از تازه بودن پاد -
یعنی deadline-violation در --mode k8s سیستماتیک بهتر از --mode sim گزارش
می‌شد. چون این image فقط app.py+requirements.txt دارد (نه ماژول common/)،
مقادیر پنجره/جریمه به‌جای import مستقیم common.config، از دو env var جدید
(COLD_START_WINDOW_SEC/COLD_START_PENALTY_SEC) خوانده می‌شوند که
k8s_client.py:build_deployment_manifest از قبل با compute_cold_start_window_sec/
compute_cold_start_penalty_sec محاسبه و ست می‌کند - دقیقاً همان الگویی که
EXEC_TIME_SEC از قبل استفاده می‌کرد.

تقریب لازم: در sim، «تازه بودن» از روی replica.ready_since (لحظه‌ی دقیق
READY شدن رپلیکا) سنجیده می‌شود. اینجا معادل دقیقش در دسترس نیست، پس زمان
start این پروسه (_pod_started_at) به‌عنوان تقریب replica.ready_since استفاده
می‌شود - اختلافش با ready_since واقعی حداکثر به‌اندازه‌ی initial_delay_seconds
(1s)+period_seconds(2s)*failure_threshold(3) در Dockerfile readiness_probe
است، که در مقابل COLD_START_WINDOW_SEC (تا سقف ۱۰ ثانیه) قابل چشم‌پوشی است.

*** تغییر معماری قبلی (بدون تغییر): release_queue_slot و ثبت متریک نهایی
توسط خودِ این پاد، بلافاصله بعد از اتمام پردازش انجام می‌شود.
"""

import os
import asyncio
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
import redis

app = FastAPI()

EXEC_TIME_SEC = float(os.environ.get("EXEC_TIME_SEC", "1.0"))
SERVICE_ID = int(os.environ.get("SERVICE_ID", "0"))
SERVER_ID = int(os.environ.get("SERVER_ID", "0"))
COLD_START_WINDOW_SEC = float(os.environ.get("COLD_START_WINDOW_SEC", "0.0"))
COLD_START_PENALTY_SEC = float(os.environ.get("COLD_START_PENALTY_SEC", "0.0"))
_pod_started_at = time.monotonic()

REDIS_HOST = os.environ.get("REDIS_HOST", "192.168.1.30")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

_process_semaphore = asyncio.Semaphore(1)


class ProcessRequest(BaseModel):
    request_id: int | None = None
    # *** BTS زمان دقیق ارسال درخواست را هم می‌فرستد تا response_time واقعی
    # (شامل زمان شبکه‌ی رفت) از دید خودِ BTS/worker هم قابل محاسبه باشد،
    # نه فقط از دید کلاینت. اختیاری - اگر نیاید فقط exec_time لحاظ می‌شود.
    sent_at_epoch: float | None = None
 


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service_id": SERVICE_ID, "server_id": SERVER_ID}


@app.post("/process")
async def process(req: ProcessRequest):
    # *** پچ cold-start: فقط طی پنجره‌ی اول عمر همین پاد اعمال می‌شود - نگاه
    # کنید توضیح بالای فایل. بعد از عبور از COLD_START_WINDOW_SEC همیشه صفر
    # می‌ماند، دقیقاً هم‌راستا با رفتار try_admit در simulator/engine.py.
    cold_start_extra = 0.0
    if COLD_START_WINDOW_SEC > 0 and (time.monotonic() - _pod_started_at) <= COLD_START_WINDOW_SEC:
        cold_start_extra = COLD_START_PENALTY_SEC

    async with _process_semaphore:
        start = time.monotonic()
        await asyncio.sleep(EXEC_TIME_SEC + cold_start_extra)
        elapsed = time.monotonic() - start

    # *** پچ (رفع باگ ۲ - دبل-دیکریمنت رزرو صف): قبلاً شمارنده‌ی صف بدون هیچ
    # چک idempotency‌ای decrement می‌شد. اگر پردازش (به‌خاطر صف پر، نه کرش)
    # بیشتر از reservation_ttl_sec طول بکشد، _reservation_sweeper_loop
    # (redis_state.py) ممکن است زودتر همین رزرو را «منقضی‌شده» تشخیص دهد و
    # شمارنده را یک واحد آزاد کند؛ اگر بعد از آن این پاد هم بدون قید‌وشرط
    # decrement کند، شمارنده واقعی *کمتر* از اشغال واقعی صف نشان داده می‌شود
    # (under-count) - سیگنال occ_ratio که مستقیماً به تصمیم scale-up/down
    # می‌رود گمراه می‌شود.
    # راه‌حل: ZREM خودِ همین request_id از ZSET رزرو تنها گیت مجاز decrement
    # است - چون ZREM اتمیک است، بین این پاد و sweeper هرکدام زودتر برسد واقعاً
    # موفق می‌شود (return=1) و دیگری ناموفق (return=0)، پس شمارنده دقیقاً
    # یک‌بار (نه صفر، نه دوبار) کم می‌شود. (نگاه کنید هم‌راستا:
    # redis_state.py:sweep_expired_reservations که همین منطق را دارد.)
    try:
        key = f"edge:replica:{SERVICE_ID}:{SERVER_ID}:queue"
        reservation_key = f"edge:reservations:{SERVICE_ID}:{SERVER_ID}"
        if req.request_id is not None:
            removed = _redis.zrem(reservation_key, str(req.request_id))
            should_decrement = bool(removed)
        else:
            # بدون request_id امکان idempotency-check نیست؛ رفتار قبلی
            # (قید‌وشرط‌نشده) حفظ می‌شود - عملاً هرگز رخ نمی‌دهد چون
            # bts_simulator.py همیشه request_id می‌فرستد.
            should_decrement = True
        if should_decrement and int(_redis.get(key) or 0) > 0:
            _redis.decr(key)
    except Exception:
        pass  # نبود Redis نباید کل پردازش را بشکند؛ فقط صف کمی ناهماهنگ می‌ماند

    response_time_sec = elapsed
    if req.sent_at_epoch is not None:
        response_time_sec = time.time() - req.sent_at_epoch

    try:
        _redis.rpush("edge:metrics:completions", __import__("json").dumps({
            "request_id": req.request_id, "service_id": SERVICE_ID, "server_id": SERVER_ID,
            "success": True, "response_time_sec": response_time_sec,
        }))
    except Exception:
        pass
    try:
        _redis.incrbyfloat(f"service:{SERVICE_ID}:server:{SERVER_ID}:busy_seconds_acc", elapsed)
    except Exception:
        pass
    return {
        "request_id": req.request_id,
        "service_id": SERVICE_ID,
        "server_id": SERVER_ID,
        "exec_time_sec": elapsed,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }