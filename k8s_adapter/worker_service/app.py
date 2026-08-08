"""
k8s_adapter/worker_service/app.py
... (docstring قبلی + این توضیح اضافه:)

*** تغییر معماری مهم: قبلاً release_queue_slot و ثبت متریک نهایی توسط
خودِ دیسپچر مرکزی، *بعد از* دریافت پاسخ HTTP این سرویس انجام می‌شد - یعنی
دیسپچر هم client (منتظر پاسخ) هم متریک‌گیر بود. حالا این پاد خودش، بلافاصله
بعد از اتمام پردازش، هم صف Redis را آزاد می‌کند و هم یک رکورد سبک متریک را
push می‌کند (edge:metrics:completions) - دیسپچر دیگر در مسیر critical این
درخواست حضور ندارد و فقط دوره‌ای این صف را می‌خواند (drain_completion_queue).
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
    async with _process_semaphore:
        start = time.monotonic()
        await asyncio.sleep(EXEC_TIME_SEC)
        elapsed = time.monotonic() - start

    # *** آزادسازی صف حالا اینجا انجام می‌شود (نه در دیسپچر مرکزی که دیگر
    # اصلاً از این تماس خبر ندارد).
    try:
        key = f"edge:replica:{SERVICE_ID}:{SERVER_ID}:queue"
        if int(_redis.get(key) or 0) > 0:
            _redis.decr(key)
    except Exception:
        pass  # نبود Redis نباید کل پردازش را بشکند؛ فقط صف کمی ناهماهنگ می‌ماند

    response_time_sec = elapsed
    if req.sent_at_epoch is not None:
        response_time_sec = time.time() - req.sent_at_epoch

    # *** گزارش سبک و async به Redis (نه HTTP به دیسپچر)
    try:
        _redis.rpush("edge:metrics:completions", __import__("json").dumps({
            "request_id": req.request_id, "service_id": SERVICE_ID, "server_id": SERVER_ID,
            "success": True, "response_time_sec": response_time_sec,
        }))
    except Exception:
        pass

    return {
        "request_id": req.request_id,
        "service_id": SERVICE_ID,
        "server_id": SERVER_ID,
        "exec_time_sec": elapsed,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    