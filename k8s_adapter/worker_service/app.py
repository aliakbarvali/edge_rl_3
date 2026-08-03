"""
k8s_adapter/worker_service/app.py

سرویس واقعی مینیمال که داخل هر پاد (طبق بخش ۱۲ سند - فاز ۳) اجرا می‌شود.
یک image واحد برای هر ۱۵ سرویس استفاده می‌شود؛ تفاوت هر سرویس فقط از طریق
متغیر محیطی EXEC_TIME_SEC در Deployment آن مشخص می‌شود (نگاه کنید به
k8s_client.py:build_deployment_manifest).

رفتار: هر درخواست POST به /process دقیقاً EXEC_TIME_SEC ثانیه طول می‌کشد
(sleep) - همان چیزی که در common/models.py:Replica.try_admit به‌صورت
تحلیلی شبیه‌سازی می‌شد، اینجا واقعاً رخ می‌دهد. چون uvicorn با
`--workers 1 --limit-concurrency 1` اجرا می‌شود (نگاه کنید Dockerfile)،
هر پاد واقعاً هم‌زمان فقط ۱ درخواست پردازش می‌کند - دقیقاً طبق بخش ۱.۲ سند
("هر رپلیکا هم‌زمان فقط ۱ درخواست را پردازش می‌کند").

صف واقعی (queue_len) توسط خودِ uvicorn/OS TCP backlog مدیریت نمی‌شود به
شکل دقیق دلخواه ما؛ به همین دلیل کنترل ظرفیت صف را realtime_dispatcher.py
از طریق Redis (قبل از ارسال درخواست) انجام می‌دهد، نه خودِ این سرویس.
"""

import os
import asyncio
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

EXEC_TIME_SEC = float(os.environ.get("EXEC_TIME_SEC", "1.0"))
SERVICE_ID = os.environ.get("SERVICE_ID", "unknown")
SERVER_ID = os.environ.get("SERVER_ID", "unknown")

# *** محدودیت «هم‌زمان فقط ۱ درخواست پردازش می‌شود» حالا اینجا و فقط روی
# /process اعمال می‌شود (نه در سطح کل uvicorn process با --limit-concurrency
# که قبلاً حتی /healthz را هم گرفتار می‌کرد و باعث می‌شد پاد هیچ‌وقت Ready
# نشود). درخواست‌های اضافه پشت این Semaphore صف می‌کشند (await)، رد نمی‌شوند.
_process_semaphore = asyncio.Semaphore(1)


class ProcessRequest(BaseModel):
    request_id: int | None = None


@app.get("/healthz")
def healthz():
    """برای readinessProbe/livenessProbe در Deployment استفاده می‌شود."""
    return {"status": "ok", "service_id": SERVICE_ID, "server_id": SERVER_ID}


@app.post("/process")
async def process(req: ProcessRequest):
    async with _process_semaphore:
        start = time.monotonic()
        # *** asyncio.sleep به‌جای time.sleep: کل event loop (و در نتیجه
        # /healthz) دیگر در طول EXEC_TIME_SEC فریز نمی‌شود.
        await asyncio.sleep(EXEC_TIME_SEC)
        elapsed = time.monotonic() - start
        return {
            "request_id": req.request_id,
            "service_id": SERVICE_ID,
            "server_id": SERVER_ID,
            "exec_time_sec": elapsed,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }