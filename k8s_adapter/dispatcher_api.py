"""
k8s_adapter/dispatcher_api.py

لایه‌ی HTTP سبکِ control-plane. برخلاف نسخه‌ی قبلی که کل چرخه‌ی درخواست
(انتخاب رپلیکا + رزرو صف + HTTP واقعی به پاد + انتظار پاسخ) در یک تابع
دیسپچر مرکزی انجام می‌شد، اینجا دیسپچر *فقط* مسیریابی را انجام می‌دهد:
BTS اطلاعات سبک درخواست (service_id + مختصات) را می‌فرستد و فقط آدرس
مقصد را پس می‌گیرد. ترافیک سنگین (payload واقعی + پاسخ پردازش) هرگز از
این ماشین (۱۹۲.۱۶۸.۱.۳۰) رد نمی‌شود؛ مستقیماً بین BTS و پاد worker
رد و بدل می‌شود.

اجرا (روی همون ماشینی که RealtimeEngine اجرا می‌شود):
    uvicorn k8s_adapter.dispatcher_api:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from k8s_adapter import redis_state

app = FastAPI(title="edge-rl control-plane dispatcher")

# نمونه‌ی RealtimeEngine قبل از استارت این process باید ست شود
# (نگاه کنید realtime_dispatcher.py:serve_control_plane)
_engine = None


def bind_engine(engine):
    global _engine
    _engine = engine


class RouteRequest(BaseModel):
    request_id: int
    service_id: int
    bts_lat: float
    bts_long: float


class ReportRequest(BaseModel):
    """
    *** fire-and-forget: BTS بعد از دریافت پاسخ واقعی از پاد، این را
    می‌فرستد تا متریک نهایی (response_time واقعی که خودِ BTS اندازه‌گیری
    کرده) ثبت شود. این هم سبک است (چند عدد، نه payload اصلی) و async/
    بدون انتظار پاسخ فرستاده می‌شود - نباید bottleneck ایجاد کند.
    """
    request_id: int
    service_id: int
    server_id: int
    success: bool
    response_time_sec: float


@app.post("/route")
async def route(req: RouteRequest):
    """فقط تصمیم مسیریابی + رزرو اتمیک صف. هیچ HTTP دیگری اینجا زده نمی‌شود."""
    if _engine is None:
        return {"status": "ENGINE_NOT_READY"}
    return await _engine.route_request(req.request_id, req.service_id, req.bts_lat, req.bts_long)


@app.post("/report")
async def report(req: ReportRequest):
    """
    *** توجه: این endpoint هم سبک نگه داشته شده (فقط چند عدد)، ولی حتی
    این را هم می‌توان بعداً به‌جای HTTP، به یک Redis List/Stream
    (edge:metrics:completions) تبدیل کرد تا حتی این ترافیک هم از این
    ماشین با overhead کمتر عبور کند - نگاه کنید متد drain_completion_queue
    در RealtimeEngine برای همان الگو با پاد worker.
    """
    if _engine is None:
        return {"status": "ENGINE_NOT_READY"}
    _engine.record_external_completion(
        req.request_id, req.service_id, req.server_id, req.success, req.response_time_sec)
    return {"status": "OK"}