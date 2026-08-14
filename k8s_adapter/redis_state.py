"""
k8s_adapter/redis_state.py

هماهنگی/state مشترک روی Redis (طبق بخش ۱۰ و ۱۲ سند، روی ماشین ۱۹۲.۱۶۸.۱.۳۰).
چون در فاز ۳ دو فرآیند جدا داریم (۱: حلقه‌ی تصمیم‌گیری هر ۳۰ ثانیه که
scale/provision را اجرا می‌کند، ۲: dispatcher که هر درخواست واقعی را
مسیریابی می‌کند)، این دو باید یک دید مشترک و لحظه‌ای از وضعیت سرورها/رپلیکاها
داشته باشند - Redis این نقش را بازی می‌کند.

کلیدهای Redis:
    edge:server:{id}:state              -> "OFF"|"BOOTING"|"ACTIVE"|"DRAINING"
    edge:replica:{svc}:{srv}:state      -> "STARTING"|"READY"|"DRAINING"|"TERMINATED"
    edge:replica:{svc}:{srv}:pod_ip     -> آدرس IP پاد
    edge:replica:{svc}:{srv}:queue      -> شمارنده‌ی اشغال صف (INCR/DECR اتمیک)
    edge:service:{svc}:ready_replicas   -> SET از server_id های READY آن سرویس
    edge:metrics:*                      -> شمارنده‌های زنده (برای مانیتورینگ حین اجرا)
"""

from __future__ import annotations
from typing import Optional
import json
import time
try:
    import redis
except ImportError as e:
    raise ImportError("کتابخانه‌ی redis نصب نیست. اجرا کنید: pip install redis") from e

REDIS_HOST = "192.168.1.30"
REDIS_PORT = 6379
REDIS_DB = 0

_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def ping() -> bool:
    return _r.ping()


# ---------------------------------------------------------------------------
# سرور
# ---------------------------------------------------------------------------

def set_server_state(server_id: int, state: str):
    _r.set(f"edge:server:{server_id}:state", state)


def get_server_state(server_id: int) -> str:
    return _r.get(f"edge:server:{server_id}:state") or "OFF"


def all_server_states(n_servers: int) -> dict[int, str]:
    return {sid: get_server_state(sid) for sid in range(1, n_servers + 1)}


# ---------------------------------------------------------------------------
# رپلیکا
# ---------------------------------------------------------------------------

def set_replica_state(service_id: int, server_id: int, state: str):
    _r.set(f"edge:replica:{service_id}:{server_id}:state", state)
    if state == "READY":
        _r.sadd(f"edge:service:{service_id}:ready_replicas", server_id)
    else:
        _r.srem(f"edge:service:{service_id}:ready_replicas", server_id)


def get_replica_state(service_id: int, server_id: int) -> str:
    return _r.get(f"edge:replica:{service_id}:{server_id}:state") or "TERMINATED"


def set_pod_ip(service_id: int, server_id: int, ip: str):
    _r.set(f"edge:replica:{service_id}:{server_id}:pod_ip", ip)


def get_pod_ip(service_id: int, server_id: int) -> Optional[str]:
    return _r.get(f"edge:replica:{service_id}:{server_id}:pod_ip")


def remove_replica(service_id: int, server_id: int):
    _r.delete(f"edge:replica:{service_id}:{server_id}:state",
              f"edge:replica:{service_id}:{server_id}:pod_ip",
              f"edge:replica:{service_id}:{server_id}:queue")
    _r.srem(f"edge:service:{service_id}:ready_replicas", server_id)


def ready_replica_server_ids(service_id: int) -> list[int]:
    return [int(x) for x in _r.smembers(f"edge:service:{service_id}:ready_replicas")]


# ---------------------------------------------------------------------------
# صف (برای اعمال قید queue_len دستی - چون uvicorn/HTTP خودش این ظرفیت را
# دقیقاً به شکل دلخواه ما مدیریت نمی‌کند)
#
# *** رفع باگ (نشتی صف / queue-slot leak): نسخه‌ی قبلی فقط یک شمارنده‌ی
# INCR/DECR ساده داشت و یک تابع set_reservation_ttl جدا که یک کلید TTLدار
# می‌نوشت، ولی هیچ‌کس هرگز آن کلید را نمی‌خواند یا منقضی‌شدنش را رصد
# نمی‌کرد (کد مرده). یعنی اگر پاد worker بین رزرو صف (try_reserve_queue_slot،
# همین‌جا در دیسپچر) و پردازش واقعی (app.py) کرش می‌کرد یا در دسترس نبود،
# شمارنده‌ی صف *برای همیشه* یک واحد بالاتر از واقعیت می‌ماند - این نشتی
# به‌تدریج صف را «پر» نشان می‌دهد، درخواست‌های بعدی را رد می‌کند و تصمیمات
# scale/provision را (که دقیقاً روی همین avg_queue_occupancy لحظه‌ای حساب
# می‌شوند) گمراه می‌کند - یعنی برخلاف هدف صریح این نسخه (utilization واقعی و
# لحظه‌ای دقیق).
#
# راه‌حل: هر رزرو موفق هم در شمارنده‌ی سریع (INCR، برای مسیر hot-path
# ادمیشن) و هم در یک ZSET جداگانه‌ی «edge:reservations:{svc}:{srv}» با
# score = زمان انقضا ثبت می‌شود. پردازش موفق در worker (app.py) رزرو خودش
# را از این ZSET پاک می‌کند. یک تسک دوره‌ای در RealtimeEngine
# (sweep_expired_reservations) رزروهایی را که از انقضا گذشته‌اند ولی هنوز
# در ZSET مانده‌اند (یعنی پاد هرگز جواب نداد) پیدا و شمارنده‌ی صف را برایشان
# آزاد می‌کند.
# ---------------------------------------------------------------------------

def try_reserve_queue_slot(service_id: int, server_id: int, queue_len: int,
                            request_id: Optional[int] = None,
                            ttl_sec: Optional[float] = None) -> bool:
    """اتمیک: اگر صف پر نبود، ۱ واحد رزرو می‌کند و True برمی‌گرداند.
    اگر request_id/ttl_sec داده شود، رزرو برای پاک‌سازی خودکار (در صورت
    لو رفتن/بی‌پاسخی پاد) هم در ZSET انقضا ثبت می‌شود."""
    key = f"edge:replica:{service_id}:{server_id}:queue"
    new_val = _r.incr(key)
    if new_val > queue_len:
        _r.decr(key)
        return False
    if request_id is not None and ttl_sec is not None:
        _r.zadd(f"edge:reservations:{service_id}:{server_id}",
                {str(request_id): time.time() + ttl_sec})
    return True


def release_queue_slot(service_id: int, server_id: int, request_id: Optional[int] = None):
    key = f"edge:replica:{service_id}:{server_id}:queue"
    if int(_r.get(key) or 0) > 0:
        _r.decr(key)
    if request_id is not None:
        _r.zrem(f"edge:reservations:{service_id}:{server_id}", str(request_id))


def get_queue_occupancy(service_id: int, server_id: int) -> int:
    return int(_r.get(f"edge:replica:{service_id}:{server_id}:queue") or 0)


def sweep_expired_reservations(n_servers: int, n_services: int) -> int:
    """برای هر (سرویس، سرور)، رزروهایی که از زمان انقضایشان گذشته ولی هنوز
    در ZSET مانده‌اند را پیدا، از ZSET حذف و شمارنده‌ی صف را برایشان یک واحد
    آزاد می‌کند.

    *** پچ (رفع باگ ۲ - دبل-دیکریمنت): قبلاً این‌جا بدون قید‌وشرط بعد از
    zrem شمارنده‌ی صف را کم می‌کرد. اگر همین رزرو هم‌زمان توسط خودِ worker
    (app.py، بعد از اتمام واقعی پردازش) هم در حال آزادسازی بود، شمارنده دو
    بار کم می‌شد (یک‌بار اینجا، یک‌بار در worker) - چون ZREM اتمیک است، فقط
    طرفی که واقعاً عضو را حذف کرد (return=1) مجاز به decrement شمارنده است؛
    طرف دیگر (return=0، یعنی دیگری زودتر رسیده) هیچ کاری نمی‌کند. این دقیقاً
    هم‌راستا با منطق مشابهی است که در k8s_adapter/worker_service/app.py
    اضافه شده.

    خروجی: تعداد رزروهای منقضی‌شده‌ی *واقعاً* آزادشده توسط این تابع."""
    now = time.time()
    released = 0
    for svc_id in range(1, n_services + 1):
        for srv_id in range(1, n_servers + 1):
            zkey = f"edge:reservations:{svc_id}:{srv_id}"
            expired = _r.zrangebyscore(zkey, "-inf", now)
            if not expired:
                continue
            for member in expired:
                removed = _r.zrem(zkey, member)
                if removed:
                    release_queue_slot(svc_id, srv_id)
                    released += 1
    return released


# ---------------------------------------------------------------------------
# متریک‌های زنده (برای مانیتورینگ حین اجرا؛ متریک نهایی هنوز از common/metrics.py می‌آید)
# ---------------------------------------------------------------------------

def incr_metric(name: str, amount: int = 1):
    _r.incrby(f"edge:metrics:{name}", amount)


def get_metric(name: str) -> int:
    return int(_r.get(f"edge:metrics:{name}") or 0)


 
def reset_all(n_servers: int, n_services: int):
  
    all_keys = []
    for pattern in ("edge:*", "service:*"):
        keys = _r.keys(pattern)
        if keys:
            all_keys.extend(keys)
    if all_keys:
        _r.delete(*all_keys) 

def push_completion(service_id: int, server_id: int, request_id: int,
                     success: bool, response_time_sec: float): 
    payload = json.dumps({
        "request_id": request_id, "service_id": service_id, "server_id": server_id,
        "success": success, "response_time_sec": response_time_sec,
    })
    _r.rpush("edge:metrics:completions", payload)


def pop_completion_batch(max_items: int = 500) -> list[dict]:
    pipe = _r.pipeline()
    pipe.lrange("edge:metrics:completions", 0, max_items - 1)
    pipe.ltrim("edge:metrics:completions", max_items, -1)
    raw_items, _ = pipe.execute()
    return [json.loads(x) for x in raw_items]

def get_busy_seconds_acc(service_id: int, server_id: int) -> float:
    """انباشت دقیق busy-seconds از رویدادهای worker — برای محاسبه‌ی energy."""
    val = _r.get(f"service:{service_id}:server:{server_id}:busy_seconds_acc")
    return float(val) if val else 0.0

def reset_busy_seconds_acc(service_id: int, server_id: int) -> None:
    """بعد از هر بار خواندن و commit کردن، صفر می‌شود تا double-count نشود."""
    _r.delete(f"service:{service_id}:server:{server_id}:busy_seconds_acc")
    
    
def pop_busy_seconds_acc(service_id: int, server_id: int) -> float:
    """اتمیک: مقدار انباشته را می‌خواند و هم‌زمان صفر می‌کند (GETSET) —
    از race با INCRBYFLOAT هم‌زمان ورکر جلوگیری می‌کند."""
    val = _r.getset(f"service:{service_id}:server:{server_id}:busy_seconds_acc", 0.0)
    return float(val) if val else 0.0