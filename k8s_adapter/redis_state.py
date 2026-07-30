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
# ---------------------------------------------------------------------------

def try_reserve_queue_slot(service_id: int, server_id: int, queue_len: int) -> bool:
    """اتمیک: اگر صف پر نبود، ۱ واحد رزرو می‌کند و True برمی‌گرداند."""
    key = f"edge:replica:{service_id}:{server_id}:queue"
    new_val = _r.incr(key)
    if new_val > queue_len:
        _r.decr(key)
        return False
    return True


def release_queue_slot(service_id: int, server_id: int):
    key = f"edge:replica:{service_id}:{server_id}:queue"
    if int(_r.get(key) or 0) > 0:
        _r.decr(key)


def get_queue_occupancy(service_id: int, server_id: int) -> int:
    return int(_r.get(f"edge:replica:{service_id}:{server_id}:queue") or 0)


# ---------------------------------------------------------------------------
# متریک‌های زنده (برای مانیتورینگ حین اجرا؛ متریک نهایی هنوز از common/metrics.py می‌آید)
# ---------------------------------------------------------------------------

def incr_metric(name: str, amount: int = 1):
    _r.incrby(f"edge:metrics:{name}", amount)


def get_metric(name: str) -> int:
    return int(_r.get(f"edge:metrics:{name}") or 0)


def reset_all(n_servers: int, n_services: int):
    """پاک‌سازی کامل state قبل از شروع یک اجرای جدید."""
    keys = _r.keys("edge:*")
    if keys:
        _r.delete(*keys)
