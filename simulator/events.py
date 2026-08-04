"""
simulator/events.py
تعریف رویدادهای موتور discrete-event (بخش ۱۰ سند).
چون simpy در محیط اجرا قابل‌نصب نبود، یک موتور سبک مبتنی بر heapq پیاده‌سازی
شده (simulator/engine.py) که همان مدل رویداد-محور را با اولویت زمانی پیاده
می‌کند؛ اگر بعداً simpy در دسترس بود، فقط engine.py باید عوض شود، نه بقیه‌ی پروژه.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    REQUEST_ARRIVAL = auto()
    DECISION_TICK = auto()          # هر DECISION_INTERVAL_SEC: scale/provision/migration
    SERVER_BOOT_DONE = auto()       # BOOTING -> ACTIVE
    SERVER_DRAIN_DONE = auto()      # DRAINING -> OFF
    REPLICA_READY = auto()          # STARTING -> READY
    REPLICA_TERMINATED = auto()     # DRAINING -> TERMINATED
    ENERGY_RESYNC = auto()          # *** رویداد سبک، فقط برای دقیق‌کردن
                                     # محاسبه‌ی انرژی در لحظه‌ی اتمام دقیق هر
                                     # درخواست (بدون این، انرژی فقط در لحظه‌ی
                                     # رویداد بعدی -که تا ۳۰ ثانیه دیرتر باشد-
                                     # به‌روزرسانی می‌شد)


@dataclass(order=True)
class Event:
    time: float
    seq: int                      
    type: EventType = field(compare=False)
    payload: Any = field(default=None, compare=False)
