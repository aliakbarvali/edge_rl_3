# common/rolling_signal.py (فایل جدید)
"""
common/rolling_signal.py
سیگنال necessity لحظه‌ای (occ_ratio یک تیک) برای سرویس‌هایی با queue_len
کوچک (svc1, svc2) عملاً باینری و نویزیه. این ماژول یک پنجره‌ی رولینگ چند
تیکه از rejected/arrivals نگه می‌داره تا رد شدن‌های پراکنده روی چند تیک
دیده بشن، نه فقط توی همون یک تیک ۳۰ثانیه‌ای.
"""
from __future__ import annotations
from collections import defaultdict, deque

ROLLING_TICKS = 3  # ~90 ثانیه با DECISION_INTERVAL_SEC=30


class RollingRejectionTracker:
    def __init__(self, window_ticks: int = ROLLING_TICKS):
        self._rejected = defaultdict(lambda: deque(maxlen=window_ticks))
        self._arrivals = defaultdict(lambda: deque(maxlen=window_ticks))

    def push_tick(self, svc_id: int, rejected: int, arrivals: int):
        self._rejected[svc_id].append(rejected)
        self._arrivals[svc_id].append(arrivals)

    def rolling_rejection_rate(self, svc_id: int) -> float:
        rej = sum(self._rejected[svc_id])
        arr = sum(self._arrivals[svc_id])
        return (rej / arr) if arr else 0.0