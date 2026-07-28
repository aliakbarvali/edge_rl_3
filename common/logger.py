"""
common/logger.py
لاگ‌گیری ساخت‌یافته (structured JSON) طبق بخش ۱۲ سند معماری.
در فاز ۱و۲ (شبیه‌سازی) این لاگ‌ها به فایل نوشته می‌شوند؛ در فاز ۳ (K8s واقعی)
همین اینترفیس بدون تغییر توسط k8s_adapter/ فراخوانی خواهد شد.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone


class EventLogger:
    """
    استفاده:
        logger = EventLogger("outputs/greedy_events.jsonl", algorithm="greedy")
        logger.log("request_completed", request_id=42, server_id=3, service_id=7,
                    response_time_sec=1.2)
    """

    def __init__(self, path: str, algorithm: str, enabled: bool = True):
        self.algorithm = algorithm
        self.enabled = enabled
        self.path = path
        self._fh = None
        if self.enabled:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def log(self, event_type: str, sim_time: float | None = None, **fields):
        if not self.enabled:
            return
        record = {
            "event_type": event_type,
            "algorithm": self.algorithm,
            "sim_time_sec": sim_time,
            "wall_time": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
