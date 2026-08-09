"""
common/logger.py

"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone


class EventLogger:

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
