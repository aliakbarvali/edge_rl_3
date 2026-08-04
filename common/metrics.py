from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from common.models import Request, RequestStatus, ServerState


def _pct(values, p):
    return float(np.percentile(values, p)) if values else 0.0


@dataclass
class MetricsCollector:
    algorithm: str = ""
    response_times: List[float] = field(default_factory=list)
    network_delays: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    deadline_violations: int = 0
    total_requests: int = 0
    completed_requests: int = 0
    rejected_queue_full: int = 0
    rejected_no_replica: int = 0
    num_server_boots: int = 0
    num_server_shutdowns: int = 0
    num_pod_creates: int = 0
    num_pod_deletes: int = 0
    num_scale_up: int = 0
    num_scale_down: int = 0
    num_turn_on: int = 0
    num_turn_off: int = 0
    _snapshot_times: List[float] = field(default_factory=list)
    _active_server_counts: List[int] = field(default_factory=list)
    _load_balance_cvs: List[float] = field(default_factory=list)

    # *** بخش ۸: «هر الگوریتم گزارش بده ... چند تا از این تصمیم‌ها (با معیار:
    # آیا واقعاً لازم بود) درست بودند» - قبلاً فقط تعداد خام هر نوع اکشن
    # ثبت می‌شد، هیچ سنجه‌ای برای لزوم/عدم‌لزوم آن نبود.
    _decision_correctness: dict = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "incorrect": 0, "missed": 0}))

    def record_decision_correctness(self, kind: str, necessary: bool):
        """برای یک اکشن *واقعاً اعمال‌شده*: آیا طبق معیار ممیزی مستقل لازم بود؟"""
        self._decision_correctness[kind]["correct" if necessary else "incorrect"] += 1

    def record_missed_opportunity(self, kind: str):
        """طبق معیار ممیزی مستقل این اکشن لازم بود، ولی این تیک اعمال نشد
        (چه چون الگوریتم تصمیم دیگری گرفت، چه چون gate/cooldown آن را رد کرد)."""
        self._decision_correctness[kind]["missed"] += 1

    def record_request(self, req: Request):
        self.total_requests += 1
        if req.status == RequestStatus.COMPLETED:
            self.completed_requests += 1
            self.response_times.append(req.response_time_sec)
            # *** رفع باگ: distances/network_delays فقط برای درخواست‌های
            # واقعاً پذیرفته‌شده (COMPLETED) در محاسبه‌ی میانگین لحاظ می‌شوند.
            # دقت کن: این به این معنا نیست که این فیلدها همیشه صفر/محاسبه‌نشده
            # می‌مانند برای REJECTED_* - در مسیر رایج فعلی (select_replica
            # همه‌ی پیاده‌سازی‌های موجود فقط replica دارای جای خالی را
            # برمی‌گرداند) این فیلدها واقعاً هرگز محاسبه نمی‌شوند، ولی اگر یک
            # الگوریتم سفارشی آینده select_replica را طوری پیاده کند که
            # try_admit بعد از انتخاب هم بتواند None برگرداند (مثلاً
            # batch-selection با race واقعی)، این فیلدها *قبل* از آن شکست
            # قبلاً ست شده‌اند اما همچنان عمداً نادیده گرفته می‌شوند - چون
            # معیار شرکت در میانگین صرفاً status==COMPLETED است، نه اینکه
            # فیلد ست شده یا نه.
            self.distances.append(self._distance_of(req))
            self.network_delays.append(req.network_delay_ms)
            if req.deadline_violated:
                self.deadline_violations += 1
        elif req.status == RequestStatus.REJECTED_QUEUE_FULL:
            self.rejected_queue_full += 1
            self.deadline_violations += 1
        elif req.status == RequestStatus.REJECTED_NO_REPLICA:
            self.rejected_no_replica += 1
            self.deadline_violations += 1

    @staticmethod
    def _distance_of(req):
        return getattr(req, "_distance_km", 0.0)

    def record_snapshot(self, now, servers):
        active = [s for s in servers.values() if s.state == ServerState.ACTIVE]
        self._snapshot_times.append(now)
        self._active_server_counts.append(len(active))
        # *** رفع بایاس: قبلاً تیک‌هایی با فقط ۱ سرور فعال کاملاً از میانگین
        # حذف می‌شدند (نه این‌که CV=0 حساب شوند) - این باعث می‌شد الگوریتمی
        # که بیشتر وقتش تک‌سروره (مثل Greedy) به‌طور مصنوعی avg_load_balance_cv
        # بهتری نشان دهد، چون دقیقاً تیک‌های "بی‌معنی برای توازن" را از
        # میانگین‌گیری حذف می‌کرد، نه این‌که واقعاً بی‌طرف حسابشان کند.
        if len(active) == 1:
            self._load_balance_cvs.append(0.0)
        elif len(active) >= 2:
            loads = np.array([s.instantaneous_utilization(now) for s in active])
            if loads.mean() > 0:
                self._load_balance_cvs.append(float(loads.std() / loads.mean()))
            else:
                self._load_balance_cvs.append(0.0)

    def record_transition(self, kind):
        if kind == "server_boot":
            self.num_server_boots += 1
        elif kind == "server_shutdown":
            self.num_server_shutdowns += 1
        elif kind == "pod_create":
            self.num_pod_creates += 1
        elif kind == "pod_delete":
            self.num_pod_deletes += 1

    def record_scale_action(self, kind):
        if kind == "SCALE_UP":
            self.num_scale_up += 1
        elif kind == "SCALE_DOWN":
            self.num_scale_down += 1
        elif kind == "TURN_ON":
            self.num_turn_on += 1
        elif kind == "TURN_OFF":
            self.num_turn_off += 1

    def finalize(self, servers):
        cumulative_energy = sum(s.cumulative_energy_joule for s in servers.values())
        total = max(self.total_requests, 1)

        decision_correctness = {}
        for kind, counts in self._decision_correctness.items():
            applied_total = counts["correct"] + counts["incorrect"]
            decision_correctness[kind] = {
                "correct": counts["correct"],
                "incorrect": counts["incorrect"],
                "missed_opportunities": counts["missed"],
                "correctness_rate_pct": (100.0 * counts["correct"] / applied_total)
                                        if applied_total else None,
            }

        return {
            "algorithm": self.algorithm,
            "avg_response_time_sec": float(np.mean(self.response_times)) if self.response_times else 0.0,
            "p95_response_time_sec": _pct(self.response_times, 95),
            "p99_response_time_sec": _pct(self.response_times, 99),
            "deadline_violations": self.deadline_violations,
            "deadline_violation_rate_pct": 100.0 * self.deadline_violations / total,
            "cumulative_energy_joule": cumulative_energy,
            "avg_distance_km": float(np.mean(self.distances)) if self.distances else 0.0,
            "avg_load_balance_cv": float(np.mean(self._load_balance_cvs)) if self._load_balance_cvs else 0.0,
            "avg_network_delay_ms": float(np.mean(self.network_delays)) if self.network_delays else 0.0,
            "p95_network_delay_ms": _pct(self.network_delays, 95),
            "p99_network_delay_ms": _pct(self.network_delays, 99),
            "num_server_boots": self.num_server_boots,
            "num_server_shutdowns": self.num_server_shutdowns,
            "num_pod_creates": self.num_pod_creates,
            "num_pod_deletes": self.num_pod_deletes,
            "num_requests_rejected_queue_full": self.rejected_queue_full,
            "num_requests_rejected_no_replica": self.rejected_no_replica,
            "avg_active_servers": float(np.mean(self._active_server_counts)) if self._active_server_counts else 0.0,
            "num_scale_up": self.num_scale_up,
            "num_scale_down": self.num_scale_down,
            "num_turn_on": self.num_turn_on,
            "num_turn_off": self.num_turn_off,
            "decision_correctness": decision_correctness,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
        }