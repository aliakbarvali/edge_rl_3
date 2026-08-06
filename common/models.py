"""
common/models.py
موجودیت‌های اصلی دامنه، طبق بخش ۲ سند معماری.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import Dict, Optional


class ServerState(Enum):
    OFF = auto()
    BOOTING = auto()
    ACTIVE = auto()
    DRAINING = auto()


class ReplicaState(Enum):
    STARTING = auto()
    READY = auto()
    DRAINING = auto()
    TERMINATED = auto()


class RequestStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()
    REJECTED_QUEUE_FULL = auto()
    REJECTED_NO_REPLICA = auto()


@dataclass
class Replica:
    """بخش ۲.۲: نمونه‌ی یک سرویس روی یک سرور."""
    service_id: int
    server_id: int
    queue_len: int
    exec_time: float
    state: ReplicaState = ReplicaState.STARTING
    created_at: float = 0.0
    ready_since: Optional[float] = None
    drain_started_at: Optional[float] = None

    # --- وضعیت صف FIFO واقعی (M/M/1-like؛ سرویس تک‌سرور، زمان سرویس ثابت) ---
    available_at: float = 0.0            # زمانی که رپلیکا از پردازش فعلی آزاد می‌شود
    departures: deque = field(default_factory=deque)  # زمان اتمام درخواست‌های در حال حاضر در سیستم (صف+سرویس)

    def is_selectable(self) -> bool:
        return self.state == ReplicaState.READY

    def queue_occupancy(self, now: float) -> int:
        """تعداد درخواست‌های فعلاً در سیستم (در صف + در حال پردازش) در لحظه‌ی now."""
        while self.departures and self.departures[0] <= now:
            self.departures.popleft()
        return len(self.departures)

    def try_admit(self, arrival_time: float, cold_start_extra: float = 0.0):
        """
        تلاش برای پذیرش یک درخواست در صف FIFO این رپلیکا.
        خروجی: None اگر صف پر بود (رد شود)، وگرنه dict شامل
        queue_enter_time, service_start_time, service_end_time, wait_time_sec.

        *** تصمیم طراحی صریح (بخش ۲.۵ سند): cold_start_extra مستقیماً به
        service_time اضافه می‌شود، یعنی self.available_at هم به همان اندازه
        عقب می‌افتد. این یعنی جریمه‌ی cold-start فقط روی response_time
        همین درخواست اثر نمی‌گذارد، بلکه wait_time تمام درخواست‌های بعدیِ
        همین replica در صف را هم افزایش می‌دهد (اثر زنجیره‌ای)، و چون
        instantaneous_utilization بر پایه‌ی is_idle/departures محاسبه
        می‌شود، این مدت اضافه در محاسبه‌ی انرژی (busy تا مدت طولانی‌تر) و
        reward PPO هم اثر غیرمستقیم دارد.
        این عمداً واقع‌گرایانه نگه داشته شده (یک replica در حال cold-start
        واقعاً کندتر است، نه فقط دیرتر پاسخ می‌دهد)؛ جداکردن این دو (فقط
        اثر روی response_time گزارش‌شده، بدون تأخیر واقعی در available_at)
        نیازمند حسابداری جدا برای departures/queue_occupancy می‌شد که
        ریسک ناهماهنگی جدید بین "اشغال صف" و "زمان واقعی پاسخ" داشت.
        """
        occ = self.queue_occupancy(arrival_time)
        if occ >= self.queue_len:
            return None
        start = max(arrival_time, self.available_at)
        service_time = self.exec_time + cold_start_extra
        finish = start + service_time
        self.available_at = finish
        self.departures.append(finish)
        return {
            "queue_enter_time": arrival_time,
            "service_start_time": start,
            "service_end_time": finish,
            "wait_time_sec": start - arrival_time,
        }

    def is_idle(self, now: float) -> bool:
        return self.queue_occupancy(now) == 0


@dataclass
class Server:
    """بخش ۲.۱: سرور فیزیکی (worker node)."""
    id: int
    profile: str
    lat: float
    long: float
    capacity: int
    p_idle: float
    p_max: float
    state: ServerState = ServerState.OFF
    hosted_replicas: Dict[int, Replica] = field(default_factory=dict)  # service_id -> Replica
    boot_started_at: Optional[float] = None
    drain_started_at: Optional[float] = None
    last_transition_time: float = -1e18  # برای cooldown/anti-flapping
    cumulative_energy_joule: float = 0.0
    cumulative_busy_cpu_seconds: float = 0.0  
    num_boots: int = 0
    num_shutdowns: int = 0

    def used_cpu(self) -> int:
        """مجموع cpu_demand همه‌ی رپلیکاهای مستقر (READY یا STARTING یا DRAINING - همه رزرو ظرفیت دارند)."""
        return sum(self._cpu_of(r) for r in self.hosted_replicas.values())

    def _cpu_of(self, replica: Replica) -> int:
        from common.config import CFG
        return CFG.services_info[replica.service_id]["cpu_demand"]

    def free_capacity(self) -> int:
        return self.capacity - self.used_cpu()

    def can_host(self, service_id: int, cpu_demand: int) -> bool:
        """قید سخت بخش ۱.۱: حداکثر ۱ رپلیکا از هر سرویس + مجموع cpu_demand <= capacity."""
        if service_id in self.hosted_replicas:
            return False
        return self.free_capacity() >= cpu_demand

    def in_cooldown(self, now: float, cooldown_sec: float) -> bool:
        return (now - self.last_transition_time) < cooldown_sec


    
    def instantaneous_utilization(self, now: float) -> float:
        busy_cpu = 0
        for r in self.hosted_replicas.values():
            if r.state in (ReplicaState.READY, ReplicaState.DRAINING) and not r.is_idle(now):
                busy_cpu += self._cpu_of(r)
        return busy_cpu / self.capacity if self.capacity > 0 else 0.0
    
    def instantaneous_power_w(self, now: float) -> float:
        if self.state == ServerState.OFF:
            return 0.0
        if self.state == ServerState.BOOTING:
            return self.p_idle
        # ACTIVE یا DRAINING: مدل خطی idle->max بر اساس utilization
        util = self.instantaneous_utilization(now)
        return self.p_idle + (self.p_max - self.p_idle) * util


@dataclass
class Request:
    """بخش ۲.۳."""
    id: int
    bts_lat: float
    bts_long: float
    service_id: int
    arrival_time: float
    assigned_server_id: Optional[int] = None
    queue_enter_time: Optional[float] = None
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    network_delay_ms: float = 0.0
    # *** بخش جدید (اصلاح معماری دیسپچر): تأخیر رفت‌وبرگشت مرحله‌ی مسیریابی
    # (BTS<->دیسپچر، قبل از این‌که سرور مقصد اصلاً معلوم شود) - جدا از
    # network_delay_ms که فقط تأخیر یک‌طرفه‌ی مرحله‌ی داده‌ی واقعی
    # (BTS<->سرور) است. response_time_sec هر دو را جمع می‌زند.
    routing_delay_sec: float = 0.0
    wait_time_sec: float = 0.0
    response_time_sec: float = 0.0
    deadline_violated: bool = False
    status: RequestStatus = RequestStatus.PENDING