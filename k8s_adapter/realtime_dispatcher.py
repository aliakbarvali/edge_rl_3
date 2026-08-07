"""
k8s_adapter/realtime_dispatcher.py

معادل واقعیِ simulator/engine.py برای فاز ۳: به‌جای شبیه‌سازی discrete-event
فشرده، واقعاً روی کلاستر Kubernetes عمل می‌کند و درخواست‌ها را با زمان‌بندی
واقعی (real-clock replay بر اساس فاصله‌ی بین‌رویدادی startSec) به سرویس‌های
واقعی مستقر روی worker nodeها هدایت می‌کند (بخش ۱۲ سند).

*** هشدار مهم: این فایل هرگز روی کلاستر واقعی تست نشده (من به شبکه‌ی
۱۹۲.۱۶۸.۱.x دسترسی ندارم). قبل از اجرای کامل روی داده‌ی واقعی، حتماً
`python3 -m k8s_adapter.smoke_test` را اجرا کنید (پایین همین پوشه) تا
اتصال Redis/K8s و ساخت/حذف یک Deployment آزمایشی تأیید شود.

معماری (اصلاح‌شده - جدایی control-plane/data-plane): این ماشین (۱۹۲.۱۶۸.۱.۳۰)
*فقط* دیسپچر (control-plane) است و هرگز میزبان ترافیک سنگین (payload واقعی
پردازش) نمی‌شود؛ آن مستقیماً بین BTS و پاد worker رد و بدل می‌شود. سه تسک
asyncio هم‌زمان اجرا می‌شوند:
    ۱) decision_loop: هر DECISION_INTERVAL_SEC ثانیه‌ی *واقعی*، دقیقاً همان
       چهار متد AlgorithmBase (scale_decision/provision_decision/
       select_placement_server/migration_decision) را که Greedy/Voila/HPA/
       PPO پیاده کرده‌اند صدا می‌زند - منطق تصمیم‌گیری هیچ تغییری نمی‌کند،
       فقط اجرای آن (k8s_client به‌جای دستکاری آبجکت در حافظه) عوض می‌شود.
    ۲) route_request (سرو شده از طریق dispatcher_api.py:/route، فراخوانده‌شده
       توسط BTS واقعی/bts_simulator.py): تماس *سبک* هر BTS برای گرفتن سرور
       مقصد پیش از ارسال درخواست واقعی - فقط lat/long+service_id ورودی و
       ip/port خروجی؛ هیچ payload سنگینی از این ماشین رد نمی‌شود. بعد از
       این پاسخ، BTS *خودش* مستقیماً به ip:port واقعی پاد وصل می‌شود.
    ۳) drain_completion_queue: به‌جای این‌که دیسپچر منتظر پاسخ هر درخواست
       بماند، هر چند صدم ثانیه صف کامل‌شده‌های Redis را (که خودِ پاد worker
       بعد از هر پردازش آنجا push کرده) می‌خواند و متریک نهایی را ثبت می‌کند.

یک نمای «سایه» (self.servers/self.replicas_by_service، همان دیتاکلاس‌های
common/models.py) در حافظه نگه‌داشته می‌شود تا بتوان از همان
AlgorithmBase.select_replica/initial_placement مشترک استفاده کرد بدون
بازنویسی منطق - این نما با هر عملیات واقعی (create/delete/ready) و با
Redis همگام می‌ماند.
"""

from __future__ import annotations
import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import json


from common.config import CFG
from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, ServerState, ReplicaState, Request, RequestStatus
from common.metrics import MetricsCollector
from common.logger import EventLogger
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from k8s_adapter import k8s_client, redis_state
 
UTIL_SAMPLE_INTERVAL_SEC = 1.0  # *** گرانولاریتی نمونه‌برداری busy/idle در فاز
                                 # ۳؛ چون رویداد گسسته‌ی دقیق مثل موتور
                                 # شبیه‌سازی نداریم، این تنها راه تقریب
                                 # انتگرال busy_cpu(t) در طول زمان واقعی است.
                                 # هرچه کوچک‌تر، دقیق‌تر ولی هزینه‌ی Redis
                                 # بیشتر - ۱ ثانیه توازن معقولی است چون کوتاه‌ترین
                                 # exec_time سیستم ۴ ثانیه است (سرویس ۱).
                                 
                                 
class RealtimeEngine:
    def __init__(self, events_df: pd.DataFrame, algorithm: AlgorithmBase,
                 algorithm_name: str, event_logger: Optional[EventLogger] = None):
        self.events_df = events_df
        self.algorithm = algorithm
        self.metrics = MetricsCollector(algorithm=algorithm_name)
        self.logger = event_logger

        self.servers: Dict[int, Server] = self._init_shadow_servers()
        self.replicas_by_service: Dict[int, List[Replica]] = defaultdict(list)
        self._request_seq = 0
        self._running = True

        # آمار «از آخرین تیک تصمیم» - دقیقاً مثل simulator/engine.py
        self._tick_total = defaultdict(int)
        self._tick_rejected = defaultdict(int)
        self._tick_violated = defaultdict(int)  
        # *** رفع کمبود: نسخه‌ی شبیه‌ساز (simulator/engine.py) این شمارنده
        # را داشت ولی اینجا فراموش شده بود؛ باعث می‌شد proximity_violation_rate
        # همیشه صفر/غایب باشد و common/state_builder.py نتواند از همان
        # snapshot این‌جا برای PPO استفاده کند بدون KeyError.
        self._tick_proximity_violated = defaultdict(int)

        # *** رفع کمبود دوم (بازبینی): demand_centroid اینجا همیشه None
        # هاردکد بود (نگاه کنید پایین همین فایل، _build_metrics_snapshot
        # قبلی). یعنی در فاز ۳ واقعی، VoilaAlgorithm.select_placement_server/
        # migration_decision/select_scale_down_victim همیشه به fallback
        # غیر-centroid-محور می‌افتادند - حتی بعد از این‌که تزریق occupancy_fn
        # به select_scale_down_victim دیورجنس sim/real را برای انتخاب قربانی
        # رفع کرد، خودِ centroid که آن انتخاب بر مبنایش انجام می‌شود هنوز
        # وجود نداشت. حالا دقیقاً مثل simulator/engine.py: موقعیت هر درخواست
        # در لحظه‌ی ورود (چه موفق چه رد شده) در یک پنجره‌ی غلتان ۳۰تایی ثبت
        # می‌شود و medoid آن به‌عنوان demand_centroid هر تیک محاسبه می‌شود.
        # عمداً هر تیک پاک نمی‌شود (drop-off طبیعی با maxlen=30 اتفاق می‌افتد)،
        # چون این یک پنجره‌ی غلتان چند-تیکی است نه شمارنده‌ی «از تیک قبل».
        self._recent_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self._service_demand_centroid: Dict[int, Optional[Tuple[float, float]]] = {}
        
        
        # *** برای میانگین‌گیری دقیق utilization روی هر پنجره‌ی تصمیم (مثل
        # _util_at_window_start در simulator/engine.py، ولی اینجا wall-clock).
        self._util_at_window_start: Dict[int, float] = defaultdict(float)
        self._util_window_start_time: float = time.monotonic()

        # *** رفع باگ (هماهنگی sim/real): simulator/engine.py هر SCALE_UP/DOWN
        # را با CFG.cooldown_sec در سطح هر سرویس گیت می‌کند (بخش ۷ سند:
        # «Cooldown مشابه ۶.۱ برای هر service_id اعمال می‌شود تا از flapping
        # جلوگیری شود» - نگاه کنید simulator/engine.py:_service_last_scale_time).
        # قبلاً این‌جا (فاز ۳ واقعی) چنین ردیابی‌ای اصلاً وجود نداشت؛
        # decision_loop هر تیک بدون قید دوباره scale_decision را اعمال می‌کرد.
        self._service_last_scale_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}

        # *** رفع باگ (بازبینی: گیت‌های ضدـflapping غایب در فاز ۳): شبیه‌ساز
        # (simulator/engine.py) قبل از اجازه‌دادن به TURN_ON/TURN_OFF واقعی،
        # علاوه بر cooldown، الزام «تداوم» overload/underload
        # (SUSTAIN_HIGH_SEC/SUSTAIN_LOW_SEC) و حداقل مدت ACTIVE-بودن
        # (MIN_ACTIVE_DURATION_SEC) را هم چک می‌کند - دقیقاً به‌خاطر یافته‌ی
        # analyze_decision_quality.py که ۹۱-۹۶٪ چرخه‌های on/off زیر ۵ دقیقه
        # dwell داشتند. اینجا (فاز ۳ واقعی) قبلاً این گیت‌ها اصلاً وجود
        # نداشتند - فقط cooldown_sec (۶۰ ثانیه) چک می‌شد - یعنی رفتار
        # production می‌توانست به‌طرز قابل‌توجهی flappy‌تر از چیزی باشد که
        # در شبیه‌سازی validate شده. حالا همان ردیابی اینجا هم پیاده می‌شود.
        self._low_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}
        self._high_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}


    # ------------------------------------------------------------------
    def _init_shadow_servers(self) -> Dict[int, Server]:
        servers = {}
        for sid, info in CFG.server_info.items():
            prof = CFG.server_profiles[info["profile"]]
            servers[sid] = Server(id=sid, profile=info["profile"], lat=info["lat"], long=info["long"],
                                   capacity=info["capacity"], p_idle=prof["p_idle"], p_max=prof["p_max"])
        return servers

    def _log(self, event_type: str, **fields):
        if self.logger:
            self.logger.log(event_type, sim_time=time.time(), **fields)

    # ------------------------------------------------------------------
    # بخش ۴: جایگذاری اولیه‌ی واقعی
    # ------------------------------------------------------------------
    async def initial_placement(self):
        first_window = self.events_df[self.events_df.global_start_sec <=
                                       self.events_df.global_start_sec.min() + CFG.monitor_window_sec]
        active_bts = list(first_window[["Lat", "Long"]].drop_duplicates().itertuples(index=False, name=None))
        selected = self.algorithm.initial_placement(self.servers, active_bts)

        for sid in selected:
            await self._activate_server(sid)

        for service_id in CFG.active_services:
            target = self._nearest_capable_server(service_id, selected)
            if target is not None:
                await self._create_replica(target, service_id)

        # صبر برای READY شدن رپلیکاهای اولیه قبل از شروع dispatch واقعی
        await self._wait_all_ready(timeout=CFG.pod_startup_delay_sec + 30)

    def _nearest_capable_server(self, service_id: int, candidate_ids: List[int]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["cpu_demand"]
        candidates = [self.servers[sid] for sid in candidate_ids
                      if self.servers[sid].can_host(service_id, cpu)]
        if not candidates:
            return None
        centroid_lat = sum(s.lat for s in candidates) / len(candidates)
        centroid_lon = sum(s.long for s in candidates) / len(candidates)
        candidates.sort(key=lambda s: haversine_km(centroid_lat, centroid_lon, s.lat, s.long))
        return candidates[0].id

    async def _wait_all_ready(self, timeout: float):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            pending = [r for reps in self.replicas_by_service.values() for r in reps
                       if r.state == ReplicaState.STARTING]
            if not pending:
                return
            await asyncio.sleep(1)
        self._log("initial_placement_ready_timeout")

    # ------------------------------------------------------------------
    # گذارهای سرور (cordon/uncordon واقعی - طبق پاسخ شما، بدون خاموشی فیزیکی)
    # ------------------------------------------------------------------
    async def _activate_server(self, server_id: int):
        s = self.servers[server_id]
        if s.state == ServerState.ACTIVE:
            return
        k8s_client.uncordon_node(server_id)
        redis_state.set_server_state(server_id, "ACTIVE")
        s.state = ServerState.ACTIVE
        s.last_transition_time = time.monotonic()
        s.num_boots += 1
        self.metrics.record_transition("server_boot")
        self._log("server_boot_started", server_id=server_id)
        self._log("server_active", server_id=server_id)

    async def _wait_specific_ready(self, replicas: Dict[int, "Replica"], timeout: float) -> Dict[int, bool]:
        """صبر می‌کند تا رپلیکاهای مشخص‌شده (service_id -> Replica، معمولاً
        مقصدهای یک migration) واقعاً READY شوند - برای Make-Before-Break
        واقعی در _drain_server. خروجی: service_id -> True/False (رسیده به
        READY تا قبل از timeout یا نه)."""
        start = time.monotonic()
        pending = set(replicas.keys())
        result = {svc_id: False for svc_id in replicas}
        while pending and (time.monotonic() - start) < timeout:
            for svc_id in list(pending):
                if replicas[svc_id].state == ReplicaState.READY:
                    result[svc_id] = True
                    pending.discard(svc_id)
            if pending:
                await asyncio.sleep(1.0)
        return result

    async def _drain_server(self, server_id: int) -> bool:
        s = self.servers[server_id]
        if s.state != ServerState.ACTIVE:
            return False

        steps = self.algorithm.migration_decision(s, self.servers)
        migrated_services = {step.service_id for step in steps}
        sole_hosted = {
            svc_id for svc_id, r in s.hosted_replicas.items()
            if r.state != ReplicaState.TERMINATED and not any(
                other.id != server_id and svc_id in other.hosted_replicas and
                other.hosted_replicas[svc_id].state != ReplicaState.TERMINATED
                for other in self.servers.values())
        }
        if not sole_hosted.issubset(migrated_services):
            self._log("server_drain_aborted", server_id=server_id, reason="migration_incomplete")
            return False

        s.state = ServerState.DRAINING
        self._log("server_drain_started", server_id=server_id)

        # *** رفع باگ مسدودکننده (نقض Make-Before-Break در فاز ۳ واقعی):
        # قبلاً بلافاصله بعد از create_deployment (بدون صبر برای READY
        # شدن واقعی) رپلیکای قدیم حذف می‌شد - دقیقاً همان مشکلی که
        # simulator/engine.py با دقت (make-before-break) حلش کرده بود.
        # نتیجه: یک پنجره‌ی واقعی (به‌اندازه‌ی زمان schedule/pull/boot پاد
        # جدید) که سرویس هیچ رپلیکای READY نداشت، و اگر create_deployment
        # به‌خاطر کمبود ظرفیت None برمی‌گرداند، رپلیکای قدیم هم بی‌صدا حذف
        # می‌شد -> آن سرویس کاملاً بدون رپلیکا می‌ماند (قطع کامل، بی‌صدا).
        # حالا: اول رپلیکای جدید هر migration ساخته می‌شود، بعد صبر واقعی
        # برای READY شدنش، و فقط بعد از موفقیت رپلیکای قدیم حذف می‌شود؛
        # هر migration که شکست بخورد (چه در create چه در READY شدن) از
        # مجموعه‌ی migrated خارج می‌شود تا رپلیکای قدیمش دست‌نخورده بماند.
        new_replicas: Dict[int, Replica] = {}
        failed_services: set = set()
        for step in steps:
            self._log("migration_started", service_id=step.service_id,
                      from_server_id=server_id, to_server_id=step.target_server_id)
            new_r = await self._create_replica(step.target_server_id, step.service_id)
            if new_r is None:
                self._log("migration_placement_failed", service_id=step.service_id,
                          from_server_id=server_id, to_server_id=step.target_server_id)
                failed_services.add(step.service_id)
                continue
            new_replicas[step.service_id] = new_r

        if new_replicas:
            ready_ok = await self._wait_specific_ready(
                new_replicas, timeout=CFG.pod_startup_delay_sec + 60.0)
            for svc_id, ok in ready_ok.items():
                if ok:
                    self._log("migration_completed", service_id=svc_id,
                              from_server_id=server_id,
                              to_server_id=new_replicas[svc_id].server_id)
                else:
                    failed_services.add(svc_id)
                    self._log("migration_ready_timeout", service_id=svc_id,
                              from_server_id=server_id,
                              to_server_id=new_replicas[svc_id].server_id)

        for service_id in list(s.hosted_replicas.keys()):
            if service_id in failed_services:
                # *** رپلیکای جدید مقصد ساخته/READY نشد؛ رپلیکای قدیم روی
                # این سرور دست‌نخورده می‌ماند تا سرویس بدون هیچ رپلیکایی
                # نماند - این سرویس چرخه‌ی بعد دوباره کاندید migration
                # می‌شود.
                continue
            await self._delete_replica(service_id, server_id)

        if failed_services:
            # *** حداقل یک migration واقعاً کامل نشد -> نمی‌توان این سرور
            # را drain کرد (سرویس(های) ناقص هنوز رویش زنده‌اند). به‌جای
            # نیمه‌کاره OFF کردنش، به ACTIVE برمی‌گردد تا چرخه‌ی بعد دوباره
            # امتحان شود - هم‌راستا با server_drain_aborted در
            # simulator/engine.py.
            s.state = ServerState.ACTIVE
            self._log("server_drain_aborted", server_id=server_id,
                      reason="migration_ready_failed", failed_services=list(failed_services))
            return False

        k8s_client.cordon_node(server_id)
        redis_state.set_server_state(server_id, "OFF")
        s.state = ServerState.OFF
        s.last_transition_time = time.monotonic()
        s.num_shutdowns += 1
        self.metrics.record_transition("server_shutdown")
        self._log("server_off", server_id=server_id)
        return True

    # ------------------------------------------------------------------
    # گذارهای رپلیکا (create/delete واقعی روی K8s)
    # ------------------------------------------------------------------
    async def _create_replica(self, server_id: int, service_id: int) -> Optional[Replica]:
        s = self.servers[server_id]
        cpu = CFG.services_info[service_id]["cpu_demand"]
        if not s.can_host(service_id, cpu):
            return None
        svc = CFG.services_info[service_id]

        k8s_client.create_deployment(service_id, server_id)
        redis_state.set_replica_state(service_id, server_id, "STARTING")
        self.metrics.record_transition("pod_create")
        self._log("pod_create_started", server_id=server_id, service_id=service_id)

        r = Replica(service_id=service_id, server_id=server_id,
                    queue_len=svc["queue_len"], exec_time=svc["exec_time"],
                    created_at=time.monotonic())
        s.hosted_replicas[service_id] = r
        self.replicas_by_service[service_id].append(r)

        asyncio.create_task(self._poll_until_ready(service_id, server_id, r))
        return r

    async def _poll_until_ready(self, service_id: int, server_id: int, replica: Replica,
                                 timeout: float = 120.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                if k8s_client.is_deployment_ready(service_id, server_id):
                    ip = k8s_client.get_pod_ip(service_id, server_id)
                    if ip:
                        redis_state.set_pod_ip(service_id, server_id, ip)
                        redis_state.set_replica_state(service_id, server_id, "READY")
                        replica.state = ReplicaState.READY
                        replica.ready_since = time.monotonic()
                        self._log("pod_ready", server_id=server_id, service_id=service_id, pod_ip=ip)
                        return
            except Exception as e:
                # *** قبلاً یک استثنای پیش‌بینی‌نشده اینجا کل task را بی‌صدا
                # می‌کشت (asyncio.create_task بدون exception handler) تا
                # هشدار "Task exception was never retrieved" دقیقه‌ها/ساعت‌ها
                # بعد، بدون هیچ اطلاعاتی از این‌که کدام سرویس/سرور بوده، چاپ
                # شود. حالا فوراً با شناسه‌ی کامل لاگ می‌شود و polling ادامه
                # پیدا می‌کند تا timeout.
                self._log("pod_ready_poll_error", server_id=server_id, service_id=service_id,
                          error=str(e))
            await asyncio.sleep(1.0)
        self._log("pod_ready_timeout", server_id=server_id, service_id=service_id)

    async def _delete_replica(self, service_id: int, server_id: int):
        s = self.servers[server_id]
        r = s.hosted_replicas.get(service_id)
        if r is None:
            return
        r.state = ReplicaState.DRAINING
        redis_state.set_replica_state(service_id, server_id, "DRAINING")
        self._log("pod_drain_started", server_id=server_id, service_id=service_id)

        # *** رفع ریسک از‌دست‌رفتن درخواست‌های درون‌پرواز: به‌جای sleep ثابت
        # (که برای سرویس‌های کند مثل سرویس ۱۵ با exec_time=120 کاملاً
        # ناکافی بود)، منتظر خالی‌شدن واقعی صف (از طریق Redis) با یک سقف
        # زمانی ایمن بر پایه‌ی worst-case این سرویس می‌مانیم.
        svc = CFG.services_info[service_id]
        max_wait = svc["queue_len"] * svc["exec_time"] + CFG.graceful_termination_delay_sec
        start = time.monotonic()
        while time.monotonic() - start < max_wait:
            if redis_state.get_queue_occupancy(service_id, server_id) <= 0:
                break
            await asyncio.sleep(1.0)
        else:
            self._log("pod_drain_timeout_forced", server_id=server_id, service_id=service_id,
                      reason="max_wait_exceeded")

        k8s_client.delete_deployment(service_id, server_id)
        redis_state.remove_replica(service_id, server_id)
        r.state = ReplicaState.TERMINATED
        self.metrics.record_transition("pod_delete")
        self._log("pod_terminated", server_id=server_id, service_id=service_id)

        del s.hosted_replicas[service_id]
        self.replicas_by_service[service_id] = [x for x in self.replicas_by_service[service_id]
                                                  if x is not r]

    # ------------------------------------------------------------------
    # بخش ۶ و ۷: حلقه‌ی تصمیم‌گیری واقعی (هر DECISION_INTERVAL_SEC ثانیه‌ی واقعی)
    # ------------------------------------------------------------------
    @staticmethod
    def _medoid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """دقیقاً همان تابع simulator/engine.py:_build_metrics_snapshot._medoid -
        نقطه‌ای از خودِ points که مجموع فاصله‌اش تا بقیه کمینه است (به‌جای
        میانگین متحرک EMA که می‌توانست به نقطه‌ای خارج از توزیع واقعی درخواست‌ها
        بیفتد)."""
        if not points:
            return None
        best, best_cost = points[0], float("inf")
        for p in points:
            cost = sum(haversine_km(p[0], p[1], q[0], q[1]) for q in points)
            if cost < best_cost:
                best, best_cost = p, cost
        return best

    def _build_metrics_snapshot(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        now = time.monotonic()
        window_elapsed = now - self._util_window_start_time

        for sid, s in self.servers.items():
            if window_elapsed > 1e-9 and s.capacity > 0:
                # *** میانگین دقیق زمانی از تیک قبل تا الان (تفاضل انباشته‌ها)،
                # نه یک نمونه‌ی لحظه‌ای - دقیقاً همون اصلاحی که در
                # simulator/engine.py انجام شد، اینجا هم با همون منطق.
                avg_util = ((s.cumulative_busy_cpu_seconds - self._util_at_window_start[sid])
                            / (s.capacity * window_elapsed))
            else:
                avg_util = 0.0
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": avg_util,
                "free_capacity": s.free_capacity(),
            } 
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            # *** رفع باگ مسدودکننده: PPOAlgorithm._build_action_masks از
            # sv.get("n_mature_ready_replicas", 0) استفاده می‌کند تا
            # can_down را بسازد. این کلید قبلاً اینجا اصلاً در snapshot
            # نبود؛ یعنی fallback همیشه ۰ برمی‌گشت -> can_down همیشه False
            # -> در فاز ۳ واقعی PPO هرگز نمی‌توانست SCALE_DOWN بزند (نه
            # «رفتار قدیمی بدون محدودیت» که کامنت قبلی ادعا می‌کرد، بلکه
            # دقیقاً برعکس: همیشه مسدود). حالا مثل simulator/engine.py
            # واقعاً محاسبه می‌شود.
            mature_ready = [r for r in ready
                             if (time.monotonic() - r.created_at) >=
                             CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(redis_state.get_queue_occupancy(svc_id, r.server_id) for r in ready)
                       / len(ready)) if ready else 0.0

            # *** رفع کمبود دوم (نگاه کنید __init__): محاسبه‌ی medoid روی
            # پنجره‌ی غلتان موقعیت‌های اخیر این سرویس - فقط وقتی این تیک
            # حداقل یک ورود واقعی داشته (هماهنگ با simulator/engine.py که
            # هم شرط دارد، تا در تیک‌های کاملاً بی‌ورودی مقدار قبلی حفظ شود
            # نه این‌که با یک تهیِ گمراه‌کننده جایگزین شود).
            if self._tick_total[svc_id] > 0:
                self._service_demand_centroid[svc_id] = self._medoid(
                    list(self._recent_positions[svc_id]))

            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "n_ready_replicas": len(ready),
                "n_mature_ready_replicas": len(mature_ready),
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": self._service_demand_centroid.get(svc_id),
                "proximity_violation_rate": self._tick_proximity_violated[svc_id] / total,
            }
        snapshot["global"] = {"avg_response_time_recent": 0.0, "energy_recent_joule": 0.0,
                               "num_rejected_recent": sum(self._tick_rejected.values())} 
        return snapshot

    def _update_sustain_tracking(self, snapshot: dict, now: float):
        """معادل simulator/engine.py:_update_sustain_tracking - ردیابی مدت
        زمانی که هر سرور ACTIVE به‌طور *مداوم* بالای/پایین آستانه بوده،
        نه فقط نمونه‌ی لحظه‌ای همین تیک."""
        for sid, s in self.servers.items():
            if s.state != ServerState.ACTIVE:
                self._low_util_since[sid] = None
                self._high_util_since[sid] = None
                continue
            util = snapshot["servers"][sid]["utilization"]
            if util < CFG.util_scale_down_threshold:
                if self._low_util_since[sid] is None:
                    self._low_util_since[sid] = now
            else:
                self._low_util_since[sid] = None

            if util > CFG.util_scale_up_threshold:
                if self._high_util_since[sid] is None:
                    self._high_util_since[sid] = now
            else:
                self._high_util_since[sid] = None

    def _any_active_server_sustained_overloaded(self, now: float) -> bool:
        for sid, since in self._high_util_since.items():
            if since is not None and (now - since) >= CFG.sustain_high_sec:
                return True
        return False

    def _any_active_server_sustained_underloaded(self, now: float) -> bool:
        n_active = sum(1 for s in self.servers.values() if s.state == ServerState.ACTIVE)
        if n_active <= 1:
            return False
        for sid, since in self._low_util_since.items():
            if since is not None and (now - since) >= CFG.sustain_low_sec:
                return True
        return False

    def _was_turn_off_necessary(self, server_id: int, now: float) -> bool:
        since = self._low_util_since.get(server_id)
        return since is not None and (now - since) >= CFG.sustain_low_sec

    def _any_service_capacity_starved(self, snapshot: dict) -> bool:
        """معادل simulator/engine.py:_any_service_capacity_starved - آستانه‌ی
        مستقل (DECISION_AUDIT_SCALE_UP_OCC_THRESHOLD) صرف‌نظر از threshold
        داخلی هر الگوریتم."""
        for svc_id in CFG.active_services:
            sv = snapshot["services"][svc_id]
            occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
            necessary = (occ_ratio > CFG.decision_audit_scale_up_occ_threshold
                         or sv["rejection_rate"] > 0.0)
            if not necessary:
                continue
            cpu = CFG.services_info[svc_id]["cpu_demand"]
            if not any(s.state == ServerState.ACTIVE and s.can_host(svc_id, cpu)
                       for s in self.servers.values()):
                return True
        return False

    async def decision_loop(self):
        while self._running:
            await asyncio.sleep(CFG.decision_interval_sec)
            snapshot = self._build_metrics_snapshot()
            now = time.monotonic()
            # *** رفع باگ (بازبینی): بدون این خط، _high_util_since/
            # _low_util_since هرگز پر نمی‌شدند و گیت‌های زیر همیشه False
            # می‌ماندند (معادل نبودشان) - این call باید هر تیک قبل از هر
            # تصمیم‌گیری صدا زده شود.
            self._update_sustain_tracking(snapshot, now)

            action = self.algorithm.provision_decision(self.servers, snapshot, now)
            if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
                s = self.servers[action.server_id]
                # *** رفع باگ (گیت ضدـflapping غایب): علاوه بر cooldown،
                # حالا نیاز به تداوم واقعی overload/starvation هم چک می‌شود
                # - هم‌راستا با simulator/engine.py:_apply_provisioning.
                turn_on_necessary = (self._any_active_server_sustained_overloaded(now)
                                      or self._any_service_capacity_starved(snapshot))
                if (s.state == ServerState.OFF and not s.in_cooldown(now, CFG.cooldown_sec)
                        and turn_on_necessary):
                    await self._activate_server(action.server_id)
                    self.metrics.record_scale_action("TURN_ON")
            elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
                s = self.servers[action.server_id]
                n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
                # *** رفع باگ (گیت‌های غایب): تداوم underload واقعی +
                # حداقل مدت ACTIVE بودن (MIN_ACTIVE_DURATION_SEC) - قبلاً
                # هیچ‌کدام اینجا چک نمی‌شدند، فقط cooldown_sec (۶۰ ثانیه).
                turn_off_necessary = self._was_turn_off_necessary(action.server_id, now)
                min_age_ok = (now - s.last_transition_time) >= CFG.min_active_duration_sec
                if (s.state == ServerState.ACTIVE and n_active > 1
                        and not s.in_cooldown(now, CFG.cooldown_sec)
                        and turn_off_necessary and min_age_ok):
                    if await self._drain_server(action.server_id):
                        self.metrics.record_scale_action("TURN_OFF")

            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)

                # *** رفع باگ (هماهنگی sim/real): همان گیت cooldown که
                # simulator/engine.py:_apply_scale_decision برای هر سرویس
                # اعمال می‌کند، اینجا هم اعمال می‌شود - بدون این، یک سرویس
                # مرزی می‌توانست هر تیک (۳۰ ثانیه) دوباره پاد بسازد/حذف کند.
                if decision == ScaleAction.NO_CHANGE:
                    continue
                if (now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec:
                    continue

                if decision == ScaleAction.SCALE_UP:
                    target = self.algorithm.select_placement_server(svc_id, self.servers)
                    if target is not None:
                        await self._create_replica(target, svc_id)
                        self.metrics.record_scale_action("SCALE_UP")
                        self._service_last_scale_time[svc_id] = now
                elif decision == ScaleAction.SCALE_DOWN:
                    ready = [r for r in self.replicas_by_service.get(svc_id, [])
                             if r.state == ReplicaState.READY]
                    # *** رفع باگ (هماهنگی sim/real): simulator/engine.py قبل
                    # از انتخاب قربانی، رپلیکاهای جوان‌تر از
                    # CFG.min_replica_age_before_scale_down_sec را کنار
                    # می‌گذارد (تا یک SCALE_UP تازه بلافاصله توسط SCALE_DOWN
                    # همان تیک/تیک بعد خنثی نشود). این فیلتر قبلاً اینجا
                    # اعمال نمی‌شد.
                    mature = [r for r in ready
                              if (now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
                    if len(ready) > 1 and mature:
                        # *** رفع ناهماهنگی مهم (بازبینی): قبلاً اینجا همیشه
                        # min-by-occupancy هاردکد بود، بدون توجه به این‌که
                        # الگوریتم (مثلاً VoilaAlgorithm) select_scale_down_victim
                        # سفارشی خودش را override کرده - دقیقاً همان کلاس
                        # مشکلی که قبلاً برای migration_decision در
                        # algorithms/ppo/env.py رفع شده بود (رفتار متفاوت
                        # آموزش/شبیه‌سازی در برابر اجرای واقعی). چون در فاز ۳
                        # اشغال صف واقعی از Redis می‌آید نه از شیء Replica
                        # سایه، از طریق occupancy_fn تزریق می‌شود.
                        victim = self.algorithm.select_scale_down_victim(
                            svc_id, mature, self.servers, now,
                            occupancy_fn=lambda r: redis_state.get_queue_occupancy(svc_id, r.server_id))
                        asyncio.create_task(self._delete_replica(svc_id, victim.server_id))
                        self.metrics.record_scale_action("SCALE_DOWN")
                        self._service_last_scale_time[svc_id] = now

            # snapshot این تیک، قبل از شروع پنجره‌ی بعدی.
            for sid, s in self.servers.items():
                self._util_at_window_start[sid] = s.cumulative_busy_cpu_seconds
            self._util_window_start_time = now
            
            self._tick_total.clear()
            self._tick_rejected.clear()
            self._tick_violated.clear()
            self._tick_proximity_violated.clear()

    # ------------------------------------------------------------------
    # *** حذف کد مرده (بازبینی - اصلاح معماری دیسپچر): dispatch_loop و
    # _handle_request قدیمی که اینجا بودند، دیسپچر مرکزی را هم client هم
    # متریک‌گیر می‌کردند - یعنی خودِ همین ماشین (۱۹۲.۱۶۸.۱.۳۰) هم مسیریابی
    # را انجام می‌داد هم منتظر پاسخ HTTP سنگین /process از پاد می‌ماند و
    # ترافیک واقعی از آن عبور می‌کرد. این دقیقاً همان الگوی غلطی است که
    # باعث تراکم/فشار غیرضروری روی دیسپچر می‌شود. جایگزین صحیح (که از قبل
    # در همین فایل پیاده‌سازی شده و run() واقعی هم از همان استفاده می‌کند):
    #   ۱) route_request() - فقط مسیریابی سبک (BTS<->دیسپچر)، هیچ HTTP سنگینی
    #      از این ماشین رد نمی‌شود.
    #   ۲) BTS واقعی (k8s_adapter/worker_service/bts_simulator.py) خودش
    #      مستقیماً به IP:PORT پاد وصل می‌شود (BTS<->سرور، بدون واسطه).
    #   ۳) record_external_completion()/drain_completion_queue() - دیسپچر
    #      فقط دوره‌ای متریک تکمیل‌شده را از Redis می‌خواند، در مسیر
    #      critical هیچ درخواستی حضور ندارد.
    # این توابع قدیمی هرگز از run()/serve_control_plane() فراخوانی نمی‌شدند
    # (کد مرده)؛ برای جلوگیری از سردرگمی حذف شدند تا کسی اشتباهاً فکر نکند
    # هنوز معماری فعال است.
    # ------------------------------------------------------------------

    async def route_request(self, request_id: int, service_id: int,
                             bts_lat: float, bts_long: float) -> dict:
        """
        *** این تابع معادل نیمه‌ی اول _handle_request قدیمی است (تا لحظه‌ی
        رزرو صف)، ولی دیگر HTTP واقعی نمی‌زند - فقط آدرس مقصد را برمی‌گرداند.
        BTS واقعی این تابع را (از طریق dispatcher_api.py:/route) صدا می‌زند،
        سپس *خودش* مستقیماً به ip:port برگشتی وصل می‌شود.
        """
        route_call_started_at = time.monotonic()
        self._tick_total[service_id] += 1
        self._log("request_arrived", request_id=request_id, service_id=service_id)

        candidates = [r for r in self.replicas_by_service.get(service_id, []) if r.is_selectable()]
        chosen = self.algorithm.select_replica(
            type("R", (), {"bts_lat": bts_lat, "bts_long": bts_long, "service_id": service_id})(),
            candidates, self.servers, time.monotonic())

        if chosen is None:
            status = "REJECTED_NO_REPLICA" if not candidates else "REJECTED_QUEUE_FULL"
            self._tick_rejected[service_id] += 1
            self._tick_violated[service_id] += 1
            self._log("request_rejected", request_id=request_id, service_id=service_id)
            return {"status": status}

        if not redis_state.try_reserve_queue_slot(service_id, chosen.server_id, chosen.queue_len):
            self._tick_rejected[service_id] += 1
            self._tick_violated[service_id] += 1
            self._log("request_rejected", request_id=request_id, service_id=service_id, reason="queue_full")
            return {"status": "REJECTED_QUEUE_FULL"}

        server = self.servers[chosen.server_id]
        distance_km = haversine_km(bts_lat, bts_long, server.lat, server.long)
        delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
        if 2 * delay_ms > CFG.proximity_l0_ms:
            self._tick_proximity_violated[service_id] += 1
        self._log("request_routed", request_id=request_id, server_id=server.id, distance_km=distance_km)

        ip = redis_state.get_pod_ip(service_id, chosen.server_id)
        port = k8s_client.worker_port(service_id)
        deadline = CFG.services_info[service_id]["deadline"]

        # *** رفع ناهماهنگی (بازبینی - اصلاح معماری دیسپچر): قبلاً BTS
        # همیشه deadline کامل سرویس را به‌عنوان مبنای timeout درخواست دوم
        # (/process) می‌گرفت، بدون کسر زمانی که همین الان صرف مرحله‌ی
        # مسیریابی (این تماس /route) شده. این یعنی BTS می‌توانست حتی بعد
        # از عبور از deadline واقعی هم منتظر پاد بماند، چون خودِ timeout
        # دیگر با deadline واقعی هم‌راستا نبود. حالا زمان سپری‌شده‌ی همین
        # مرحله‌ی مسیریابی از deadline کسر می‌شود؛ اگر از قبل منفی/خیلی کم
        # شده (یعنی خودِ مسیریابی هم دیر جواب داده)، حداقل چند صدم ثانیه
        # باقی می‌ماند تا BTS/httpx بلافاصله timeout ندهد.
        routing_elapsed_sec = max(0.0, time.monotonic() - route_call_started_at)
        remaining_deadline = max(deadline - routing_elapsed_sec, 0.1)

        # *** یک رزرو TTL-دار در Redis می‌گذاریم تا اگر BTS هرگز /process را
        # صدا نزد (کرش کلاینت بین /route و /process)، صف برای همیشه اشغال
        # نماند - محافظ در برابر عدم قطعیت شبکه‌ی واقعی که در شبیه‌سازی
        # اصلاً وجود نداشت.
        redis_state.set_reservation_ttl(service_id, chosen.server_id, request_id,
                                         ttl_sec=deadline + 5)

        return {
            "status": "ROUTED",
            "server_id": server.id,
            "ip": ip,
            "port": port,
            "deadline_sec": remaining_deadline,
        }

    def record_external_completion(self, request_id: int, service_id: int, server_id: int,
                                    success: bool, response_time_sec: float):
        """
        *** جایگزین بخش دوم _handle_request قدیمی (بعد از دریافت پاسخ
        HTTP). حالا این اطلاعات را از بیرون (BTS واقعی، از طریق /report)
        دریافت می‌کنیم، نه اینکه خودمان منتظر پاسخ HTTP بمانیم.
        """
        req = Request(id=request_id, bts_lat=0.0, bts_long=0.0, service_id=service_id,
                       arrival_time=time.time())
        req.response_time_sec = response_time_sec
        deadline = CFG.services_info[service_id]["deadline"]
        req.deadline_violated = (not success) or (response_time_sec > deadline)
        req.status = RequestStatus.COMPLETED if success else RequestStatus.REJECTED_NO_REPLICA
        if req.deadline_violated:
            self._tick_violated[service_id] += 1
        self._log("request_completed" if success else "request_failed", request_id=request_id,
                  service_id=service_id, server_id=server_id, response_time_sec=response_time_sec)
        self.metrics.record_request(req)

    async def drain_completion_queue(self):
        """
        *** جایگزین جایگزین: به‌جای اتکا به HTTP /report (که هنوز یک تماس
        شبکه‌ی اضافه از BTS است)، این حلقه‌ی پس‌زمینه هر چند صدم ثانیه
        صف کامل‌شده‌های Redis را که *خودِ پاد worker* پس از پردازش هر
        درخواست آنجا push کرده (نگاه کنید worker_service/app.py) می‌خواند.
        این کاملاً دیسپچر را از مسیر پرترافیک request/response بیرون نگه
        می‌دارد - پاد مستقیماً با BTS صحبت می‌کند، و فقط یک رکورد کوچک
        متریک را (async، batch) به Redis می‌نویسد که این حلقه جمع می‌کند.
        """
        while self._running:
            batch = redis_state.pop_completion_batch(max_items=500)
            for item in batch:
                self.record_external_completion(
                    item["request_id"], item["service_id"], item["server_id"],
                    item["success"], item["response_time_sec"])
            await asyncio.sleep(0.2)

    # ------------------------------------------------------------------
    # *** رفع باگ کد مرده: این فایل قبلاً *دو* تعریف run() داشت (این یکی و
    # نسخه‌ی کامل‌تر پایین‌تر که _utilization_energy_sampler_loop را هم
    # اجرا می‌کند). چون پایتون تعریف دوم را جایگزین اولی می‌کند، این نسخه
    # هرگز واقعاً صدا زده نمی‌شد - ولی برای هرکسی که فایل را می‌خواند
    # گمراه‌کننده بود (به نظر می‌رسید این نسخه‌ی فعال است، در حالی‌که
    # نبود). حذف شد تا فقط یک run() (پایین‌تر) وجود داشته باشد.


        
    # ------------------------------------------------------------------
    
    
    async def run(self, extra_tasks: list | None = None) -> dict:
        redis_state.reset_all(CFG.n_servers, CFG.n_services)
        await self.initial_placement()
        self._util_window_start_time = time.monotonic()

        tasks = [
            self.decision_loop(),
            self.drain_completion_queue(),
            self._utilization_energy_sampler_loop(),
        ]
        if extra_tasks:
            tasks.extend(extra_tasks)

        # *** چون در حالت واقعی هیچ cutoff طبیعی (مثل simulator/engine.py)
        # وجود ندارد، یک watcher اضافه می‌کنیم که بعد از پایان بازه‌ی زمانی
        # واقعی داده (+ کمی margin) self._running را False می‌کند تا حلقه‌ها
        # واقعاً تمام شوند و run() برگردد.
        tasks.append(self._lifetime_watcher())

        await asyncio.gather(*tasks)
        return self.metrics.finalize(self.servers)

    async def _lifetime_watcher(self):
        span_sec = (float(self.events_df.global_start_sec.max())
                    - float(self.events_df.global_start_sec.min())) if len(self.events_df) else 0.0
        margin_sec = CFG.decision_interval_sec + CFG.server_drain_grace_sec + 30.0
        await asyncio.sleep(span_sec + margin_sec)
        self._running = False
        self._log("realtime_run_finished", reason="data_span_elapsed")

    # ------------------------------------------------------------------
    # نقطه‌ی ورود سرویس control-plane (uvicorn این را serve می‌کند)
    # ------------------------------------------------------------------
     
    # *** تابع مستقل سطح‌ماژول (نه متد کلاس) - نقطه‌ی ورود واقعی سرویس
    # control-plane. قبلاً هیچ‌جا (نه run.py، نه هیچ فایل دیگر) صدا زده
    # نمی‌شد؛ الان مستقیماً از run.py برای --mode k8s استفاده می‌شود.
    async def serve_control_plane(events_df, algorithm, algorithm_name, event_logger=None,
                                    http_host="0.0.0.0", http_port=9000) -> dict:
        from k8s_adapter import dispatcher_api
        import uvicorn

        engine = RealtimeEngine(events_df, algorithm, algorithm_name, event_logger=event_logger)
        dispatcher_api.bind_engine(engine)

        config = uvicorn.Config(dispatcher_api.app, host=http_host, port=http_port, log_level="info")
        server = uvicorn.Server(config)

        async def _serve_and_shutdown():
            # *** وقتی _lifetime_watcher علامت پایان می‌زند، خودِ uvicorn هم باید
            # graceful shutdown شود؛ وگرنه asyncio.gather در run() هرگز کامل
            # نمی‌شود چون server.serve() به‌تنهایی هیچ‌وقت خودش تمام نمی‌شود.
            serve_task = asyncio.create_task(server.serve())
            while engine._running and not serve_task.done():
                await asyncio.sleep(1.0)
            if not serve_task.done():
                server.should_exit = True
                await serve_task

        return await engine.run(extra_tasks=[_serve_and_shutdown()])

    async def _utilization_energy_sampler_loop(self):
        """
        *** جایگزین رویدادهای گسسته‌ی موتور شبیه‌سازی: هر UTIL_SAMPLE_INTERVAL_SEC
        ثانیه‌ی واقعی، busy/idle واقعی هر رپلیکا را از Redis (نه از shadow
        replica.departures که هرگز در فاز ۳ به‌روزرسانی نمی‌شود - نگاه کنید
        توضیح بالای فایل) می‌خواند و به‌صورت مستطیلی (rectangle rule) در
        cumulative_busy_cpu_seconds انباشت می‌کند. با گام ۱ ثانیه، این تقریب
        برای burstهای کوتاه (مثلاً یک درخواست ۵ ثانیه‌ای در یک پنجره‌ی ۳۰
        ثانیه‌ای) به‌اندازه‌ی کافی دقیق است - خطای حداکثر هر بازه ~۱ ثانیه است.

        *** به همین مناسبت، انرژی هم اینجا انباشته می‌شود - چون RealtimeEngine
        فعلی اصلاً هیچ معادلی برای _advance_energy_to موتور شبیه‌سازی نداشت؛
        یعنی cumulative_energy_joule در فاز ۳ همیشه ۰ می‌ماند بود (یک باگ جدا،
        ولی چون همان حلقه‌ی نمونه‌برداری برای هر دو لازم است، این‌جا با هم
        رفع می‌شوند).
        """
        last_sample = time.monotonic()
        while self._running:
            await asyncio.sleep(UTIL_SAMPLE_INTERVAL_SEC)
            now = time.monotonic()
            elapsed = now - last_sample
            last_sample = now

            for sid, s in self.servers.items():
                if s.state == ServerState.OFF:
                    continue

                busy_cpu = 0
                if s.state in (ServerState.ACTIVE, ServerState.DRAINING):
                    for svc_id, r in s.hosted_replicas.items():
                        if r.state not in (ReplicaState.READY, ReplicaState.DRAINING):
                            continue
                        # *** منبع درست: اشغال واقعی صف از Redis، نه
                        # r.queue_occupancy(now) که روی shadow همیشه ۰ است.
                        if redis_state.get_queue_occupancy(svc_id, sid) > 0:
                            busy_cpu += CFG.services_info[svc_id]["cpu_demand"]

                s.cumulative_busy_cpu_seconds += busy_cpu * elapsed

                # انرژی: همون مدل خطی idle->max بر پایه‌ی busy_cpu لحظه‌ای همین بازه
                if s.state == ServerState.BOOTING:
                    power = s.p_idle
                elif s.state in (ServerState.ACTIVE, ServerState.DRAINING):
                    util = busy_cpu / s.capacity if s.capacity > 0 else 0.0
                    power = s.p_idle + (s.p_max - s.p_idle) * util
                else:
                    power = 0.0
                s.cumulative_energy_joule += power * elapsed