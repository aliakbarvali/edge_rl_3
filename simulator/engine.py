"""
simulator/engine.py
قبلاً رپلیکای جدید STARTING می‌شد و هم‌زمان رپلیکای قدیم هم بلافاصله
   DRAINING می‌شد. چون رپلیکای DRAINING فوراً از کاندیدهای Router حذف
   می‌شود ولی رپلیکای جدید تا POD_STARTUP_DELAY_SEC ثانیه بعد READY نیست،
   یک پنجره‌ی واقعی وجود داشت که هیچ رپلیکای READY از آن سرویس در دسترس
   نبود -> REJECTED_NO_REPLICA غیرواقعی. حالا رپلیکای قدیم READY می‌ماند و
   فقط پس از READY شدن رپلیکای جدید وارد DRAINING می‌شود ( self._pending_migrations و _handle_replica_ready).

"""

from __future__ import annotations
import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from common.config import CFG
from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, Request, ServerState, ReplicaState, RequestStatus
from common.metrics import MetricsCollector
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from simulator.events import Event, EventType
from collections import deque

class SimulationEngine:
    def __init__(self, events_df: pd.DataFrame, algorithm: AlgorithmBase,
                 algorithm_name: str, event_logger=None, verbose: bool = False):
        self.events_df = events_df
        self.algorithm = algorithm
        self.metrics = MetricsCollector(algorithm=algorithm_name)
        self.logger = event_logger
        self.verbose = verbose

        self.servers: Dict[int, Server] = self._init_servers()
        self.replicas_by_service: Dict[int, List[Replica]] = defaultdict(list)

        self._heap: List[Event] = []
        self._seq = 0
        self.now = 0.0
        self._energy_last_update: Dict[int, float] = {sid: 0.0 for sid in self.servers}
        self._low_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}

        self._tick_total = defaultdict(int)
        self._tick_rejected = defaultdict(int)
        self._tick_lat_sum = defaultdict(float)
        self._tick_lon_sum = defaultdict(float)
        self._tick_violated = defaultdict(int)
        self._tick_response_times: List[float] = []
        self._tick_proximity_violated = defaultdict(int)
        
        self._energy_at_last_tick = 0.0
        self._last_tick_decisions = {"provision": None, "scale": {}}

        self._request_seq = 0
        self._service_demand_centroid: Dict[int, tuple] = {}

        self._pending_migrations: Dict[Tuple[int, int], int] = {}
        self._emergency_boot_for_service: Dict[int, int] = {}
        self._high_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}

     
        self._service_last_scale_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}
        self._service_last_scale_up_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}
        self._recent_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        
    
    def _init_servers(self) -> Dict[int, Server]:
        servers = {}
        for sid, info in CFG.server_info.items():
            prof = CFG.server_profiles[info["profile"]]
            servers[sid] = Server(id=sid, profile=info["profile"], lat=info["lat"], long=info["long"],
                                   capacity=info["capacity"], p_idle=prof["p_idle"], p_max=prof["p_max"])
        return servers

    def _push(self, time: float, etype: EventType, payload=None):
        self._seq += 1
        heapq.heappush(self._heap, Event(time, self._seq, etype, payload))

    def _advance_energy_to(self, t: float):
        for sid, s in self.servers.items():
            last = self._energy_last_update[sid]
            if t > last:
                power = s.instantaneous_power_w(last)
                s.cumulative_energy_joule += power * (t - last)
                self._energy_last_update[sid] = t

    
    # جایگذاری اولیه
    def _initial_placement(self):
        first_window = self.events_df[self.events_df.global_start_sec <=
                                       self.events_df.global_start_sec.min() + CFG.monitor_window_sec]
        active_bts = list(first_window[["Lat", "Long"]].drop_duplicates().itertuples(index=False, name=None))
        selected = self.algorithm.initial_placement(self.servers, active_bts)

        for sid in selected:
            self._start_server_boot(sid)

        for service_id in CFG.active_services:
            target = self._nearest_capable_server(service_id, selected)
            if target is not None:
                self._place_replica(target, service_id)

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


    def _trigger_emergency_boot(self, unmigrated_services: set, draining_server: Server):
        """
        بخش ۶.۲ سند: «اگر هیچ سرور ACTIVE مناسبی پیدا نشد، یک سرور OFF جدید
        Boot اضطراری شود و migration به محض ACTIVE شدنش انجام می‌شود».
        *** قبلاً این‌جا هیچ اقدامی نبود - drain فقط لغو و چرخه‌ی بعد دوباره
        امتحان می‌شد، بدون هیچ پیشرفتی (در لاگ واقعی PPO این باعث ۵۱۰ بار
        تلاش شکست‌خورده‌ی پیاپی روی همان سرور شده بود - نگاه کنید
        server_drain_aborted در ppo_events.jsonl). حالا برای هر سرویسِ
        بی‌مقصد یک سرور OFF مناسب (نزدیک‌ترین با ظرفیت کافی) boot می‌شود.
        """
        off_servers = [x for x in self.servers.values() if x.state == ServerState.OFF]
        reserved_cpu: Dict[int, int] = defaultdict(int)
        for svc_id in unmigrated_services:
            if svc_id in self._emergency_boot_for_service:
                continue  # قبلاً یک سرور برای همین سرویس در حال boot است
            cpu = CFG.services_info[svc_id]["cpu_demand"]
            candidates = [x for x in off_servers if x.capacity - reserved_cpu[x.id] >= cpu]
            if not candidates:
                # هیچ سرور OFF مناسبی موجود نیست؛ چرخه‌ی بعد دوباره امتحان می‌شود
                continue
            candidates.sort(key=lambda x: haversine_km(
                draining_server.lat, draining_server.long, x.lat, x.long))
            target = candidates[0]
            reserved_cpu[target.id] += cpu
            self._emergency_boot_for_service[svc_id] = target.id
            self._start_server_boot(target.id)
            self._log("emergency_boot_triggered", server_id=target.id, service_id=svc_id,
                      source_server_id=draining_server.id, reason="migration_target_unavailable")
            
    # ------------------------------------------------------------------
    # گذارهای سرور
    # ------------------------------------------------------------------
    def _start_server_boot(self, server_id: int):
        s = self.servers[server_id]
        if s.state != ServerState.OFF:
            return
        s.state = ServerState.BOOTING
        s.boot_started_at = self.now
        s.last_transition_time = self.now
        s.num_boots += 1
        s.cumulative_energy_joule += CFG.e_boot_server_j  # انرژی گذار ثابت
        self.metrics.record_transition("server_boot")
        self._log("server_boot_started", server_id=server_id)
        self._push(self.now + CFG.boot_delay_sec, EventType.SERVER_BOOT_DONE, server_id)


    def _handle_boot_done(self, server_id: int):
        s = self.servers[server_id]
        s.state = ServerState.ACTIVE
        s.last_transition_time = self.now
        self._log("server_active", server_id=server_id)
        for r in s.hosted_replicas.values():
            if r.state == ReplicaState.STARTING and r.ready_since is None and r.created_at <= self.now:
                self._schedule_replica_ready(r)

        # *** بخش ۶.۲: اگر این boot به‌خاطر رفع migration_incomplete بود،
        # حالا که ACTIVE شد کاندید معتبری برای migration_decision الگوریتم
        # است (چرخه‌ی بعدی که همان سرور مبدأ دوباره تلاش TURN_OFF می‌کند
        # این را می‌بیند). رکورد انتظار پاک می‌شود تا اگر بعداً دوباره
        # گیر کرد بتوان یک emergency boot جدید trigger کرد.
        rescued = [svc_id for svc_id, target_id in self._emergency_boot_for_service.items()
                   if target_id == server_id]
        for svc_id in rescued:
            del self._emergency_boot_for_service[svc_id]
            self._log("emergency_boot_completed", server_id=server_id, service_id=svc_id)
            
            
    def _start_server_drain(self, server_id: int) -> bool:
        s = self.servers[server_id]
        if s.state != ServerState.ACTIVE:
            return False

        # بخش ۶.۲: مهاجرت رپلیکاهای تک‌رپلیکایی *قبل* از drain کردن خودشان
        steps = self.algorithm.migration_decision(s, self.servers)
        migrated_services = {step.service_id for step in steps}
        sole_hosted = {
            svc_id for svc_id, r in s.hosted_replicas.items()
            if r.state != ReplicaState.TERMINATED and not any(
                other.id != server_id and svc_id in other.hosted_replicas and
                other.hosted_replicas[svc_id].state != ReplicaState.TERMINATED
                for other in self.servers.values())
        }
        # *** محافظ: اگر migration نتواند مقصدی برای همه‌ی سرویس‌های تک‌رپلیکا
        # پیدا کند، drain را این چرخه لغو کن (به‌جای قطع کامل آن سرویس‌ها).
        # طبق بخش ۶.۲ در این حالت باید یک سرور جدید Boot اضطراری شود؛ فعلاً
        # به‌عنوان راه‌حل ایمن‌تر، drain را عقب می‌اندازیم تا چرخه‌ی بعد با
        # وضعیت ظرفیت جدید دوباره امتحان شود.
      
        unmigrated = sole_hosted - migrated_services
        if unmigrated:
            self._trigger_emergency_boot(unmigrated, s)
            self._log("server_drain_aborted", server_id=server_id,
                      reason="migration_incomplete", unmigrated=list(unmigrated))
            return False
        s.state = ServerState.DRAINING
        s.drain_started_at = self.now
        s.last_transition_time = self.now
        self._log("server_drain_started", server_id=server_id)

        # *** Make-Before-Break (بخش ۶.۲): رپلیکای جدید را روی مقصد می‌سازیم
        # ولی رپلیکای قدیم را *هنوز* drain نمی‌کنیم؛ فقط علامت "در انتظار
        # مهاجرت" می‌زنیم. تا وقتی رپلیکای جدید READY نشود، رپلیکای قدیم
        # همچنان یک کاندید معتبر برای Router باقی می‌ماند (سرویس هرگز بدون
        # replica نمی‌ماند).
        for step in steps:
            self._log("migration_started", service_id=step.service_id,
                      from_server_id=server_id, to_server_id=step.target_server_id)
            self._place_replica(step.target_server_id, step.service_id)
            
            
            self._pending_migrations[(step.target_server_id, step.service_id)] = server_id
    

        # سرویس‌هایی که مهاجرت نمی‌کنند (چندرپلیکایی/رپلیکای دیگری هم دارند)
        # طبق ۶.۲ فوراً DRAINING می‌شوند؛ سرویس‌های در حال مهاجرت را تا
        # تکمیل مهاجرت دست نمی‌زنیم (_handle_replica_ready آن‌ها را در
        # زمان مناسب drain می‌کند).
        for r in list(s.hosted_replicas.values()):
            if r.service_id in migrated_services:
                continue
            self._start_replica_drain(r)

        self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
        return True

    def _handle_drain_done(self, server_id: int):
        s = self.servers[server_id]
        # اگر هنوز رپلیکایی (در حال پردازش، در انتظار مهاجرت، یا در حال
        # تخلیه) روی این سرور باقی مانده، صبر بیشتر (نگاه کنید بند make-
        # before-break بالا: رپلیکای READY منتظر تکمیل مهاجرت هم باید اینجا
        # لحاظ شود، نه فقط DRAINING).
        if any(r.state != ReplicaState.TERMINATED for r in s.hosted_replicas.values()):
            self._push(self.now + CFG.server_drain_grace_sec, EventType.SERVER_DRAIN_DONE, server_id)
            return
        s.hosted_replicas = {sid: r for sid, r in s.hosted_replicas.items()
                              if r.state != ReplicaState.TERMINATED}
        s.state = ServerState.OFF
        s.last_transition_time = self.now
        s.num_shutdowns += 1
        self.metrics.record_transition("server_shutdown")
        self._log("server_off", server_id=server_id)

    # ------------------------------------------------------------------
    # گذارهای رپلیکا
    # ------------------------------------------------------------------
    def _place_replica(self, server_id: int, service_id: int) -> Optional[Replica]:
        s = self.servers[server_id]
        cpu = CFG.services_info[service_id]["cpu_demand"]
        if not s.can_host(service_id, cpu):
            return None
        svc = CFG.services_info[service_id]
        r = Replica(service_id=service_id, server_id=server_id,
                    queue_len=svc["queue_len"], exec_time=svc["exec_time"], created_at=self.now)
        s.hosted_replicas[service_id] = r
        self.replicas_by_service[service_id].append(r)
        self.metrics.record_transition("pod_create")
        s.cumulative_energy_joule += CFG.e_pod_create_j
        self._log("pod_create_started", server_id=server_id, service_id=service_id)
        if s.state == ServerState.ACTIVE:
            self._schedule_replica_ready(r)
        return r

    def _schedule_replica_ready(self, r: Replica):
        self._push(self.now + CFG.pod_startup_delay_sec, EventType.REPLICA_READY,
                    (r.server_id, r.service_id))

    def _handle_replica_ready(self, key):
        server_id, service_id = key
        r = self.servers[server_id].hosted_replicas.get(service_id)
        if r is None or r.state != ReplicaState.STARTING:
            return
        r.state = ReplicaState.READY
        r.ready_since = self.now
        self._log("pod_ready", server_id=server_id, service_id=service_id)

        # *** اگر این رپلیکا مقصد یک migration در انتظار بود، حالا که READY
        # شد نوبت drain کردن رپلیکای قدیم (مبدأ) است - همین‌جا make-before-
        # break تکمیل می‌شود (بخش ۶.۲).
        pending_key = (server_id, service_id)
        source_server_id = self._pending_migrations.pop(pending_key, None)
        if source_server_id is not None:
            source_server = self.servers.get(source_server_id)
            old_replica = source_server.hosted_replicas.get(service_id) if source_server else None
            if old_replica is not None and old_replica.state not in (ReplicaState.TERMINATED,
                                                                       ReplicaState.DRAINING):
                self._start_replica_drain(old_replica)
            self._log("migration_completed", server_id=source_server_id, service_id=service_id,
                      target_server_id=server_id)

    def _start_replica_drain(self, r: Replica):
        if r.state == ReplicaState.TERMINATED:
            return
        r.state = ReplicaState.DRAINING
        r.drain_started_at = self.now
        self._log("pod_drain_started", server_id=r.server_id, service_id=r.service_id)
        # *** بخش ۲.۴/۹: به‌جای تأخیر ثابت یکسان برای هر ۱۵ سرویس، منتظر
        # زمان واقعی خالی‌شدن این replica می‌مانیم (r.available_at = زمان
        # اتمام آخرین درخواست پذیرفته‌شده، از try_admit). سرویس‌های کند با
        # صف پر (مثلاً سرویس ۱۵: exec_time=120, queue_len=10) می‌توانند تا
        # ۱۲۰۰ ثانیه واقعی برای خالی‌شدن نیاز داشته باشند - در برابر
        # GRACEFUL_TERMINATION_DELAY_SEC=10s ثابت قبلی که فقط برای صف خالی
        # کافی بود.
        drain_wait = max(CFG.graceful_termination_delay_sec, r.available_at - self.now)
        self._push(self.now + drain_wait, EventType.REPLICA_TERMINATED,
                    (r.server_id, r.service_id))

    def _handle_replica_terminated(self, key):
        server_id, service_id = key
        s = self.servers[server_id]
        r = s.hosted_replicas.get(service_id)
        if r is None:
            return
        r.state = ReplicaState.TERMINATED
        self.metrics.record_transition("pod_delete")
        self._log("pod_terminated", server_id=server_id, service_id=service_id)
        del s.hosted_replicas[service_id]
        self.replicas_by_service[service_id] = [
            x for x in self.replicas_by_service[service_id] if x is not r]

    # ------------------------------------------------------------------
    # بخش ۳: چرخه‌ی درخواست
    # ------------------------------------------------------------------
    def _handle_arrival(self, row):
        self._request_seq += 1
        req = Request(id=self._request_seq, bts_lat=row.Lat, bts_long=row.Long,
                       service_id=int(row.ServiceID), arrival_time=self.now)
        self._tick_total[req.service_id] += 1
        self._tick_lat_sum[req.service_id] += req.bts_lat
        self._tick_lon_sum[req.service_id] += req.bts_long
        # *** رفع حلقه‌ی بازخورد خودتقویت‌شونده (بخش ۵ سند - فلسفه‌ی VOILA):
        # قبلاً موقعیت فقط برای درخواست‌های COMPLETED ثبت می‌شد؛ یعنی
        # مناطقی که به‌خاطر نبود پوشش دائم رد می‌شدند هرگز در demand_centroid
        # دیده نمی‌شدند (کمبود پوشش -> رد شدن -> centroid کور به همان کمبود
        # -> کمبود ادامه‌دار). حالا موقعیت در لحظه‌ی ورود ثبت می‌شود، صرف‌نظر
        # از نتیجه‌ی نهایی (موفق یا رد).
        self._recent_positions[req.service_id].append((req.bts_lat, req.bts_long))
        self._log("request_arrived", request_id=req.id, service_id=req.service_id)

        candidates = [r for r in self.replicas_by_service.get(req.service_id, [])
                      if r.is_selectable()]
        chosen = self.algorithm.select_replica(req, candidates, self.servers, self.now)

        if chosen is None:
            if not candidates:
                req.status = RequestStatus.REJECTED_NO_REPLICA
                self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                          reason="no_replica")
            else:
                req.status = RequestStatus.REJECTED_QUEUE_FULL
                self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                          reason="queue_full")
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._finalize_request(req)
            return

        server = self.servers[chosen.server_id]
        distance_km = haversine_km(req.bts_lat, req.bts_long, server.lat, server.long)
        delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
        
        req._distance_km = distance_km
        req.network_delay_ms = delay_ms
        req.assigned_server_id = server.id

        # *** رفع باگ: قبلاً از CFG.l0_ms (ثابت پوشش اولیه، خیلی سخاوتمندانه)
        # استفاده می‌شد که هرگز trigger نمی‌شد - نگاه کنید common/config.py:PROXIMITY_L0_MS.
        if 2 * delay_ms > CFG.proximity_l0_ms:
            self._tick_proximity_violated[req.service_id] += 1
        
        
         
        self._log("request_routed", request_id=req.id, service_id=req.service_id,
            server_id=server.id, distance_km=distance_km, network_delay_ms=delay_ms)
        cold_start_extra = 0.0
        if chosen.ready_since is not None and (self.now - chosen.ready_since) <= CFG.cold_start_window_sec:
            cold_start_extra = CFG.cold_start_penalty_sec

        admit = chosen.try_admit(self.now, cold_start_extra=cold_start_extra)
        if admit is None:
            # *** این شاخه با تمام پیاده‌سازی‌های فعلی select_replica
            # (AlgorithmBase/Voila/PPO) هرگز نباید برسد، چون همه از قبل فقط
            # replicaیی با جای خالی را برمی‌گردانند و بین چک و اینجا هیچ
            # رویداد دیگری در موتور تک‌نخی پردازش نمی‌شود. اگر این لاگ
            # دیدید، یعنی یک select_replica سفارشی این فرض را نقض کرده -
            # بررسی کنید که آیا واقعاً race وجود دارد.
            self._log("unexpected_admit_race", request_id=req.id, service_id=req.service_id,
                      server_id=chosen.server_id)
            req.status = RequestStatus.REJECTED_QUEUE_FULL
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._log("request_rejected", request_id=req.id, service_id=req.service_id,
                      reason="queue_full")
            self._finalize_request(req)
            return

        req.queue_enter_time = admit["queue_enter_time"]
        req.service_start_time = admit["service_start_time"]
        req.service_end_time = admit["service_end_time"]
        req.wait_time_sec = admit["wait_time_sec"]
        self._log("request_queued", request_id=req.id, service_id=req.service_id,
                  server_id=server.id, wait_time_sec=req.wait_time_sec)
        # *** بخش ۲.۴: بدون این رویداد، _advance_energy_to فقط در لحظه‌ی
        # رویداد بعدی (که می‌تواند تا DECISION_INTERVAL_SEC=30s دیرتر باشد)
        # صدا زده می‌شد - یعنی دوره‌ی idle-شدنِ واقعیِ این replica با توانِ
        # "مشغول" محاسبه می‌شد. این رویداد سبک صرفاً موتور را دقیقاً در لحظه‌ی
        # اتمام واقعی این درخواست "بیدار" می‌کند.
        self._push(req.service_end_time, EventType.ENERGY_RESYNC, None)        
        # بخش ۳: response_time = ۲×network_delay (رفت+برگشت) + wait + exec (+cold start در exec لحاظ شد)
        req.response_time_sec = (2 * delay_ms / 1000.0) + req.wait_time_sec + \
                                 (req.service_end_time - req.service_start_time)
        req.deadline_violated = req.response_time_sec > CFG.services_info[req.service_id]["deadline"]
        req.status = RequestStatus.COMPLETED
        if req.deadline_violated:
            self._tick_violated[req.service_id] += 1
 
        self._tick_response_times.append(req.response_time_sec)
        self._log("request_completed", request_id=req.id, service_id=req.service_id,
                  server_id=server.id, response_time_sec=req.response_time_sec)

        self._finalize_request(req)
    def _finalize_request(self, req: Request):
        self.metrics.record_request(req)

    def _log(self, event_type: str, **fields):
        if self.logger is not None:
            self.logger.log(event_type, sim_time=self.now, **fields)

    # ------------------------------------------------------------------
    # بخش ۶ و ۷: تیک تصمیم (هر DECISION_INTERVAL_SEC)
    # ------------------------------------------------------------------
    def peek_snapshot(self) -> dict:
        """
        snapshot لحظه‌ای بدون اجرای منطق تیک تصمیم (فقط خواندن وضعیت فعلی).
        برای observation اولیه‌ی reset() محیط PPO استفاده می‌شود - چون بلافاصله
        بعد از prime() هنوز هیچ اکشنی از عامل نرسیده و نباید تیک واقعی رخ دهد.
        """
        return self._build_metrics_snapshot_readonly()

    def _build_metrics_snapshot_readonly(self) -> dict:
        """مثل _build_metrics_snapshot ولی شمارنده‌های تیک را پاک/تغییر نمی‌دهد."""
        snapshot = {"servers": {}, "services": {}, "global": {}}
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(self.now),
                "free_capacity": s.free_capacity(),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature_ready = [r for r in ready
                             if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
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
                               "num_rejected_recent": 0}
        return snapshot
        
    def _build_metrics_snapshot(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(self.now),
                "free_capacity": s.free_capacity(),
            }
        def _medoid(points):
            if not points:
                return None
            best, best_cost = points[0], float("inf")
            for p in points:
                cost = sum(haversine_km(p[0], p[1], q[0], q[1]) for q in points)
                if cost < best_cost:
                    best, best_cost = p, cost
            return best
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature_ready = [r for r in ready
                             if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(r.queue_occupancy(self.now) for r in ready) / len(ready)) if ready else 0.0
            if self._tick_total[svc_id] > 0:
                
                self._service_demand_centroid[svc_id] = _medoid(list(self._recent_positions[svc_id]))
                #self._service_demand_centroid[svc_id] = (new_lat, new_lon)
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
        current_energy = sum(s.cumulative_energy_joule for s in self.servers.values())
        snapshot["global"] = {
            "avg_response_time_recent": (sum(self._tick_response_times) / len(self._tick_response_times))
                                         if self._tick_response_times else 0.0,
            "energy_recent_joule": current_energy - self._energy_at_last_tick,
            "num_rejected_recent": sum(self._tick_rejected.values()),
        }
        self._energy_at_last_tick = current_energy
        return snapshot
        
    def _apply_provisioning(self, action, snapshot: dict):
        self._last_tick_decisions["provision"] = action      
        applied = False
        skip_reason = None 
        turn_on_necessary = (self._any_active_server_sustained_overloaded()
                              or self._any_service_capacity_starved(snapshot))
        turn_off_opportunity = self._any_active_server_sustained_underloaded()

        if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
            s = self.servers[action.server_id]
            # *** بخش ۶.۱: علاوه بر cooldown، حالا باید حداقل یک سرور ACTIVE
            # وجود داشته باشد که overload‌اش *مداوم* (>= SUSTAIN_HIGH_SEC) بوده
            # باشد - نه صرفاً یک نمونه‌ی لحظه‌ای این تیک (قبلاً این تداوم
            # اصلاً چک نمی‌شد؛ نگاه کنید _update_sustain_tracking).
            if s.state != ServerState.OFF:
                skip_reason = "not_off"
            elif s.in_cooldown(self.now, CFG.cooldown_sec):
                skip_reason = "cooldown"
            elif not turn_on_necessary:
                skip_reason = "overload_not_sustained"
            else:
                self._start_server_boot(action.server_id)
                self.metrics.record_scale_action("TURN_ON")
                applied = True
                # *** رفع خودارجاعی: به‌جای True ثابت (که چون gate بالا خودش
                # پیش‌شرط اعمال بود، نتیجه از پیش تضمین می‌شد)، از یک معیار
                # لحظه‌ای و مستقل استفاده می‌شود - نگاه کنید
                # _was_turn_on_necessary_audit.
                self.metrics.record_decision_correctness(
                    "TURN_ON", self._was_turn_on_necessary_audit(snapshot))
        elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
            s = self.servers[action.server_id]
            n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
            turn_off_necessary = self._was_turn_off_necessary(action.server_id)
            if s.state != ServerState.ACTIVE:
                skip_reason = "not_active"
            elif not turn_off_necessary:
                skip_reason = "low_util_not_sustained"
            elif n_active <= 1:
                skip_reason = "last_active_server"
            elif s.in_cooldown(self.now, CFG.cooldown_sec):
                skip_reason = "cooldown"
            elif (self.now - s.last_transition_time) < CFG.min_active_duration_sec:
                skip_reason = "min_active_duration"
            else: 
                if self._start_server_drain(action.server_id):
                    self.metrics.record_scale_action("TURN_OFF")
                    applied = True
                    # *** رفع خودارجاعی، مشابه TURN_ON بالا.
                    self.metrics.record_decision_correctness(
                        "TURN_OFF", self._was_turn_off_necessary_audit(action.server_id, snapshot))
                else:
                    skip_reason = "migration_incomplete"

        # *** بخش ۸: فرصت ازدست‌رفته - صرف‌نظر از تصمیم این تیک، آیا طبق
        # معیار مستقل سیستم واقعاً به TURN_ON/TURN_OFF نیاز داشت ولی این
        # تیک اعمال نشد؟
        if turn_on_necessary and not (action.action == ProvisionActionType.TURN_ON and applied):
            self.metrics.record_missed_opportunity("TURN_ON")
        if turn_off_opportunity and not (action.action == ProvisionActionType.TURN_OFF and applied):
            self.metrics.record_missed_opportunity("TURN_OFF")

        # *** بخش ۱۲: لاگ provision_decision با وضعیت نهایی اعمال/رد و
        # نتیجه‌ی ممیزی مستقل (audit trail کامل).
        self._log("provision_decision", action=action.action.name, server_id=action.server_id,
                  applied=applied, skip_reason=skip_reason,
                  necessary_turn_on=turn_on_necessary, turn_off_opportunity=turn_off_opportunity)

    def _any_active_server_sustained_overloaded(self) -> bool:
        """بخش ۶.۱: آیا حداقل یک سرور ACTIVE وجود دارد که overload‌اش برای
        حداقل CFG.sustain_high_sec به‌طور مداوم برقرار بوده (نه یک نمونه‌ی
        لحظه‌ای تنها)؟ نگاه کنید _update_sustain_tracking برای پرشدن
        self._high_util_since. این تابع هم برای gate کردن TURN_ON و هم
        به‌عنوان معیار ممیزی مستقلِ بخش ۸ (decision correctness) استفاده
        می‌شود - عمداً، چون هر دو یک سؤال را می‌پرسند: «آیا سیستم واقعاً به
        TURN_ON نیاز داشت؟»."""
        for sid, since in self._high_util_since.items():
            if since is not None and (self.now - since) >= CFG.sustain_high_sec:
                return True
        return False

    def _any_service_capacity_starved(self, snapshot: dict) -> bool:
        """
        *** یافته‌ی جدید بعد از فعال‌شدن واقعی TURN_OFF (به لطف فیکس
        emergency-boot): معیار قدیمی turn_on_necessary فقط utilization
        لحظه‌ای (busy-fraction) سرورهای ACTIVE را می‌سنجد، نه اینکه اصلاً
        ظرفیت آزاد برای رپلیکای جدید مانده یا نه. یک سرور می‌تواند کاملاً
        پر (free_capacity=0) باشد ولی چون هم‌زمان صددرصد busy نیست
        utilization<0.95 بماند - یعنی TURN_ON هرگز trigger نمی‌شد حتی وقتی
        3500 از 3508 تلاش SCALE_UP واقعی با no_target_server شکست خورده
        بودند (نگاه کنید greedy_events.jsonl). این تابع سیگنال مکمل است.
        """
   
        for svc_id in CFG.active_services:
            if not self._was_scale_up_necessary(svc_id, snapshot):
                continue
            cpu = CFG.services_info[svc_id]["cpu_demand"]
            # *** هماهنگ با algorithms/base.py:_capacity_starved_services -
            # سرور BOOTING هم می‌تواند به‌زودی این starvation را رفع کند،
            # نباید نادیده گرفته شود.
            if not any(s.state in (ServerState.ACTIVE, ServerState.BOOTING) and s.can_host(svc_id, cpu)
                       for s in self.servers.values()):
                return True
        return False
    def _any_active_server_sustained_underloaded(self) -> bool:
        """قرینه‌ی بالا برای TURN_OFF: آیا حداقل یک سرور ACTIVE (غیر از آخرین
        سرور فعال سیستم) برای مدت کافی زیر آستانه بوده؟ برای «فرصت ازدست‌رفته»ی
        بخش ۸ استفاده می‌شود."""
        n_active = sum(1 for s in self.servers.values() if s.state == ServerState.ACTIVE)
        if n_active <= 1:
            return False
        for sid, since in self._low_util_since.items():
            if since is not None and (self.now - since) >= CFG.sustain_low_sec:
                return True
        return False

    def _was_turn_off_necessary(self, server_id: int) -> bool:
        since = self._low_util_since.get(server_id)
        return since is not None and (self.now - since) >= CFG.sustain_low_sec
    def _was_turn_on_necessary_audit(self, snapshot: dict) -> bool:
        """بخش ۸: معیار ممیزی *مستقل* برای TURN_ON. برخلاف turn_on_necessary
        در _apply_provisioning (که به‌عنوان gate عمل می‌کند و نیاز به تداوم
        SUSTAIN_HIGH_SEC ثانیه‌ای overload دارد)، این تابع فقط وضعیت
        *لحظه‌ای* همین snapshot را می‌سنجد - بدون نیاز به تداوم. بدون این
        تفکیک، هر TURN_ON اعمال‌شده (که فقط بعد از عبور از gate ممکن
        می‌شود) به‌تعریف correct ثبت می‌شد؛ یک خودارجاعی بدون ارزش ممیزی
        واقعی. حالا ممکن است gate اجازه دهد (چون تداوم ۳۰ ثانیه‌ای برقرار
        بوده) ولی audit بگوید در همین لحظه‌ی خاص دیگر overloaded نیست -
        این دقیقاً نشانه‌ی استقلال واقعی دو معیار است.
        """
        overloaded_now = any(
            snapshot["servers"][sid]["utilization"] > CFG.util_scale_up_threshold
            for sid, s in self.servers.items() if s.state == ServerState.ACTIVE
        )
        return overloaded_now or self._any_service_capacity_starved(snapshot)

    def _was_turn_off_necessary_audit(self, server_id: int, snapshot: dict) -> bool:
        """قرینه‌ی بالا برای TURN_OFF: وضعیت لحظه‌ای utilization همین سرور
        خاص، مستقل از تداوم SUSTAIN_LOW_SEC که در گیت واقعی (_low_util_since)
        لازم است."""
        return snapshot["servers"][server_id]["utilization"] < CFG.util_scale_down_threshold
    def _was_scale_up_necessary(self, svc_id: int, snapshot: dict) -> bool:
        """بخش ۸: معیار ممیزی *مستقل* از threshold داخلی هر الگوریتم (Greedy،
        Voila، HPA، PPO هر کدام threshold/فرمول خودشان را دارند) - یک خط‌کش
        واحد برای مقایسه‌ی منصفانه‌ی «درستی» تصمیم هر ۴ الگوریتم."""
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
        return occ_ratio > CFG.decision_audit_scale_up_occ_threshold or sv["rejection_rate"] > 0.0

    def _was_scale_down_necessary(self, svc_id: int, snapshot: dict) -> bool:
        sv = snapshot["services"][svc_id]
        occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0 
        return occ_ratio < CFG.decision_audit_scale_down_occ_threshold and sv["n_ready_replicas"] > 1
    
    def _apply_scale_decision(self, svc_id: int, decision: ScaleAction, snapshot: dict):
        self._last_tick_decisions["scale"][svc_id] = decision
            
        applied = False
        skip_reason = None
        necessary_up = self._was_scale_up_necessary(svc_id, snapshot)
        necessary_down = self._was_scale_down_necessary(svc_id, snapshot)

        if decision == ScaleAction.NO_CHANGE:
            pass
        elif (self.now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec:
            # *** بخش ۷: «Cooldown مشابه ۶.۱ برای هر service_id» - قبلاً این
            # کنترل اصلاً وجود نداشت و هر تیک می‌توانست دوباره SCALE_UP/DOWN
            # بزند (flapping)، دقیقاً چیزی که سند صراحتاً می‌خواست جلویش
            # گرفته شود.
            skip_reason = "cooldown"
        elif decision == ScaleAction.SCALE_UP:
            target = self.algorithm.select_placement_server(svc_id, self.servers)
            if target is not None:
                placed = self._place_replica(target, svc_id)
                if placed is not None:
                    self.metrics.record_scale_action("SCALE_UP")
                    self._service_last_scale_time[svc_id] = self.now
                    self._service_last_scale_up_time[svc_id] = self.now
                    applied = True
                    self.metrics.record_decision_correctness("SCALE_UP", necessary_up)
                else:
                    skip_reason = "placement_failed"
            else:
                skip_reason = "no_target_server"
        elif decision == ScaleAction.SCALE_DOWN:
            # *** رفع باگ: قبلاً محافظت anti-flapping در سطح سرویس بود
            # (since_last_up از _service_last_scale_up_time) - یعنی فقط از
            # "scale-down بلافاصله بعد از هر scale-up" جلوگیری می‌کرد، بدون
            # سنجش سن خودِ replica انتخاب‌شده به‌عنوان قربانی. پس بعد از
            # MIN_REPLICA_AGE_BEFORE_SCALE_DOWN_SEC ثانیه از آخرین SCALE_UP،
            # select_scale_down_victim (کمترین اشغال) می‌توانست دقیقاً همان
            # replica تازه‌ساخته را انتخاب کند - چون تازه‌ساخته‌ها معمولاً
            # کم‌بارترین‌اند - و هدف اصلی ضد-flapping دور زده می‌شد. حالا
            # کاندیدهای حذف مستقیماً بر اساس created_at خودِ replica فیلتر
            # می‌شوند.
            ready = [r for r in self.replicas_by_service.get(svc_id, [])
                     if r.state == ReplicaState.READY]
            if len(ready) > 1:
                mature = [r for r in ready
                          if (self.now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
                if not mature:
                    skip_reason = "no_mature_replica"
                else:
                    victim = self.algorithm.select_scale_down_victim(svc_id, mature, self.servers, self.now)
                    self._start_replica_drain(victim)
                    self.metrics.record_scale_action("SCALE_DOWN")
                    self._service_last_scale_time[svc_id] = self.now
                    applied = True
                    self.metrics.record_decision_correctness("SCALE_DOWN", necessary_down)
            else:
                skip_reason = "only_one_replica_left"

        # *** بخش ۸: فرصت ازدست‌رفته - صرف‌نظر از تصمیم این تیک، آیا طبق
        # معیار مستقل واقعاً SCALE_UP/DOWN لازم بود ولی اعمال نشد؟
        if necessary_up and not (decision == ScaleAction.SCALE_UP and applied):
            self.metrics.record_missed_opportunity("SCALE_UP")
        if necessary_down and not (decision == ScaleAction.SCALE_DOWN and applied):
            self.metrics.record_missed_opportunity("SCALE_DOWN")

        # *** بخش ۱۲: لاگ رویداد scale_decision برای هر سرویس هر تیک، همراه
        # با وضعیت نهایی اعمال/رد و نتیجه‌ی ممیزی مستقل (audit trail کامل).
        self._log("scale_decision", service_id=svc_id, decision=decision.name,
                  applied=applied, skip_reason=skip_reason,
                  necessary_scale_up=necessary_up, necessary_scale_down=necessary_down)

    def _update_sustain_tracking(self, snapshot: dict):
        for sid, s in self.servers.items():
            if s.state != ServerState.ACTIVE:
                self._low_util_since[sid] = None
                self._high_util_since[sid] = None
                continue
            util = snapshot["servers"][sid]["utilization"]
            if util < CFG.util_scale_down_threshold:
                if self._low_util_since[sid] is None:
                    self._low_util_since[sid] = self.now
            else:
                self._low_util_since[sid] = None

            if util > CFG.util_scale_up_threshold:
                if self._high_util_since[sid] is None:
                    self._high_util_since[sid] = self.now
            else:
                self._high_util_since[sid] = None

    def _handle_decision_tick(self, external_actions: dict | None = None) -> dict:
        """
        بخش ۶ و ۷: تیک تصمیم. اگر external_actions داده شود (فقط برای آموزش
        PPO از طریق algorithms/ppo/env.py استفاده می‌شود)، به‌جای فراخوانی
        متدهای self.algorithm از آن استفاده می‌شود؛ در غیر این‌صورت رفتار
        عادی (Greedy/Voila/HPA/PPO-inference از طریق ppo_algorithm.py) اجرا
        می‌شود. خروجی: همان metrics_snapshot این تیک (برای ساخت reward/state).
        """
        self.metrics.record_snapshot(self.now, self.servers)
        snapshot = self._build_metrics_snapshot()
        self._update_sustain_tracking(snapshot)
        self._last_tick_decisions = {"provision": None, "scale": {}}

        if external_actions is not None:
            self._apply_provisioning(external_actions["provision"], snapshot)
            for svc_id, decision in external_actions["scale"].items():
                self._apply_scale_decision(svc_id, decision, snapshot)
        else:
            action = self.algorithm.provision_decision(self.servers, snapshot, self.now)
            self._apply_provisioning(action, snapshot)
            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)
                self._apply_scale_decision(svc_id, decision, snapshot)

        self._tick_total.clear()
        self._tick_rejected.clear()
        self._tick_violated.clear()
        self._tick_response_times.clear()
        self._tick_lat_sum.clear()
        self._tick_lon_sum.clear()
        self._tick_proximity_violated.clear()
        return snapshot

    # ------------------------------------------------------------------
    # اجرای batch عادی (Greedy/Voila/HPA/PPO-inference از طریق ppo_algorithm.py)
    # ------------------------------------------------------------------
    def run(self) -> dict:
        self.prime()
        while True:
            _, done = self.step()
            if done:
                break
        return self.metrics.finalize(self.servers)

    # ------------------------------------------------------------------
    # اجرای قابل‌step (برای algorithms/ppo/env.py هنگام آموزش)
    # ------------------------------------------------------------------
    def prime(self):
        """بارگذاری رویدادهای ورود + جایگذاری اولیه؛ باید یک‌بار قبل از step() فراخوانی شود."""
        for row in self.events_df.itertuples(index=False):
            self._push(float(row.global_start_sec), EventType.REQUEST_ARRIVAL, row)
        start_time = float(self.events_df.global_start_sec.min()) if len(self.events_df) else 0.0
        max_time = float(self.events_df.global_start_sec.max()) if len(self.events_df) else 0.0
        self.now = start_time
        self._initial_placement()
        self._push(start_time, EventType.DECISION_TICK)
        self._cutoff = max_time + CFG.decision_interval_sec + CFG.server_drain_grace_sec

    def step(self, external_actions: dict | None = None):
        """
        رویدادها را تا رسیدن به تیک تصمیم *بعدی* پردازش می‌کند.
        خروجی: (metrics_snapshot این تیک یا None اگر پایان یافت, done: bool)
        """
        while self._heap:
            ev = heapq.heappop(self._heap)
            if ev.time > self._cutoff:
                return None, True
            self._advance_energy_to(ev.time)
            self.now = ev.time

            if ev.type == EventType.REQUEST_ARRIVAL:
                self._handle_arrival(ev.payload)
            elif ev.type == EventType.DECISION_TICK:
                snapshot = self._handle_decision_tick(external_actions)
                self._push(self.now + CFG.decision_interval_sec, EventType.DECISION_TICK)
                return snapshot, False
            elif ev.type == EventType.SERVER_BOOT_DONE:
                self._handle_boot_done(ev.payload)
            elif ev.type == EventType.SERVER_DRAIN_DONE:
                self._handle_drain_done(ev.payload)
            elif ev.type == EventType.REPLICA_READY:
                self._handle_replica_ready(ev.payload)
            elif ev.type == EventType.REPLICA_TERMINATED:
                self._handle_replica_terminated(ev.payload)
            elif ev.type == EventType.ENERGY_RESYNC:
                pass  # فقط برای دقیق‌کردن _advance_energy_to (بالای همین حلقه) لازم بود
        return None, True