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

معماری: دو تسک asyncio هم‌زمان:
    ۱) decision_loop: هر DECISION_INTERVAL_SEC ثانیه‌ی *واقعی*، دقیقاً همان
       چهار متد AlgorithmBase (scale_decision/provision_decision/
       select_placement_server/migration_decision) را که Greedy/Voila/HPA/
       PPO پیاده کرده‌اند صدا می‌زند - منطق تصمیم‌گیری هیچ تغییری نمی‌کند،
       فقط اجرای آن (k8s_client به‌جای دستکاری آبجکت در حافظه) عوض می‌شود.
    ۲) dispatch_loop: رویدادهای CSV را به ترتیب و با فاصله‌ی زمانی *واقعی*
       (نه فشرده) پخش می‌کند، select_replica مشترک را صدا می‌زند، و با HTTP
       واقعی (httpx) به IP واقعی پاد درخواست می‌فرستد.

یک نمای «سایه» (self.servers/self.replicas_by_service، همان دیتاکلاس‌های
common/models.py) در حافظه نگه‌داشته می‌شود تا بتوان از همان
AlgorithmBase.select_replica/initial_placement مشترک استفاده کرد بدون
بازنویسی منطق - این نما با هر عملیات واقعی (create/delete/ready) و با
Redis همگام می‌ماند.
"""

from __future__ import annotations
import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional

import httpx
import pandas as pd

from common.config import CFG
from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, ServerState, ReplicaState, Request, RequestStatus
from common.metrics import MetricsCollector
from common.logger import EventLogger
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from k8s_adapter import k8s_client, redis_state


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
        for step in steps:
            self._log("migration_started", service_id=step.service_id,
                      from_server_id=server_id, to_server_id=step.target_server_id)
            await self._create_replica(step.target_server_id, step.service_id)
            self._log("migration_completed", service_id=step.service_id,
                      from_server_id=server_id, to_server_id=step.target_server_id)

        for service_id in list(s.hosted_replicas.keys()):
            await self._delete_replica(service_id, server_id)

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
            if k8s_client.is_deployment_ready(service_id, server_id):
                ip = k8s_client.get_pod_ip(service_id, server_id)
                if ip:
                    redis_state.set_pod_ip(service_id, server_id, ip)
                    redis_state.set_replica_state(service_id, server_id, "READY")
                    replica.state = ReplicaState.READY
                    replica.ready_since = time.monotonic()
                    self._log("pod_ready", server_id=server_id, service_id=service_id, pod_ip=ip)
                    return
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
    def _build_metrics_snapshot(self) -> dict:
        snapshot = {"servers": {}, "services": {}, "global": {}}
        now = time.monotonic()
        for sid, s in self.servers.items():
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": s.instantaneous_utilization(now),
                "free_capacity": s.free_capacity(),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(redis_state.get_queue_occupancy(svc_id, r.server_id) for r in ready)
                       / len(ready)) if ready else 0.0
            snapshot["services"][svc_id] = {
                "n_replicas": len([r for r in reps if r.state in
                                    (ReplicaState.READY, ReplicaState.STARTING)]),
                "avg_queue_occupancy": avg_occ,
                "queue_len": CFG.services_info[svc_id]["queue_len"],
                "rejection_rate": self._tick_rejected[svc_id] / total,
                "deadline_violation_rate": self._tick_violated[svc_id] / total,
                "recent_arrivals": self._tick_total[svc_id],
                "demand_centroid": None,  # *** ساده‌سازی فاز ۳: EMA مرکز ثقل پیاده نشده
            }
        snapshot["global"] = {"avg_response_time_recent": 0.0, "energy_recent_joule": 0.0,
                               "num_rejected_recent": sum(self._tick_rejected.values())}
        return snapshot

    async def decision_loop(self):
        while self._running:
            await asyncio.sleep(CFG.decision_interval_sec)
            snapshot = self._build_metrics_snapshot()
            now = time.monotonic()

            action = self.algorithm.provision_decision(self.servers, snapshot, now)
            if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
                s = self.servers[action.server_id]
                if s.state == ServerState.OFF and not s.in_cooldown(now, CFG.cooldown_sec):
                    await self._activate_server(action.server_id)
                    self.metrics.record_scale_action("TURN_ON")
            elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
                s = self.servers[action.server_id]
                n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
                if s.state == ServerState.ACTIVE and n_active > 1 and not s.in_cooldown(now, CFG.cooldown_sec):
                    if await self._drain_server(action.server_id):
                        self.metrics.record_scale_action("TURN_OFF")

            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)
                if decision == ScaleAction.SCALE_UP:
                    target = self.algorithm.select_placement_server(svc_id, self.servers)
                    if target is not None:
                        await self._create_replica(target, svc_id)
                        self.metrics.record_scale_action("SCALE_UP")
                elif decision == ScaleAction.SCALE_DOWN:
                    ready = [r for r in self.replicas_by_service.get(svc_id, [])
                             if r.state == ReplicaState.READY]
                    if len(ready) > 1:
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
                            svc_id, ready, self.servers, time.monotonic(),
                            occupancy_fn=lambda r: redis_state.get_queue_occupancy(svc_id, r.server_id))
                        
                        
                                      
                                      
                        asyncio.create_task(self._delete_replica(svc_id, victim.server_id))
                        self.metrics.record_scale_action("SCALE_DOWN")

            self._tick_total.clear()
            self._tick_rejected.clear()
            self._tick_violated.clear()

    # ------------------------------------------------------------------
    # dispatcher واقعی: replay با زمان‌بندی واقعی + HTTP واقعی
    # ------------------------------------------------------------------
    async def dispatch_loop(self, http_client: httpx.AsyncClient):
        base_time = float(self.events_df.global_start_sec.min())
        wall_start = time.monotonic()
        for row in self.events_df.itertuples(index=False):
            target_offset = float(row.global_start_sec) - base_time
            now_offset = time.monotonic() - wall_start
            if target_offset > now_offset:
                await asyncio.sleep(target_offset - now_offset)
            asyncio.create_task(self._handle_request(row, http_client))
        # صبر برای اتمام درخواست‌های درحال‌پرواز قبل از پایان کامل
        await asyncio.sleep(CFG.graceful_termination_delay_sec + 5)
        self._running = False

    async def _handle_request(self, row, http_client: httpx.AsyncClient):
        self._request_seq += 1
        req = Request(id=self._request_seq, bts_lat=row.Lat, bts_long=row.Long,
                       service_id=int(row.ServiceID), arrival_time=time.time())
        self._tick_total[req.service_id] += 1
        self._log("request_arrived", request_id=req.id, service_id=req.service_id)

        candidates = [r for r in self.replicas_by_service.get(req.service_id, []) if r.is_selectable()]
        chosen = self.algorithm.select_replica(req, candidates, self.servers, time.monotonic())

        if chosen is None:
            req.status = (RequestStatus.REJECTED_NO_REPLICA if not candidates
                          else RequestStatus.REJECTED_QUEUE_FULL)
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._log("request_rejected", request_id=req.id, service_id=req.service_id)
            self.metrics.record_request(req)
            return

        if not redis_state.try_reserve_queue_slot(req.service_id, chosen.server_id, chosen.queue_len):
            req.status = RequestStatus.REJECTED_QUEUE_FULL
            self._tick_rejected[req.service_id] += 1
            self._tick_violated[req.service_id] += 1
            self._log("request_rejected", request_id=req.id, service_id=req.service_id, reason="queue_full")
            self.metrics.record_request(req)
            return

        server = self.servers[chosen.server_id]
        distance_km = haversine_km(req.bts_lat, req.bts_long, server.lat, server.long)
        delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
        setattr(req, "_distance_km", distance_km)
        req.network_delay_ms = delay_ms
        req.assigned_server_id = server.id
        self._log("request_routed", request_id=req.id, server_id=server.id, distance_km=distance_km)

        ip = redis_state.get_pod_ip(req.service_id, chosen.server_id)
        deadline = CFG.services_info[req.service_id]["deadline"]
        t0 = time.monotonic()
        try:
            port = k8s_client.worker_port(req.service_id)
            resp = await http_client.post(f"http://{ip}:{port}/process", json={"request_id": req.id},
                                           timeout=deadline + 10)
            resp.raise_for_status()
            ok = True
        except Exception as e:
            ok = False
            self._log("request_http_error", request_id=req.id, error=str(e))
        finally:
            redis_state.release_queue_slot(req.service_id, chosen.server_id)

        response_time = (2 * delay_ms / 1000.0) + (time.monotonic() - t0)
        req.response_time_sec = response_time
        req.deadline_violated = (not ok) or (response_time > deadline)
        req.status = RequestStatus.COMPLETED if ok else RequestStatus.REJECTED_NO_REPLICA
        if req.deadline_violated:
            self._tick_violated[req.service_id] += 1
        self._log("request_completed" if ok else "request_failed", request_id=req.id,
                  service_id=req.service_id, response_time_sec=response_time)
        self.metrics.record_request(req)

    # ------------------------------------------------------------------
    async def run(self) -> dict:
        redis_state.reset_all(CFG.n_servers, CFG.n_services)
        await self.initial_placement()

        async with httpx.AsyncClient() as http_client:
            await asyncio.gather(self.decision_loop(), self.dispatch_loop(http_client))

        return self.metrics.finalize(self.servers)