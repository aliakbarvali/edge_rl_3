"""
k8s_adapter/realtime_dispatcher.py

 
"""

from __future__ import annotations
import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import json


from common.geo import haversine_km, network_delay_ms
from common.models import Server, Replica, ServerState, ReplicaState, Request, RequestStatus
from common.metrics import MetricsCollector
from common.logger import EventLogger
from common.config import CFG, compute_exec_time_sec
from algorithms.base import AlgorithmBase, ScaleAction, ProvisionActionType
from k8s_adapter import k8s_client, redis_state
 
UTIL_SAMPLE_INTERVAL_SEC = 5.0
RESERVATION_SWEEP_INTERVAL_SEC = 10.0
# مدت‌زمان صبر قبل از اولین sweep: کمی بیشتر از بیشینه‌ی deadline سرویس‌ها
# (نگاه کنید common/config.py:SERVICES_INFO) + حاشیه‌ی امن ۵ ثانیه که در
# route_request به‌عنوان ttl هر رزرو استفاده می‌شود، تا رزروهای کاملاً سالم
# زودتر از موعدشان جاروب نشوند.
                                 
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

        self._tick_total = defaultdict(int)
        self._tick_rejected = defaultdict(int)
        self._tick_violated = defaultdict(int)  
        self._tick_proximity_violated = defaultdict(int)
        # *** رفع باگ: قبلاً snapshot["global"]["avg_response_time_recent"] و
        # ["energy_recent_joule"] در این موتور همیشه هاردکد 0.0 بودند (هیچ‌جا
        # محاسبه نمی‌شدند) - یعنی دو بعد آخر بردار state (152بعدی) که PPO روی
        # آن‌ها آموزش دیده، در اجرای واقعی k8s همیشه صفر بودند: یک شکاف
        # توزیعی sim<->real در دقیقاً همان چیزی که مدل روی آن یاد گرفته.
        # این بافر مثل _tick_response_times در simulator/engine.py هر تیک
        # پر و در پایان هر تیک خالی می‌شود (نگاه کنید decision_loop پایین‌تر).
        self._tick_response_times: List[float] = []
        # هم‌راستا با simulator/engine.py:_energy_at_last_tick - دلتای انرژی
        # تجمعی بین دو تیک تصمیم را نگه می‌دارد.
        self._energy_at_last_tick: float = 0.0
        self._recent_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self._service_demand_centroid: Dict[int, Optional[Tuple[float, float]]] = {}
        
        
        self._util_at_window_start: Dict[int, float] = defaultdict(float)
        self._util_window_start_time: float = time.monotonic()

        self._service_last_scale_time: Dict[int, float] = {sid: -1e18 for sid in CFG.active_services}

        self._low_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}
        self._high_util_since: Dict[int, Optional[float]] = {sid: None for sid in self.servers}


    def _init_shadow_servers(self) -> Dict[int, Server]:
        servers = {}
        for sid, info in CFG.server_info.items():
            prof = CFG.server_profiles[info["profile"]]
            servers[sid] = Server(id=sid, profile=info["profile"], lat=info["lat"], long=info["long"],
                                   capacity=info["capacity_mips"], p_idle=prof["p_idle"], p_max=prof["p_max"])
        return servers

    def _log(self, event_type: str, **fields):
        if self.logger:
            self.logger.log(event_type, sim_time=time.time(), **fields)

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

        await self._wait_all_ready(timeout=CFG.pod_startup_delay_sec + 30)

    def _nearest_capable_server(self, service_id: int, candidate_ids: List[int]) -> Optional[int]:
        cpu = CFG.services_info[service_id]["resource_mips"]
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
                continue
            await self._delete_replica(service_id, server_id)

        if failed_services:
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

    async def _create_replica(self, server_id: int, service_id: int) -> Optional[Replica]:
        s = self.servers[server_id]
        cpu = CFG.services_info[service_id]["resource_mips"]
        if not s.can_host(service_id, cpu):
            return None
        svc = CFG.services_info[service_id]

        k8s_client.create_deployment(service_id, server_id)
        redis_state.set_replica_state(service_id, server_id, "STARTING")
        self.metrics.record_transition("pod_create")
        self._log("pod_create_started", server_id=server_id, service_id=service_id)

        r = Replica(service_id=service_id, server_id=server_id,
                    queue_len=svc["queue_len"], exec_time=compute_exec_time_sec(service_id, CFG.server_profiles[s.profile]["mips_per_core"]),
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

        svc = CFG.services_info[service_id]
        max_wait = svc["queue_len"] * r.exec_time + CFG.graceful_termination_delay_sec
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

    @staticmethod
    def _medoid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
      
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
        # *** رفع باگ بحرانی: PPOAlgorithm._build_action_masks (بدون .get،
        # بی‌قید‌وشرط) این سه کلید را از snapshot["servers"][sid] می‌خواند؛
        # نبودشان باعث KeyError واقعی در همان تیک تصمیم اول اجرای
        # `run.py --algorithm ppo --mode k8s` می‌شد (بازتولید و تأیید شد).
        # همین‌جا هم برای محاسبه‌ی is_last_active_server لازم است.
        n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)

        for sid, s in self.servers.items():
            if window_elapsed > 1e-9 and s.capacity > 0:
                avg_util = ((s.cumulative_busy_cpu_seconds - self._util_at_window_start[sid])
                            / (s.capacity * window_elapsed))
            else:
                avg_util = 0.0
            snapshot["servers"][sid] = {
                "state": s.state, "utilization": avg_util,
                "free_capacity": s.free_capacity(),
                "provision_cooldown_active": s.in_cooldown(now, CFG.cooldown_sec),
                "min_active_duration_met": (now - s.last_transition_time) >= CFG.min_active_duration_sec,
                "is_last_active_server": (s.state == ServerState.ACTIVE and n_active <= 1),
            }
        for svc_id in CFG.active_services:
            reps = self.replicas_by_service.get(svc_id, [])
            ready = [r for r in reps if r.state == ReplicaState.READY]
            mature_ready = [r for r in ready
                             if (time.monotonic() - r.created_at) >=
                             CFG.min_replica_age_before_scale_down_sec]
            total = max(self._tick_total[svc_id], 1)
            avg_occ = (sum(redis_state.get_queue_occupancy(svc_id, r.server_id) for r in ready)
                       / len(ready)) if ready else 0.0

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
                # *** رفع باگ: بدون این، PPOAlgorithm._build_action_masks با
                # snapshot["services"][sid].get("scale_cooldown_active", False)
                # همیشه False می‌گرفت - یعنی ماسک PPO در k8s واقعی هیچ‌وقت
                # SCALE_UP/DOWN را به‌خاطر cooldown غیرفعال نمی‌کرد (اجرای
                # واقعی decision_loop جداگانه cooldown را چک می‌کند و اکشن را
                # بی‌صدا/بدون لاگ رد می‌کند - یعنی مدل مدام اکشن‌هایی
                # "انتخاب" می‌کرد که دور ریخته می‌شدند، بدون هیچ سرنخ دیباگ).
                "scale_cooldown_active": (now - self._service_last_scale_time[svc_id]) < CFG.cooldown_sec,
            }

        # *** رفع باگ: قبلاً همیشه هاردکد 0.0 بود (نگاه کنید __init__ برای
        # توضیح کامل) - حالا هم‌راستا با simulator/engine.py:_build_metrics_snapshot
        # از میانگین واقعیِ پنجره‌ی جاری محاسبه می‌شود.
        current_energy = sum(s.cumulative_energy_joule for s in self.servers.values())
        snapshot["global"] = {
            "avg_response_time_recent": (sum(self._tick_response_times) / len(self._tick_response_times))
                                         if self._tick_response_times else 0.0,
            "energy_recent_joule": current_energy - self._energy_at_last_tick,
            "num_rejected_recent": sum(self._tick_rejected.values()),
        }
        self._energy_at_last_tick = current_energy
        return snapshot

    def _update_sustain_tracking(self, snapshot: dict, now: float):
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
        for svc_id in CFG.active_services:
            sv = snapshot["services"][svc_id]
            occ_ratio = (sv["avg_queue_occupancy"] / sv["queue_len"]) if sv["queue_len"] else 0.0
            necessary = (occ_ratio > CFG.decision_audit_scale_up_occ_threshold
                         or sv["rejection_rate"] > 0.0)
            if not necessary:
                continue
            cpu = CFG.services_info[svc_id]["resource_mips"]
            # *** رفع واگرایی sim/real: simulator/engine.py سرورهای BOOTING را
            # هم "قابل‌اتکا" حساب می‌کند (چون به‌زودی ACTIVE می‌شوند)؛ این
            # نسخه قبلاً فقط ACTIVE را می‌پذیرفت، یعنی وقتی سروری همین الان
            # داشت برای همین سرویس بوت می‌شد، سیستم همچنان "starved" تشخیص
            # می‌داد و می‌توانست TURN_ON اضافه‌ی غیرضروری صادر کند.
            if not any(s.state in (ServerState.ACTIVE, ServerState.BOOTING) and s.can_host(svc_id, cpu)
                       for s in self.servers.values()):
                return True
        return False

    async def decision_loop(self):
        while self._running:
            await asyncio.sleep(CFG.decision_interval_sec)
            snapshot = self._build_metrics_snapshot()
            now = time.monotonic()
            self._update_sustain_tracking(snapshot, now)

            action = self.algorithm.provision_decision(self.servers, snapshot, now)
            if action.action == ProvisionActionType.TURN_ON and action.server_id is not None:
                s = self.servers[action.server_id]
                turn_on_necessary = (self._any_active_server_sustained_overloaded(now)
                                      or self._any_service_capacity_starved(snapshot))
                if (s.state == ServerState.OFF and not s.in_cooldown(now, CFG.cooldown_sec)
                        and turn_on_necessary):
                    await self._activate_server(action.server_id)
                    self.metrics.record_scale_action("TURN_ON")
            elif action.action == ProvisionActionType.TURN_OFF and action.server_id is not None:
                s = self.servers[action.server_id]
                n_active = sum(1 for x in self.servers.values() if x.state == ServerState.ACTIVE)
                turn_off_necessary = self._was_turn_off_necessary(action.server_id, now)
                min_age_ok = (now - s.last_transition_time) >= CFG.min_active_duration_sec
                if (s.state == ServerState.ACTIVE and n_active > 1
                        and not s.in_cooldown(now, CFG.cooldown_sec)
                        and turn_off_necessary and min_age_ok):
                    if await self._drain_server(action.server_id):
                        self.metrics.record_scale_action("TURN_OFF")

            for svc_id in CFG.active_services:
                decision = self.algorithm.scale_decision(svc_id, snapshot)

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
                    mature = [r for r in ready
                              if (now - r.created_at) >= CFG.min_replica_age_before_scale_down_sec]
                    if len(ready) > 1 and mature:
                        victim = self.algorithm.select_scale_down_victim(
                            svc_id, mature, self.servers, now,
                            occupancy_fn=lambda r: redis_state.get_queue_occupancy(svc_id, r.server_id))
                        asyncio.create_task(self._delete_replica(svc_id, victim.server_id))
                        self.metrics.record_scale_action("SCALE_DOWN")
                        self._service_last_scale_time[svc_id] = now

            for sid, s in self.servers.items():
                self._util_at_window_start[sid] = s.cumulative_busy_cpu_seconds
            self._util_window_start_time = now
            
            self._tick_total.clear()
            self._tick_rejected.clear()
            self._tick_violated.clear()
            self._tick_proximity_violated.clear()
            self._tick_response_times.clear()


    async def route_request(self, request_id: int, service_id: int,
                             bts_lat: float, bts_long: float) -> dict:
      
        route_call_started_at = time.monotonic()
        self._tick_total[service_id] += 1
        self._log("request_arrived", request_id=request_id, service_id=service_id,
                  bts_lat=bts_lat, bts_long=bts_long)
        # *** رفع باگ: بدون این خط، self._recent_positions[service_id] همیشه
        # خالی می‌ماند و در نتیجه self._service_demand_centroid همیشه None
        # است (نگاه کنید _build_metrics_snapshot -> _medoid) - یعنی VOILA و
        # PPO (select_placement_server/migration_decision) در حالت k8s واقعی
        # هرگز از موقعیت واقعی ترافیک BTS استفاده نمی‌کردند و بی‌صدا به
        # fallback (مرکز سرورهای فعال) سقوط می‌کردند؛ درست برخلاف
        # simulator/engine.py که این خط را در _handle_arrival دارد.
        self._recent_positions[service_id].append((bts_lat, bts_long))

        deadline = CFG.services_info[service_id]["deadline"]
        reservation_ttl_sec = deadline + 5

        candidates = [r for r in self.replicas_by_service.get(service_id, []) if r.is_selectable()]

        request_obj = type("Request", (), {
            "bts_lat": bts_lat, 
            "bts_long": bts_long, 
            "service_id": service_id
        })()
        
        # *** رفع باگ (نشتی صف): request_id/ttl هم به رزرو صف پاس داده
        # می‌شود تا اگر پاد مقصد هرگز پاسخ نداد، sweep_expired_reservations
        # (نگاه کنید redis_state.py و _reservation_sweeper_loop پایین‌تر)
        # بتواند این رزرو را خودکار آزاد کند - قبلاً این رزرو تا ابد در
        # شمارنده‌ی صف باقی می‌ماند.
        chosen = self.algorithm.select_replica(
            request_obj,
            candidates, 
            self.servers, 
            time.monotonic(),
            # *** فیکس: occupancy_fn جدا و فقط-خواندنی از Redis؛ بدون این،
            # select_replica برای رتبه‌بندی/تخمین ترافیک به‌جای مقدار واقعی
            # صف، به deque محلی و همیشه-صفر Replica.queue_occupancy برمی‌گشت
            # (چون try_admit فقط در simulator صدا زده می‌شود، نه اینجا) -
            # یعنی «مسیریابی بر اساس فاصله و ترافیک» عملاً فقط فاصله بود.
            occupancy_fn=lambda r: redis_state.get_queue_occupancy(service_id, r.server_id),
            admit_fn=lambda r: redis_state.try_reserve_queue_slot(
                service_id, r.server_id, r.queue_len,
                request_id=request_id, ttl_sec=reservation_ttl_sec))

        if chosen is None:
            status = "REJECTED_NO_REPLICA" if not candidates else "REJECTED_QUEUE_FULL"
            self._tick_rejected[service_id] += 1
            self._tick_violated[service_id] += 1
            self._log("request_rejected", request_id=request_id, service_id=service_id,
                      reason="no_replica" if not candidates else "queue_full")
            # *** رفع باگ: بدون این، self.metrics (MetricsCollector) اصلاً از
            # وجود این درخواست رد‌شده خبردار نمی‌شد - record_request() فقط
            # از طریق record_external_completion (کامل‌شده‌ها، موفق/ناموفق
            # بعد از پردازش واقعی روی پاد) صدا زده می‌شد، نه از اینجا. نتیجه:
            # در گزارش نهایی k8s واقعی، num_requests_rejected_queue_full،
            # num_requests_rejected_no_replica و total_requests کمتر از
            # واقع بودند (این رد‌شده‌ها هرگز به هیچ پاد واقعی نمی‌رسند، پس
            # هیچ‌وقت از طریق drain_completion_queue هم گزارش نمی‌شدند) -
            # درست برخلاف simulator/engine.py که _finalize_request را برای
            # رد‌شده‌ها هم صدا می‌زند.
            rejected_req = Request(id=request_id, bts_lat=bts_lat, bts_long=bts_long,
                                    service_id=service_id, arrival_time=time.time())
            rejected_req.status = (RequestStatus.REJECTED_NO_REPLICA if not candidates
                                    else RequestStatus.REJECTED_QUEUE_FULL)
            self.metrics.record_request(rejected_req)
            return {"status": status}

        server = self.servers[chosen.server_id]
        
        distance_km = haversine_km(bts_lat, bts_long, server.lat, server.long)
        delay_ms = network_delay_ms(distance_km, CFG.base_latency_ms, CFG.k_ms_per_km)
        
        # *** هم‌راستا با رفع همین رگرسیون در simulator/engine.py: آستانه باید
        # روی تأخیر رفت‌وبرگشت (RTT) سنجیده شود، وگرنه با بیشینه‌ی فاصله‌ی
        # ممکن در این محدوده‌ی جغرافیایی (~182 کیلومتر -> ~5.6ms یک‌طرفه)
        # این شرط هرگز True نمی‌شود و proximity violation در حالت k8s واقعی
        # هم کاملاً خاموش می‌ماند.
        if 2 * delay_ms >= CFG.proximity_l0_ms:
            self._tick_proximity_violated[service_id] += 1
            
        self._log("request_routed", request_id=request_id, server_id=server.id, 
                  distance_km=distance_km, network_delay_ms=delay_ms)

        ip = redis_state.get_pod_ip(service_id, chosen.server_id)
        port = k8s_client.worker_port(service_id)

        routing_elapsed_sec = max(0.0, time.monotonic() - route_call_started_at)
        remaining_deadline = max(deadline - routing_elapsed_sec, 0.1)

        # *** توجه: رزرو صف با TTL ایمنی از قبل، هم‌زمان با try_reserve_queue_slot
        # در admit_fn بالا ثبت شد؛ فراخوانی جدا اینجا لازم نیست (set_reservation_ttl
        # قبلی کد مرده بود - نگاه کنید redis_state.py برای جزئیات رفع باگ).

        return {
            "status": "ROUTED",
            "server_id": server.id,
            "ip": ip,
            "port": port,
            "deadline_sec": remaining_deadline,
        }

    def record_external_completion(self, request_id: int, service_id: int, server_id: int,
                                    success: bool, response_time_sec: float):
        req = Request(id=request_id, bts_lat=0.0, bts_long=0.0, service_id=service_id,
                       arrival_time=time.time())
        req.response_time_sec = response_time_sec
        deadline = CFG.services_info[service_id]["deadline"]
        req.deadline_violated = (not success) or (response_time_sec > deadline)
        req.status = RequestStatus.COMPLETED if success else RequestStatus.REJECTED_NO_REPLICA
        if success:
            # *** برای snapshot["global"]["avg_response_time_recent"] - نگاه
            # کنید __init__ و _build_metrics_snapshot برای توضیح کامل باگ.
            self._tick_response_times.append(response_time_sec)
        if req.deadline_violated:
            self._tick_violated[service_id] += 1
        self._log("request_completed" if success else "request_failed", request_id=request_id,
                  service_id=service_id, server_id=server_id, response_time_sec=response_time_sec)
        self.metrics.record_request(req)

    async def drain_completion_queue(self):
        while self._running:
            batch = redis_state.pop_completion_batch(max_items=500)
            for item in batch:
                self.record_external_completion(
                    item["request_id"], item["service_id"], item["server_id"],
                    item["success"], item["response_time_sec"])
            await asyncio.sleep(0.2)

    
    
    async def run(self, extra_tasks: list | None = None) -> dict:
        redis_state.reset_all(CFG.n_servers, CFG.n_services)
        await self.initial_placement()
        self._util_window_start_time = time.monotonic()

        tasks = [
            self.decision_loop(),
            self.drain_completion_queue(),
            self._utilization_energy_sampler_loop(),
            self._reservation_sweeper_loop(),
        ]
        if extra_tasks:
            tasks.extend(extra_tasks)
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

    async def _utilization_energy_sampler_loop(self):
        last_sample = time.monotonic()
        while self._running:
            await asyncio.sleep(UTIL_SAMPLE_INTERVAL_SEC)
            now = time.monotonic()
            elapsed = now - last_sample
            last_sample = now

            for sid, s in self.servers.items():
                if s.state == ServerState.OFF:
                    continue

                busy_mips_seconds = 0.0
                if s.state in (ServerState.ACTIVE, ServerState.DRAINING):
                    for svc_id, r in s.hosted_replicas.items():
                        if r.state not in (ReplicaState.READY, ReplicaState.DRAINING):
                            continue
                        exact_busy_sec = redis_state.pop_busy_seconds_acc(svc_id, sid)
                        exact_busy_sec = min(exact_busy_sec, elapsed) 
                        busy_mips_seconds += exact_busy_sec * CFG.services_info[svc_id]["resource_mips"]

                s.cumulative_busy_cpu_seconds += busy_mips_seconds

                if s.state == ServerState.BOOTING:
                    power = s.p_idle
                elif s.state in (ServerState.ACTIVE, ServerState.DRAINING):
                    avg_util = (busy_mips_seconds / elapsed) / s.capacity if s.capacity > 0 and elapsed > 0 else 0.0
                    power = s.p_idle + (s.p_max - s.p_idle) * avg_util
                else:
                    power = 0.0
                s.cumulative_energy_joule += power * elapsed
                

    async def _reservation_sweeper_loop(self):
        """امن‌سازی در برابر نشتی صف: هر رزرو صفی که پاد مقصدش هرگز پاسخ
        نداد (کرش/شبکه قطع/timeout) را پیدا و آزاد می‌کند - نگاه کنید
        redis_state.sweep_expired_reservations برای توضیح کامل باگ رفع‌شده."""
        while self._running:
            await asyncio.sleep(RESERVATION_SWEEP_INTERVAL_SEC)
            released = redis_state.sweep_expired_reservations(CFG.n_servers, CFG.n_services)
            if released:
                self._log("reservation_sweep", released_count=released)


async def serve_control_plane(events_df, algorithm, algorithm_name, event_logger=None,
                                http_host="0.0.0.0", http_port=9000) -> dict:
    from k8s_adapter import dispatcher_api
    import uvicorn

    engine = RealtimeEngine(events_df, algorithm, algorithm_name, event_logger=event_logger)
    dispatcher_api.bind_engine(engine)

    config = uvicorn.Config(dispatcher_api.app, host=http_host, port=http_port, log_level="info")
    server = uvicorn.Server(config)

    async def _serve_and_shutdown():
        serve_task = asyncio.create_task(server.serve())
        while engine._running and not serve_task.done():
            await asyncio.sleep(1.0)
        if not serve_task.done():
            server.should_exit = True
            await serve_task

    return await engine.run(extra_tasks=[_serve_and_shutdown()])